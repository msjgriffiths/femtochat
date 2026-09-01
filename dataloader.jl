module DataLoading

using DuckDB: StreamResult
using DBInterface: execute
using Tables: rows
using ..Tokenizer: bos_token_id

export DataLoader,
       DataLoaderState,
       eachbatch,
       eachdocument,
       read_documents,
       tokenize_documents

"""
Position of the next document to read.

`file` indexes the loader's rank-local file list; `row` indexes a document
within that file. All fields use Julia's one-based indexing.
"""
struct DataLoaderState
    file::Int
    row::Int
    epoch::Int
end

DataLoaderState() = DataLoaderState(1, 1, 1)

mutable struct DataLoader
    files::Vector{String}
    rank::Int
    world_size::Int
    state::DataLoaderState
end

function DataLoader(
    files::AbstractVector{<:AbstractString};
    rank::Int = 0,
    world_size::Int = 1,
    state::DataLoaderState = DataLoaderState(),
)
    world_size > 0 || throw(ArgumentError("world_size must be positive"))
    0 <= rank < world_size ||
        throw(ArgumentError("rank must satisfy 0 ≤ rank < world_size"))

    files = sort!(String.(files))
    files = files[rank+1:world_size:end]
    isempty(files) && throw(ArgumentError("rank $rank has no assigned files"))

    1 <= state.file <= length(files) ||
        throw(ArgumentError("state.file is outside the assigned file list"))
    state.row > 0 || throw(ArgumentError("state.row must be positive"))
    state.epoch > 0 || throw(ArgumentError("state.epoch must be positive"))

    DataLoader(files, rank, world_size, state)
end

current_file(loader::DataLoader) = loader.files[loader.state.file]

"""
Stream documents from the loader's current file, starting at its saved row.

The result contains `text` and the zero-based Parquet `file_row_number`.
This function does not advance `loader.state`.
"""
function read_documents(loader::DataLoader, connection)
    file = replace(current_file(loader), "'" => "''")
    row = loader.state.row - 1

    execute(
        connection,
        """
        SELECT text, file_row_number
        FROM read_parquet('$file', file_row_number=true)
        WHERE file_row_number >= $row
        """,
        StreamResult,
    )
end

"""Tokenize a batch of documents and prepend BOS to each one."""
function tokenize_documents(tokenizer, texts)
    bos = bos_token_id(tokenizer)

    map(texts) do text
        tokens = tokenizer(text)
        pushfirst!(tokens, bos)
        tokens
    end
end

struct DocumentIterator{L,C,T,I}
    loader::L
    connection::C
    tokenizer::T
    bos::I
end

"""Iterate forever over tokenized, BOS-prefixed documents."""
eachdocument(loader::DataLoader, connection, tokenizer) =
    DocumentIterator(loader, connection, tokenizer, bos_token_id(tokenizer))

function model_batch(documents, sequence_len, bos)
    batch_size = length(documents)
    tokens = fill(Int(bos), sequence_len, batch_size)
    targets = fill(-1, sequence_len, batch_size)

    for (batch, document) in enumerate(documents)
        n = min(sequence_len, length(document) - 1)
        n == 0 && continue

        @views tokens[1:n, batch] .= document[1:n]
        @views targets[1:n, batch] .= document[2:n+1]
    end

    return tokens, targets
end

"""Iterate forever over model-ready batches of `k` documents."""
function eachbatch(loader::DataLoader, connection, tokenizer, k::Int, sequence_len::Int)
    k > 0 || throw(ArgumentError("batch size must be positive"))
    sequence_len > 0 || throw(ArgumentError("sequence length must be positive"))

    bos = bos_token_id(tokenizer)
    documents = eachdocument(loader, connection, tokenizer)
    batches = Iterators.partition(documents, k)

    (model_batch(batch, sequence_len, bos) for batch in batches)
end

Base.IteratorSize(::Type{<:DocumentIterator}) = Base.IsInfinite()

function next_file!(loader::DataLoader)
    (; file, epoch) = loader.state

    loader.state = if file == length(loader.files)
        DataLoaderState(1, 1, epoch + 1)
    else
        DataLoaderState(file + 1, 1, epoch)
    end
end

function next_document(documents::DocumentIterator, document_rows, result)
    while isnothing(result)
        next_file!(documents.loader)
        document_rows = rows(read_documents(documents.loader, documents.connection))
        result = iterate(document_rows)
    end

    document, state = result
    (; file, epoch) = documents.loader.state
    documents.loader.state =
        DataLoaderState(file, Int(document.file_row_number) + 2, epoch)

    tokens = documents.tokenizer(document.text)
    pushfirst!(tokens, documents.bos)

    return tokens, (document_rows, state)
end

function Base.iterate(documents::DocumentIterator)
    document_rows = rows(read_documents(documents.loader, documents.connection))
    next_document(documents, document_rows, iterate(document_rows))
end

function Base.iterate(documents::DocumentIterator, state)
    document_rows, row_state = state
    next_document(documents, document_rows, iterate(document_rows, row_state))
end

end
