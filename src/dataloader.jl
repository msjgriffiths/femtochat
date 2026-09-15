module DataLoading

using DuckDB: DB, StreamResult
using DBInterface: connect, execute
using Tables: rows
using ..Dataset: MAX_SHARD
using ..Tokenizer: bos_token_id

export DataLoader,
       DataLoaderState,
       eachbatch,
       batch_state,
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

mutable struct DataLoader{C}
    files::Vector{String}
    rank::Int
    world_size::Int
    state::DataLoaderState
    connection::C
end

function DataLoader(
    files::AbstractVector{<:AbstractString};
    rank::Int = 0,
    world_size::Int = 1,
    state::DataLoaderState = DataLoaderState(),
    connection = connect(DB, ":memory:"),
)
    files = sort!(String.(files))
    files = files[rank+1:world_size:end]
    DataLoader(files, rank, world_size, state, connection)
end

function DataLoader(directory::AbstractString, split::Symbol; kwargs...)
    split in (:train, :test) ||
        throw(ArgumentError("split must be :train or :test"))

    test_shard = "shard_$(lpad(MAX_SHARD, 5, '0')).parquet"
    files = filter(readdir(directory; join=true)) do path
        endswith(path, ".parquet") &&
            (split == :test) == (basename(path) == test_shard)
    end

    DataLoader(files; kwargs...)
end

current_file(loader::DataLoader) = loader.files[loader.state.file]

"""
Stream documents from the loader's current file, starting at its saved row.

The result contains `text` and the zero-based Parquet `file_row_number`.
This function does not advance `loader.state`.
"""
function read_documents(loader::DataLoader)
    file = replace(current_file(loader), "'" => "''")
    row = loader.state.row - 1

    execute(
        loader.connection,
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
    map(text -> tokenize_document(tokenizer, text, bos), texts)
end

function tokenize_document(tokenizer, text, bos)
    tokens = tokenizer(text)
    pushfirst!(tokens, bos)
    tokens
end

struct DocumentIterator{L,T,I}
    loader::L
    tokenizer::T
    bos::I
end

"""Iterate forever over tokenized, BOS-prefixed documents."""
eachdocument(loader::DataLoader, tokenizer) =
    DocumentIterator(loader, tokenizer, bos_token_id(tokenizer))

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

mutable struct ActiveDocument{I}
    source::DataLoaderState
    tokens::Vector{I}
    position::Int
end

mutable struct BatchIterator{L,T,I}
    loader::L
    tokenizer::T
    bos::I
    sequence_len::Int
    slots::Vector{Union{Nothing,ActiveDocument{I}}}
    document_rows::Any
    row_state::Any
    exhausted::Bool
end

"""
Iterate over batches with at most one chunk from each document per batch.

Returns `(tokens, targets, positions, sources, epoch)` as a named tuple. The
three matrices have shape `(sequence_len, k)`; positions are zero-based within
each BOS-prefixed document. Padding has target `-1` and position `0`. A source
is the document's `DataLoaderState`, or `nothing` for an empty slot.

Each slot continues its document in the next batch. At an epoch boundary,
active documents finish before the next epoch starts; unused slots are padded.
Existing `(tokens, targets)` destructuring remains supported. Positions are
metadata for a future model change; they are not yet passed to RoPE.

Resume with `state=batch_state(previous_batches)`. The snapshot includes
active token vectors and requires the same files, rank, batch size, sequence
length, and tokenizer. `loader.state` alone only tracks document reading.
"""
function eachbatch(loader::DataLoader, tokenizer, k::Int, sequence_len::Int; state=nothing)
    k > 0 || throw(ArgumentError("batch size must be positive"))
    sequence_len > 0 || throw(ArgumentError("sequence length must be positive"))
    isempty(loader.files) && throw(ArgumentError("rank has no assigned files"))

    bos = bos_token_id(tokenizer)
    slots = Union{Nothing,ActiveDocument{typeof(bos)}}[nothing for _ in 1:k]
    batches = BatchIterator(loader, tokenizer, bos, sequence_len, slots, nothing, nothing, false)

    if !isnothing(state)
        (state.files, state.rank, state.world_size, state.sequence_len, length(state.slots)) ==
            (loader.files, loader.rank, loader.world_size, sequence_len, k) ||
            throw(ArgumentError("batch snapshot does not match this loader and batch shape"))
        batches.slots = deepcopy(state.slots)
        batches.exhausted = state.exhausted
        loader.state = state.reader
    end

    batches
end

"""Copy the next-read cursor and active documents; no live database iterator is saved."""
batch_state(batches::BatchIterator) = (
    files=copy(batches.loader.files),
    rank=batches.loader.rank,
    world_size=batches.loader.world_size,
    sequence_len=batches.sequence_len,
    reader=batches.loader.state,
    slots=deepcopy(batches.slots),
    exhausted=batches.exhausted,
)

function next_active_document!(batches::BatchIterator)
    loader = batches.loader
    while !batches.exhausted
        result = if isnothing(batches.document_rows)
            batches.document_rows = rows(read_documents(loader))
            iterate(batches.document_rows)
        else
            iterate(batches.document_rows, batches.row_state)
        end

        if isnothing(result)
            batches.document_rows = nothing
            batches.row_state = nothing
            if loader.state.file == length(loader.files)
                batches.exhausted = true
            else
                next_file!(loader)
            end
            continue
        end

        document, row_state = result
        (; file, epoch) = loader.state
        row = Int(document.file_row_number) + 1
        tokens = tokenize_document(batches.tokenizer, document.text, batches.bos)
        batches.row_state = row_state
        loader.state = DataLoaderState(file, row + 1, epoch)
        length(tokens) > 1 && return ActiveDocument(DataLoaderState(file, row, epoch), tokens, 0)
    end
    nothing
end

function model_batch!(batches::BatchIterator)
    (; slots, sequence_len, bos) = batches
    tokens = fill(Int(bos), sequence_len, length(slots))
    targets = fill(-1, sequence_len, length(slots))
    positions = zeros(Int, sequence_len, length(slots))
    sources = [isnothing(doc) ? nothing : doc.source for doc in slots]

    for (column, doc) in enumerate(slots)
        isnothing(doc) && continue
        p = doc.position
        n = min(sequence_len, length(doc.tokens) - 1 - p)
        @views tokens[1:n, column] .= doc.tokens[p+1:p+n]
        @views targets[1:n, column] .= doc.tokens[p+2:p+n+1]
        positions[1:n, column] .= p .+ (0:n-1)
        doc.position += n
    end

    (; tokens, targets, positions, sources, epoch=batches.loader.state.epoch)
end

function Base.iterate(batches::BatchIterator, ::Nothing=nothing)
    # A resumed cursor may already be at EOF. Try the next epoch as well,
    # but fail instead of spinning forever when all documents have no targets.
    for _ in 1:2
        for column in eachindex(batches.slots)
            doc = batches.slots[column]
            if !isnothing(doc) && doc.position == length(doc.tokens) - 1
                batches.slots[column] = nothing
            end
            if isnothing(batches.slots[column])
                batches.slots[column] = next_active_document!(batches)
            end
        end
        any(!isnothing, batches.slots) && return model_batch!(batches), nothing

        batches.loader.state = DataLoaderState(1, 1, batches.loader.state.epoch + 1)
        batches.document_rows = nothing
        batches.row_state = nothing
        batches.exhausted = false
    end
    throw(ArgumentError("assigned files contain no next-token targets"))
end

Base.IteratorSize(::Type{<:DocumentIterator}) = Base.IsInfinite()
Base.IteratorSize(::Type{<:BatchIterator}) = Base.IsInfinite()
Base.IteratorEltype(::Type{<:BatchIterator}) = Base.EltypeUnknown()

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
        document_rows = rows(read_documents(documents.loader))
        result = iterate(document_rows)
    end

    document, state = result
    (; file, epoch) = documents.loader.state
    tokens = tokenize_document(documents.tokenizer, document.text, documents.bos)
    documents.loader.state =
        DataLoaderState(file, Int(document.file_row_number) + 2, epoch)

    return tokens, (document_rows, state)
end

function Base.iterate(documents::DocumentIterator)
    document_rows = rows(read_documents(documents.loader))
    next_document(documents, document_rows, iterate(document_rows))
end

function Base.iterate(documents::DocumentIterator, state)
    document_rows, row_state = state
    next_document(documents, document_rows, iterate(document_rows, row_state))
end

end
