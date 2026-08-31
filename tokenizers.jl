module Tokenizer

using DuckDB: DB, StreamResult, nextDataChunk
using DBInterface: connect, execute
using DataStructures: BinaryMaxHeap
using Tables

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

const Token = UInt16
const Pair  = UInt32

mutable struct Value
    ids::Vector{Token}
    frequency::UInt64
end

mutable struct MergeJob
    pair::Pair
    count::UInt64
    positions::Vector{UInt32}
end

Base.isless(a::MergeJob, b::MergeJob) =
    a.count == b.count ? a.pair > b.pair : a.count < b.count

@inline pair(a::Token, b::Token)::Pair =
    (Pair(a) << 16) | Pair(b)

@inline left(p::Pair)::Token  = Token(p >> 16)
@inline right(p::Pair)::Token = Token(p & 0xffff)

struct BPETokenizer
    vocab::Vector{Vector{UInt8}}
    merges::Dict{Pair,Token}
end

function count_partition!(counts, texts, partition, partition_count)
    for i in partition:partition_count:length(texts)
        for M in eachmatch(SPLIT_PATTERN, texts[i])
            piece = M.match
            counts[piece] = get(counts, piece, UInt64(0)) + one(UInt64)
        end
    end

    return nothing
end

function merge_pair!(value::Value, p::Pair, new_token::Token)
    a = left(p)
    b = right(p)
    ids = value.ids
    output = Vector{Token}(undef, length(ids))
    deltas = Tuple{Pair,Int8}[]
    sizehint!(deltas, 6)

    write = 0
    read = 1

    @inbounds while read <= length(ids)
        if read < length(ids) && ids[read] == a && ids[read + 1] == b
            if write > 0
                push!(deltas, (pair(output[write], a), Int8(-1)))
                push!(deltas, (pair(output[write], new_token), Int8(1)))
            end

            push!(deltas, (p, Int8(-1)))

            if read + 2 <= length(ids)
                push!(deltas, (pair(b, ids[read + 2]), Int8(-1)))
                push!(deltas, (pair(new_token, ids[read + 2]), Int8(1)))
            end

            write += 1
            output[write] = new_token
            read += 2
        else
            write += 1
            output[write] = ids[read]
            read += 1
        end
    end

    resize!(output, write)
    value.ids = output

    return deltas
end

function train!(T::BPETokenizer, target_vocab_size, directory::String)
    con = connect(DB, ":memory:")
    result = execute(
        con,
        "SELECT text FROM read_parquet('$(directory)')",
        StreamResult,
    );
    thread_count = Threads.nthreads()
    local_counts = [Dict{String,UInt64}() for _ in 1:thread_count]

    for chunk in Tables.partitions(result)
        texts = Tables.getcolumn(chunk.tbl, :text)
        Threads.@threads :static for partition in 1:thread_count
            count_partition!(
                local_counts[partition],
                texts,
                partition,
                thread_count,
            )
        end
    end

    counts = Dict{String,UInt64}()
    mergewith!(+, counts, local_counts...)

    # Create the byte vectors
    word_values = Vector{Value}(undef, length(counts))
    for (i, (piece, frequency)) in enumerate(counts)
        ids = Vector{Token}(undef, ncodeunits(piece))

        @inbounds for j in eachindex(ids)
            ids[j] = Token(codeunit(piece, j)) + one(Token)
        end

        word_values[i] = Value(ids, frequency)
    end

    pair_counts = Dict{Pair,UInt64}()
    pair_values = Dict{Pair,Vector{UInt32}}()
    
    for (value_id, value) in enumerate(word_values)
        ids = value.ids
        length(ids) < 2 && continue

        seen = Pair[]
        sizehint!(seen, length(ids) - 1)

        @inbounds for i in 1:length(ids)-1
            p = pair(ids[i], ids[i + 1])

            pair_counts[p] =
                get(pair_counts, p, UInt64(0)) + value.frequency

            if p ∉ seen
                push!(seen, p)
                push!(
                    get!(pair_values, p) do
                        UInt32[]
                    end,
                    UInt32(value_id),
                )
            end
        end
    end

    if isempty(T.vocab)
        sizehint!(T.vocab, typemax(Token))
        append!(T.vocab, [UInt8[b] for b in 0x00:0xff])
    end

    heap = BinaryMaxHeap{MergeJob}()
    for (p, positions) in pair_values
        push!(heap, MergeJob(p, pair_counts[p], positions))
    end
    empty!(pair_values)

    merges_to_do = max(target_vocab_size - length(T.vocab), 0)
    merges_done = 0
    last_percent = 0

    while length(T.vocab) < target_vocab_size
        isempty(heap) && break

        job = pop!(heap)
        best_pair = job.pair
        best_count = get(pair_counts, best_pair, UInt64(0))

        best_count == 0 && continue

        if job.count != best_count
            job.count = best_count
            push!(heap, job)
            continue
        end

        a = left(best_pair)
        b = right(best_pair)

        new_token = Token(length(T.vocab) + 1)

        T.merges[best_pair] = new_token
        push!(T.vocab, vcat(T.vocab[Int(a)], T.vocab[Int(b)]))

        new_positions = Dict{Pair,Vector{UInt32}}()

        # Only values known to contain this pair can change
        for value_id in job.positions
            value = word_values[Int(value_id)]
            ids = value.ids
            frequency = value.frequency

            # Heap positions can contain stale entries after earlier merges.
            found = false
            @inbounds for i in 1:length(ids)-1
                if ids[i] == a && ids[i+1] == b
                    found = true
                    break
                end
            end
            found || continue

            seen = Pair[]
            for (p, delta) in merge_pair!(value, best_pair, new_token)
                amount = frequency * UInt64(abs(delta))

                if delta < 0
                    pair_counts[p] -= amount
                else
                    pair_counts[p] = get(pair_counts, p, UInt64(0)) + amount

                    if p ∉ seen
                        push!(seen, p)
                        push!(
                            get!(new_positions, p) do
                                UInt32[]
                            end,
                            value_id,
                        )
                    end
                end
            end
        end

        for (p, positions) in new_positions
            count = get(pair_counts, p, UInt64(0))
            count > 0 && push!(heap, MergeJob(p, count, positions))
        end

        merges_done += 1
        percent = merges_to_do == 0 ? 100 : 100 * merges_done ÷ merges_to_do

        if percent > last_percent
            @info "BPE training" progress=percent merges=merges_done count=best_count
            last_percent = percent
        end
    end

    return nothing
end


end
