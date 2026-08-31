module Tokenizer

using DuckDB: DB, StreamResult, nextDataChunk
using DBInterface: connect, execute
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

mutable struct Value
    ids::Vector{UInt32}
    frequency::UInt64
end

const Token = UInt16
const Pair  = UInt32

struct BPETokenizer
    vocab::Vector{Vector{UInt8}}
    merges::Dict{Pair,Token}
end

function train!(T::BPETokenizer, directory::String)    
    con = connect(DB, ":memory:")
    result = DBInterface.execute(
        con,
        "SELECT text FROM read_parquet('$(directory)')",
        DuckDB.StreamResult,
    );
    locals = Dict(
        tid => Dict{String,UInt64}()
        for tid in 1:Threads.maxthreadid()
    )
    for chunk in Tables.partitions(result)
        texts = Tables.getcolumn(chunk.tbl, :text)
        Threads.@threads :static for text in texts
            d = locals[Threads.threadid()]
            for M in eachmatch(SPLIT_PATTERN, text)
                piece = M.match
                d[piece] = get(d, piece, 0) + 1
            end
        end
    end
    counts = Dict{String,UInt64}()
    mergewith!(+, counts, values(locals)...)
    

end


end 