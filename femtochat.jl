module FemtoChat

include("parameters.jl")
include("optim.jl")
include("kernels.jl")
include("gpt.jl")
include("tokenizers.jl")
include("dataset.jl")
include("dataloader.jl")

using .Parameters
using .Optimizer
using .GPT: initialize!, cross_entropy
using .Tokenizer
using .Dataset
using .DataLoading

export Params,
       AdamW,
       Muon,
       MuonAdamW,
       polar_express,
       GPTConfig,
       parameter_layout,
       🤖,
       initialize!,
       cross_entropy,
       BPETokenizer,
       bos_token_id,
       train!,
       download_dataset!,
       DataLoader,
       DataLoaderState,
       eachbatch,
       eachdocument,
       read_documents,
       tokenize_documents

end
