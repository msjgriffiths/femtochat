module FemtoChat

include("parameters.jl")
include("kernels.jl")
include("gpt.jl")
include("tokenizers.jl")
include("dataset.jl")
include("dataloader.jl")

using .Parameters
using .GPT: initialize!, cross_entropy
using .Tokenizer
using .Dataset
using .DataLoading

export Params,
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
