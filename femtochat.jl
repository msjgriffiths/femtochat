module FemtoChat

include("parameters.jl")
include("kernels.jl")
include("gpt.jl")
include("tokenizers.jl")

using .Parameters
using .GPT: initialize!, cross_entropy
using .Tokenizer

export Params,
       GPTConfig,
       parameter_layout,
       🤖,
       initialize!,
       cross_entropy,
       BPETokenizer,
       train!

end