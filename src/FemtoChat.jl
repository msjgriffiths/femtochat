module FemtoChat

include("parameters.jl")
include("optim.jl")
include("kernels.jl")
include("gpt.jl")
include("tokenizers.jl")
include("dataset.jl")
include("dataloader.jl")
include("generation.jl")

using .Parameters
using .Optimizer
using .GPT: initialize!, cross_entropy
using .Tokenizer
using .Dataset
using .DataLoading
using .Generation

function gpu_reference_loss end
function gpu_reference_gradient! end
function gpu_reference_state end
function loss_and_gradient! end
function gradient_state end

ℒ(Θ, config, layout, tokens, targets) =
    sum(🤖(Θ, config, layout)(tokens, targets))

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
       tokenize_documents,
       generate,
       ℒ,
       gradient_state,
       loss_and_gradient!,
       gpu_reference_state,
       gpu_reference_loss,
       gpu_reference_gradient!

end
