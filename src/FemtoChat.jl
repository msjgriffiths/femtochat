module FemtoChat

include("parameters.jl")
include("optim.jl")
include("scaling.jl")
include("checkpoints.jl")
include("kernels.jl")
include("gpt.jl")
include("tokenizers.jl")
include("dataset.jl")
include("dataloader.jl")
include("generation.jl")
include("loss_eval.jl")
include("common.jl")

using .Parameters
using .Optimizer
using .Scaling
using .Checkpoints
using .GPT: initialize!, cross_entropy
using .Tokenizer
using .Dataset
using .DataLoading
using .Generation
using .Loss
using .Common


function loss_and_gradient! end
function gradient_state end

ℒ(Θ, config, layout, tokens, targets; positions=nothing) =
    🤖(Θ, config, layout)(tokens, targets; positions)

ℒ(ℳ::🤖, tokens, targets; positions=nothing) = ℳ(tokens, targets; positions)

export Params,
       AdamW,
       Muon,
       MuonAdamW,
       num_scaling_params,
       num_matmul_params,
       estimate_flops,
       training_scaling,
       training_schedule,
       save_checkpoint,
       load_checkpoint,
       optimizer_state,
       restore_optimizer!,
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
       batch_state,
       eachdocument,
       read_documents,
       tokenize_documents,
       generate,
       ℒ,
       gradient_state,
       loss_and_gradient!,
       evaluate_bpb,
       print_banner

end
