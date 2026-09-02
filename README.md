# Femtochat

Femtochat is a copy of [nanochat](https://github.com/karpathy/nanochat) in Julia, trying to minimize dependencies. 

I picked the name `femtochat` because (a) `femto` means small (`nano` is `10^-9` and `femto` is `10^-15`) and (b) early version of Julia were built partially with [femtolisp](https://github.com/JeffBezanson/femtolisp) (_and it's still used in some places_).

The main dependencies are:

  * [Enzyme](https://github.com/EnzymeAD) for auto-differentiation. Enzyme (like tapanade) is a compiler-level autodiff engine, which supports custom rules (e.g. Flash Attention reverse)
  * [DuckDB](https://duckdb.org/) for handling Parquet files. 
  * [CUDA.jl](https://cuda.juliagpu.org/stable/) for running on GPUs.

### Environments

The Dell laptop uses a CUDA.jl release compatible with its older NVIDIA
driver, while development machines use the newest compatible CUDA 6 release.
Both environments use the same local FemtoChat package:

```sh
# Dell laptop: CUDA.jl 5.8.5
julia --project=environments/dell

# Development/production GPU: latest locked CUDA.jl 6.x
julia --project=environments/dev
```

Run `using Pkg; Pkg.instantiate()` once after selecting an environment. The
committed manifest then reproduces the exact versions tested for that target.

The Dell-compatible, full-device reference backward can be run with:

```sh
julia --project=environments/dell scripts/gpu_reference_train.jl
```

It uses direct CUDA.jl plus Enzyme's deferred device differentiation—no
KernelAbstractions. It intentionally runs the complete model in one GPU thread
as a correctness reference; it is not the eventual high-throughput backend.

We do steal a the `BinaryMaxHeap` data structure from [DataStructures.jl](https://juliacollections.github.io/DataStructures.jl/latest/) (+ inline it into codebase) because importing the full dependency for one 30-line implementation feels like a lot. In a "real" (non-toy) codebase we'd import the full dependency for flexibility. [DataStructures.jl](https://juliacollections.github.io/DataStructures.jl/latest/) is a great package.

Everything is coded "from scratch" in an effort to really understand the full (pre/mid/post-training) stack without layers of misdirection (e.g. PyTorch). I've used PyTorch since ~2019 (_since it was more popular on ASAPP's research team than Keras_) and it took me a long time to understand what was going on under the hood. 

Using Julia allows some of the benefits of Python (strong REPL, interactive experimentation) while eliminating the two-language problem. The "two-language problem" is IMO worst when you are trying to understand something, but it also comes in handy to avoid speed issues. [tokenizers.jl](https://github.com/msjgriffiths/femtochat/blob/main/src/tokenizers.jl) is _very slightly_ faster than [rustbpe](https://github.com/karpathy/rustbpe/blob/master/src/lib.rs), with similar lines of code. In that sense I think it's a nice language "for humans", even while languages like Rust seem more likely to win the LLM-based coding war.

## Notes:

  * Julia is columnar orientated, so the batch shape is different. We use `D x T x B` i.e. each token is a column, and then we group by batches. PyTorch tends to us `B x T x D` instead.
  * We add `parameters.jl` to keep track of parameters, which is something PyTorch handles in `nanochat`. We create a flat parameter vector and map the model into it; this allows us to accumulate gradients in a second flat parameter vector. 

## Example

### Tokenizer

```julia
using FemtoChat

tokenizer = BPETokenizer()
train!(tokenizer, 2^11,  "data/")

@info tokenizer("this is a test")
# [ Info: UInt16[0x0182, 0x0113, 0x0133, 0x0102, 0x046a]

@info tokenizer(UInt16[0x0182, 0x0113, 0x0133, 0x0102, 0x046a])
# [ Info: this is a test
```

### Model Loss

On CPU, the loss and reverse pass can be written out directly. Enzyme sees the
flat parameter vector `Θ` as active and accumulates its gradient into `δ`; it
does not need a second model.

```julia
using FemtoChat
using Enzyme
using Random

config = GPTConfig(
    sequence_len = 4,
    vocab_size = 8,
    n_layer = 1,
    n_head = 4,
    n_kv_head = 2,
    n_embed = 24,
    window_pattern = "L",
)

layout = parameter_layout(config)
params = Params(Vector{Float32}(undef, layout.nparams))
ℛ = MersenneTwister(42)
initialize!(params, layout, ℛ)

(; Θ, δ) = params

tokens = Int[1:4 2:5]
targets = Int[2:5 3:6]

# Make the parameter dependence explicit to Enzyme.
ℒ(Θ, config, layout, tokens, targets) =
    🤖(Θ, config, layout)(tokens, targets)

fill!(δ, 0f0)
Enzyme.API.strictAliasing!(false)
Enzyme.API.looseTypeAnalysis!(true)
mode = Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal)

_, ℒ₀ = Enzyme.autodiff(
    mode,
    ℒ,
    Enzyme.Active,
    Enzyme.Duplicated(params),
    Enzyme.Const(config),
    Enzyme.Const(layout),
    Enzyme.Const(tokens),
    Enzyme.Const(targets),
)

Θ .-= 0.01f0 .* δ
ℒ₁ = 🤖(params, config, layout)(tokens, targets)

@info (; ℒ₀, ℒ₁)
@assert ℒ₀ > ℒ₁
```

`loss_and_gradient!(params, config, layout, tokens, targets)` packages this
same block for use in a training loop. The first new model shape incurs Enzyme
compilation; subsequent calls reuse the compiled reverse pass.

### Dataset Loading

This standalone example follows nanochat's small CPU configuration: four
layers, a sequence length of 512, one document per device batch, and ten
training steps. Thus each step contains 512 tokens.

```julia
using FemtoChat
using Enzyme
using Random
using Iterators: take

directory = joinpath("data", "climbmix-400b-shuffle")
tokenizer = BPETokenizer()
train!(tokenizer, 2^9, joinpath(directory, "shard_00000.parquet"),)

config = GPTConfig(sequence_len = 512, vocab_size = length(tokenizer.vocab), n_layer = 4,n_head = 2,n_kv_head = 2,n_embed = 256,window_pattern = "SSSL",)

layout = parameter_layout(config)
params = Params(Vector{Float32}(undef, layout.nparams))
ℛ = MersenneTwister(42)
initialize!(params, layout, ℛ)

(; Θ, δ) = params

loader = DataLoader(directory, :train)
batches = eachbatch(loader, tokenizer, 1, config.sequence_len,)

η = 0.01f0
for (kₛ, (x̄, ȳ)) in enumerate(take(batches, 10))
    ℒₛ = loss_and_gradient!(params, config, layout, x̄, ȳ)

    Θ .-= η .* δ # Update model parameters with small step in gradient direction

    @info (; kₛ, ℒₛ)
end
```

### Dell GPU training

The Dell environment pins CUDA and Mooncake versions compatible with its GTX
1050. The training script reads shards `00000` through `00010`, logs metrics and
model artifacts to Weights & Biases using the credentials saved by `wandb
login`, and writes one local weight checkpoint after each shard.

```powershell
julia environments/dell/setup.jl
julia --project=environments/dell scripts/train_dell_gpu.jl
```

Set `FEMTOCHAT_WANDB=false` to train without W&B. Model dimensions, batch size,
and an optional step limit can be changed with the `FEMTOCHAT_N_LAYER`,
`FEMTOCHAT_N_EMBED`, `FEMTOCHAT_SEQUENCE_LEN`, `FEMTOCHAT_BATCH_SIZE`, and
`FEMTOCHAT_MAX_STEPS` environment variables.

Checkpoints contain the model configuration, training counters, completed
shard, and a CPU copy of the flat weight vector `Θ`. They intentionally omit
optimizer state.

```julia
using CUDA
using FemtoChat
using Serialization: deserialize

checkpoint = open(deserialize, "checkpoints/<run>/shard_00010.jls")
layout = parameter_layout(checkpoint.config)
params = Params(CuArray(checkpoint.Θ))
model = 🤖(params, checkpoint.config, layout)
```
