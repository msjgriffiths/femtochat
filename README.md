# Femtochat

Femtochat is a copy of [nanochat](https://github.com/karpathy/nanochat) in Julia, trying to minimize dependencies. 

I picked the name `femtochat` because (a) `femto` means small (`nano` is `10^-9` and `femto` is `10^-15`) and (b) early version of Julia were built partially with [femtolisp](https://github.com/JeffBezanson/femtolisp) (_and it's still used in some places_).

The main dependencies are:

  * [Enzyme](https://github.com/EnzymeAD) for auto-differentiation. Enzyme (like tapanade) is a compiler-level autodiff engine, which supports custom rules (e.g. Flash Attention reverse) as well as other languages like Rust.
  * [Mooncake](https://github.com/chalk-lab/Mooncake.jl) as an alternative autodiff backend because my 5-year old Dell laptop I'm prototyping this on uses an ancient `GTX
1050` that supports CUDA 11.1. Enzyme fails compilation. This requires quite a lot of additional backwards rules (which I've had Codex write). 
  * [DuckDB](https://duckdb.org/) for handling Parquet files. 
  * [CUDA.jl](https://cuda.juliagpu.org/stable/) for running on GPUs.

We do steal the `BinaryMaxHeap` data structure from [DataStructures.jl](https://juliacollections.github.io/DataStructures.jl/latest/) (+ inline it into codebase) because importing the full dependency for one 30-line implementation feels like a lot. In a "real" (non-toy) codebase we'd import the full dependency for flexibility. [DataStructures.jl](https://juliacollections.github.io/DataStructures.jl/latest/) is a great package.

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

### Dataset Loading

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
