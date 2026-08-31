# Femtochat

Femtochat is a copy of [nanochat](https://github.com/karpathy/nanochat) in Julia, trying to minimize dependencies. 

I picked the name `femtochat` because (a) `femto` means small (`nano` is `10^-9` and `femto` is `10^-15`) and (b) early version of Julia were built partially with [femtolisp](https://github.com/JeffBezanson/femtolisp) (_and it's still used in some places_).

The main dependencies are:

  * [Enzyme](https://github.com/EnzymeAD) for auto-differentiation. Enzyme (like tapanade) is a compiler-level autodiff engine, which supports custom rules (e.g. Flash Attention reverse)
  * [DuckDB](https://duckdb.org/) for handling Parquet files. 
  * [CUDA.jl](https://cuda.juliagpu.org/stable/) for running on GPUs.

We do steal a the `BinaryMaxHeap` data structure from [DataStructures.jl](https://juliacollections.github.io/DataStructures.jl/latest/) (+ inline it into codebase) because importing the full dependency for one 30-line implementation feels like a lot. In a "real" (non-toy) codebase we'd import the full dependency for flexibility. [DataStructures.jl](https://juliacollections.github.io/DataStructures.jl/latest/) is a great package.

Everything is coded "from scratch" in an effort to really understand the full (pre/mid/post-training) stack without layers of misdirection (e.g. PyTorch). I've used PyTorch since ~2019 (_since it was more popular on ASAPP's research team than Keras_) and it took me a long time to understand what was going on under the hood. 

Using Julia allows some of the benefits of Python (strong REPL, interactive experimentation) while eliminating the two-language problem. The "two-language problem" is IMO worst when you are trying to understand something, but it also comes in handy to avoid speed issues. [tokenizers.jl](https://github.com/msjgriffiths/femtochat/blob/main/tokenizers.jl) is _very slightly_ faster than [rustbpe](https://github.com/karpathy/rustbpe/blob/master/src/lib.rs), with similar lines of code. In that sense I think it's a nice language "for humans", even while languages like Rust seem more likely to win the LLM-based coding war. 

## Notes:

  * Julia is columnar orientated, so the batch shape is different. We use `D x T x B` i.e. each token is a column, and then we group by batches. PyTorch tends to us `B x T x D` instead.
  * We add `parameters.jl` to keep track of parameters, which is something PyTorch handles in `nanochat`. We create a flat parameter vector and map the model into it; this allows us to accumulate gradients in a second flat parameter vector. 

## Example

```julia
Enzyme.API.strictAliasing!(false)
Enzyme.API.looseTypeAnalysis!(true)

# Tiny model
config = GPTConfig(sequence_len = 4, vocab_size = 8, n_layer = 1, n_head = 4, n_kv_head = 2, n_embed = 24,window_pattern = "L",)

# Map layers onto parameter vector
layout = parameter_layout(config)

# params.Θ contains weights; params.δ receives gradients.
params = Params(Vector{Float32}(undef, layout.nparams))
GPT.initialize!(params, layout, MersenneTwister(123))

{; Θ, δ} = params
model = 🤖(Θ, config, layout)
shadow = 🤖(δ, config, layout)

fill!(δ, 0f0) # Clear gradients
foreach(buffer -> fill!(buffer, 0f0), shadow.rope_sin_cos) # Initialize shadow RoPE

tokens = Int[1:4 2:5]
targets = Int[2:5 3:6]

ℒ₀ = model(tokens, targets)

# Compute ∂loss/∂Θ, accumulating it into params.δ.
Enzyme.autodiff(
    Enzyme.Reverse,
    (m, x, y) -> m(x, y),
    Enzyme.Active,
    Enzyme.Duplicated(model, shadow),  
    Enzyme.Const(tokens),
    Enzyme.Const(targets),
)

# One plain SGD step.
learning_rate = 0.01f0
Θ .-= learning_rate .* δ

ℒ₁ = model(tokens, targets)

```