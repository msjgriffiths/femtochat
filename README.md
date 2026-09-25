# Femtochat

Femtochat is a copy of [nanochat](https://github.com/karpathy/nanochat) in Julia, trying to minimize dependencies. 

I picked the name `femtochat` because (a) `femto` means small (`nano` is `10^-9` and `femto` is `10^-15`) and (b) early version of Julia were built partially with [femtolisp](https://github.com/JeffBezanson/femtolisp) (_and it's still used in some places_).

The main dependencies are:

  * [Enzyme](https://github.com/EnzymeAD) for auto-differentiation. Enzyme (like tapanade) is a compiler-level autodiff engine, which supports custom rules (e.g. Flash Attention reverse) as well as other languages like Rust.
  * [Mooncake](https://github.com/chalk-lab/Mooncake.jl) as an alternative autodiff backend because my 5-year old Dell laptop I'm prototyping this on uses an ancient `GTX
1050` that supports CUDA 11.1. Enzyme fails compilation. This requires quite a lot of additional backwards rules (which I've had Codex write). 
  * [Reactant.jl](https://enzymead.github.io/Reactant.jl/stable/) provides MLIR / XLA compilation, which is broadly similar to `torch.compile`. It provides a ~25% speed boost to plain Julia w/ Enzume. It remains about 5% slower than PyTorch. 
  * [DuckDB](https://duckdb.org/) for handling Parquet files. 
  * [CUDA.jl](https://cuda.juliagpu.org/stable/) for running on GPUs.

We do steal the `BinaryMaxHeap` data structure from [DataStructures.jl](https://juliacollections.github.io/DataStructures.jl/latest/) (+ inline it into codebase) because importing the full dependency for one 30-line implementation feels like a lot. In a "real" (non-toy) codebase we'd import the full dependency for flexibility. [DataStructures.jl](https://juliacollections.github.io/DataStructures.jl/latest/) is a great package.

Everything is coded "from scratch" in an effort to really understand the full (pre/mid/post-training) stack without layers of misdirection (e.g. PyTorch). I've used PyTorch since ~2019 (_since it was more popular on ASAPP's research team than Keras_) and it took me a long time to understand what was going on under the hood. 

Using Julia allows some of the benefits of Python (strong REPL, interactive experimentation) while eliminating the two-language problem. The "two-language problem" is IMO worst when you are trying to understand something, but it also comes in handy to avoid speed issues. [tokenizers.jl](https://github.com/msjgriffiths/femtochat/blob/main/src/tokenizers.jl) is _very slightly_ faster than [rustbpe](https://github.com/karpathy/rustbpe/blob/master/src/lib.rs), with similar lines of code. In that sense I think it's a nice language "for humans", even while languages like Rust seem more likely to win the LLM-based coding war.

## Notes:

  * Julia is columnar orientated, so the batch shape is different. We use `D x T x B` i.e. each token is a column, and then we group by batches. PyTorch tends to us `B x T x D` instead.
  * We add `parameters.jl` to keep track of parameters, which is something PyTorch handles in `nanochat`. We create a flat parameter vector and map the model into it (Θ); this allows us to accumulate gradients in a second flat parameter vector (δ). 

## Training in the REPL

From the project directory, install the declared dependencies once:

```powershell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then start a REPL with two CPU threads for data preparation and training:

```powershell
julia --project=. --threads=2 -i scripts/repl.jl
```

The helper loads the project and defines `train` without starting a run. Training
uses the existing `my_tok.bin` tokenizer and downloaded data in `data/`. On this
Windows laptop, use Reactant's CPU backend:

```julia
result = train(backend="cpu", steps=2, micro_steps=2, batch_size=2);
```

`train()` selects the GPU backend, which needs a supported Reactant GPU setup.
`LocalPreferences.toml` disables the Reactant and ReactantCUDAExt warm-up workloads to avoid
the Windows `jl_cstr_to_string` linker error. Package precompilation remains
enabled; the first training call still compiles the model.

On a modern NVIDIA GPU with the Reactant attention bridge built, the same loop
supports BF16 compute while retaining FP32 weights, accumulated gradients, and
optimizer moments:

```julia
using Core: BFloat16
include("scripts/train.jl")

result = train(compute_type=BFloat16, steps=100, batch_size=4, micro_steps=4,
               save_every=50, output="checkpoints/base");
```

The loop uses `MuonAdamW`, linear LR warmup/warmdown, and nanochat's momentum and
weight-decay schedules. `total_batch_size` is the number of token slots per update;
the gradient is divided by the number of non-padding targets actually consumed.
Set `steps=nothing` to derive the horizon from the parameter/data ratio (or
`target_flops`), and `micro_steps=nothing` for automatic total-batch scaling.

Checkpoints include weights, optimizer moments/counters, and the exact data cursor
including partially consumed documents. Resume with the same training options:

```julia
result = train(compute_type=BFloat16, steps=100, batch_size=4, micro_steps=4,
               save_every=50, output="checkpoints/resumed",
               resume="checkpoints/base/checkpoint_000050.jls");
```

Each run writes raw step metrics to `training-<start_step>.csv`; `logger=row -> ...`
can forward the same metrics to another logger. Evaluation, FP8, and distributed
training are not part of this loop yet.

### Native CUDA on the GTX 1050

Start a fresh Julia process using the Dell environment, which pins CUDA 5.8.5:

```powershell
julia -t auto --project=environments/dell -i scripts/train_cuda.jl
```

The script defines `train_cuda` without starting training in an interactive REPL:

```julia
result = train_cuda(steps=2, micro_steps=2, batch_size=2);
result.losses
```

This path uses Float32 CUDA arrays and Mooncake differentiation, including the
model's native CUDA attention kernels. It carries the loader's document positions
through RoPE and weights microbatch gradients by their valid-token counts.
Expect several minutes of compilation on the first training call. This uses the local Mooncake
checkout configured in `environments/dell`, the existing `my_tok.bin` tokenizer,
and downloaded Parquet data.

The root workspace installs Reactant for interactive use, while keeping it an
optional dependency of FemtoChat so the Dell environment can use older CUDA.

### Native CPU training

Start a CPU REPL from the project directory:

```powershell
julia -O1 -t auto --project=. -i scripts/train_cpu.jl
```

```julia
result = train_cpu(steps=2, micro_steps=2, batch_size=2);
result.losses
```

This uses ordinary Float32 arrays and Enzyme, with the same small model and
token-weighted microbatch accumulation as `train_cuda`. It prefetches CPU batches
and carries their document positions through RoPE. It loads neither CUDA nor
Reactant. The first training call compiles the derivative; subsequent batches
reuse it within that Julia process.

`-O1` requests less compiler optimization work. On this laptop, the small-model
first-gradient check took about 102 seconds with Enzyme at `-O1`, versus 232
seconds with default Enzyme settings and 340 seconds with Mooncake. These are
local first-use timings with package caches already populated; repeated Enzyme
gradients took about 2–3 milliseconds for a batch of two four-token sequences.

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

### Model

```julia
import FemtoChat
using FemtoChat: GPTConfig, parameter_layout, Params, initialize!, 🤖
using Reactant, Enzyme, CUDA, Random
using Enzyme: ReverseWithPrimal, Const, Active, Duplicated
using Reactant: @compile, to_rarray, ConcreteRArray, ReactantRNG

Reactant.set_default_backend("gpu")

let
    𝓡 = Random.seed!(ReactantRNG(), 123)
    config = GPTConfig(sequence_len=4, max_document_tokens=64, vocab_size=16, n_layer=2, n_head=2, n_kv_head=1, n_embed=32, window_pattern="L")
    layout = parameter_layout(config)
    # Create parameter and gradient vector on GPU
    params = Params(ConcreteRArray{Float32}(undef, layout.nparams))
    initialize!(params, layout, 𝓡)
    model = 🤖(params, config, layout)
    tokens, targets, positions = to_rarray.((Int32[1:4 2:5], Int32[2:5 3:6], Int32[0:3 8:11]))
    Nₜ = count(!=(-1), targets)

    # Register Julia implemtation of flash attention with Reactant
    extension = Base.get_extension(FemtoChat, :FemtoChatReactantExt)
    extension.prepare_attention(config, tokens)

    ℒ = (tokens, targets, positions) -> model(tokens, targets; positions, reduction=:sum)
    # Compile the forward and backward pass of the model w/ XLA + MLIR (like torch.compile)
    ∇ℒ! = @compile sync=true ((tokens, targets, positions) -> Enzyme.autodiff(ReverseWithPrimal, Duplicated(ℒ, params), Active, Const(tokens), Const(targets), Const(positions)))(tokens, targets, positions)

    η = .01f0
    (; Θ, δ) = params
    for step in 1:10
        fill!(δ, 0f0) # Zero out gradients
        _, ℒₛ = ∇ℒ!(tokens, targets, positions)
        δ ./= Nₜ # Divide gradient by tokens to get average
        Θ .-= η .* δ # Step in gradient direction

        @info (; step, ℒₛ=ℒₛ / Nₜ)
    end
end
```

`max_document_tokens` sizes the precomputed RoPE table. Give `DataLoader` the
same value with `max_document_tokens=config.max_document_tokens`; the loader cap
includes BOS. Its batches contain zero-based `positions` for each document,
including continuations in later batches. Move `batch.positions` to the device
along with `batch.tokens` and `batch.targets`, and pass them as above.

`eachbatch` packs successive documents into each column, separated by BOS,
and carries unfinished documents into the next batch. Positions restart at each
document; attention can cross BOS boundaries. Only exhausted streams at the end
of an epoch are padded. `batch_state(batches)` saves the reader and unfinished
documents for exact resume; older, unpacked-loader checkpoints cannot resume
with this packing policy.

The cached gradient API accepts `positions` in both
`gradient_state(params, config, layout, tokens, targets; positions)` and
`loss_and_gradient!(params, state, layout, tokens, targets; positions)`.
Reuse the state with new positions of the same shape and type for each batch.
Omitting positions starts every column at zero. Position tracking does not
carry attention context between batches.

