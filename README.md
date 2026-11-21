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

### Precompiling the training gradient

The RunPod image compiles a small BF16 gradient on CPU while building the container. Julia saves the host method specializations reached by Reactant in the package cache. A fresh training process loads those methods from the depot and compiles the real graph for its shapes and GPU.

From the repository root, build the image for an x86-64 pod. Choose a CUDA version supported by the pod driver and the pinned Reactant build.

```sh
docker buildx build --platform linux/amd64 -f environments/runpod/Dockerfile \
    --build-arg CUDA_VERSION=12.9 -t "ghcr.io/msjgriffiths/femtochat:$(git rev-parse --short HEAD)" --push .
```

Set the RunPod template to the published image. Mount the tokenizer at `/opt/femtochat/my_tok.bin`, the training shards at `/opt/femtochat/data/climbmix-400b-shuffle`, and the checkpoints volume at `/opt/femtochat/checkpoints`, or pass those paths to `train` explicitly.

To enable the same workload in another build, set the following preference in the active project's `LocalPreferences.toml`.

```toml
[FemtoChat]
precompile_training_gradient = true
```

Run `julia --project=. -e 'using Pkg; Pkg.instantiate(); using FemtoChat, Enzyme, CUDA, Reactant'` during the image build. For an NVIDIA deployment, select the matching CUDA artifacts first and build the adapters in `deps/reactant` against the selected Reactant library. Retain the Julia depot and checkout paths in the container. The supplied Dockerfile sets `JULIA_CPU_TARGET=generic` to share the native host cache across compatible x86-64 CPUs.

`scripts/train.jl` reports the first synchronized gradient after attention setup and device input synchronization. The first real microbatch supplies that gradient. The trainer then compiles accumulation when needed and prepares the optimizer for the first update.

For a CPU comparison, run the command below in separate fresh processes with the workload preference set to `false` and `true`. Use different output filenames and compare them with `cmp` to check every parameter gradient. Pass depth `8` to exercise a depth outside the two-layer build workload.

```sh
OPENBLAS_NUM_THREADS=1 julia --project=. --startup-file=no --threads=1 \
    scripts/measure_first_gradient.jl 2 /tmp/gradient.f32 --check-accumulation
```
