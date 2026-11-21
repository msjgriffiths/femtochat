using FemtoChat, Enzyme, CUDA, Reactant, Random, LinearAlgebra, Statistics

length(ARGS) in (2, 3) || error("Usage: measure_first_gradient.jl DEPTH OUTPUT.f32 [--check-accumulation]")
check_accumulation = length(ARGS) == 3
check_accumulation && ARGS[3] != "--check-accumulation" && error("Unknown flag: $(ARGS[3])")
depth = parse(Int, ARGS[1])
depth > 0 || error("DEPTH must be positive")

if get(ENV, "FEMTOCHAT_BENCH_REACTANT_O0", "0") == "1"
    @eval Reactant Base.Experimental.@optlevel 0
end
Reactant.set_default_backend("cpu")
const extension = Base.get_extension(FemtoChat, :FemtoChatReactantExt)
const attention_options = (; backward=(; warps=Val(4), query_tile=Val(64)))

function prepare(depth)
    cfg = GPTConfig(sequence_len=4, max_document_tokens=64, vocab_size=16, n_layer=depth,
        n_head=2, n_kv_head=2, n_embed=32, window_pattern="L")
    layout = parameter_layout(cfg)
    host = Params(zeros(Float32, layout.nparams))
    rng = Xoshiro(123)
    initialize!(host, layout, rng)
    host.Θ .+= 0.002f0 .* randn(rng, Float32, length(host.Θ))
    params = Params(Reactant.to_rarray(host.Θ))
    model = FemtoChat.🤖(params, cfg, layout; compute_type=Core.BFloat16)
    tokens = Int[mod(t+b-2, 16)+1 for t in 1:4, b in 1:2]
    targets = Int[mod(t+b-1, 16)+1 for t in 1:4, b in 1:2]
    positions = Int[mod(t-1+(b-1)*8, 64) for t in 1:4, b in 1:2]
    batch = Reactant.to_rarray.((tokens, targets, positions))
    extension.prepare_attention(cfg, batch[1]; compute_type=Core.BFloat16, attention_options...)
    Reactant.synchronize((params.Θ, params.δ, model.rope_sin_cos..., batch...))
    return (; model, params, batch)
end

function compile_gradient(inputs, reset)
    return extension.compile_training_gradient(inputs.model, inputs.params, inputs.batch...;
        reset, attention_options)
end

function execute(executable, inputs)
    loss = Float32(last(executable(inputs.batch...)))
    Reactant.synchronize(inputs.params.δ)
    return loss
end

inputs = prepare(depth)
println("Ready at depth ", depth, " with workload preference ", FemtoChat._precompile_training_gradient)
flush(stdout)
built = @timed Base.invokelatest(compile_gradient, inputs, true)
ran = @timed Base.invokelatest(execute, built.value, inputs)
gradient = Array(inputs.params.δ)
@assert all(isfinite, gradient) && norm(gradient) > 0 && isfinite(ran.value)
println((; first_s=built.time + ran.time, compile_s=built.time, execute_sync_s=ran.time,
    sum_loss=ran.value, gradient_norm=norm(gradient), nparams=length(gradient)))
write(ARGS[2], gradient)
samples = [@elapsed execute(built.value, inputs) for _ in 1:38]
println((; warm_median_s=median(samples[4:end])))

if check_accumulation
    accumulate = compile_gradient(inputs, false)
    @assert execute(accumulate, inputs) == ran.value
    @assert Array(inputs.params.δ) == 2f0 .* gradient
    println("Exact accumulation across all ", length(gradient), " parameters")
end
