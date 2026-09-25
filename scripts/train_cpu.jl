# Start a native CPU REPL:
# julia -O1 -t auto --project=. -i scripts/train_cpu.jl
using Enzyme, Random
using FemtoChat: GPTConfig, parameter_layout, Params, initialize!,
                BPETokenizer, DataLoader, eachbatch, gradient_state, loss_and_gradient!
using FemtoChat: Tokenizer

function train_cpu(; steps=100, micro_steps=10, batch_size=32, η=.1f0, seed=123)
    @info "Native CPU training" steps micro_steps batch_size
    root = normpath(joinpath(@__DIR__, ".."))
    tokenizer = Tokenizer.load(BPETokenizer, joinpath(root, "my_tok.bin"))
    config = GPTConfig(sequence_len=4, vocab_size=length(tokenizer.vocab),
                       n_layer=2, n_head=2, n_kv_head=1, n_embed=32, window_pattern="L")
    layout = parameter_layout(config)
    params = Params(Vector{Float32}(undef, layout.nparams))
    initialize!(params, layout, MersenneTwister(seed))

    loader = DataLoader(joinpath(root, "data", "climbmix-400b-shuffle"), :train;
                        max_document_tokens=config.max_document_tokens)
    batches = eachbatch(loader, tokenizer, batch_size, config.sequence_len)
    ready = Channel(10, spawn=true) do queue
        for _ in 1:(steps * micro_steps)
            batch = first(batches)
            put!(queue, (; batch..., batch_tokens=count(!=(-1), batch.targets)))
        end
    end
    (; tokens, targets, positions, batch_tokens) = take!(ready)

    # Each call replaces the microbatch's mean gradient; accumulate token sums
    # separately, then divide once by the total number of valid targets.
    (; Θ, δ) = params
    micro_params = Params(Θ, similar(δ))
    @info "Preparing native forward and backward"
    state = gradient_state(micro_params, config, layout, tokens, targets; positions)

    losses = Float32[]
    for step in 1:steps
        fill!(δ, 0f0)
        ℒₜ = 0f0
        Nₜ = 0
        for micro_step in 1:micro_steps
            ℒₛ = loss_and_gradient!(micro_params, state, layout, tokens, targets; positions)
            δ .+= batch_tokens .* micro_params.δ
            ℒₜ += batch_tokens * ℒₛ
            Nₜ += batch_tokens
            if step < steps || micro_step < micro_steps
                (; tokens, targets, positions, batch_tokens) = take!(ready)
            end
        end
        δ ./= Nₜ
        Θ .-= η .* δ
        push!(losses, Float32(ℒₜ / Nₜ))
        @info (; step, ℒ=last(losses), tokens=Nₜ)
    end
    (; model=state.model.val, params, losses)
end

if !isinteractive() && abspath(PROGRAM_FILE) == @__FILE__
    train_cpu()
end
