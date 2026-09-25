import FemtoChat
using FemtoChat: GPTConfig, parameter_layout, Params, initialize!, 🤖,
                BPETokenizer, DataLoader, eachbatch, batch_state, MuonAdamW
using FemtoChat: Tokenizer
using FemtoChat: training_scaling, training_schedule, estimate_flops, save_checkpoint,
                load_checkpoint, restore_optimizer!
using Reactant, Enzyme, CUDA, Random, SHA
using Enzyme: ReverseWithPrimal, Const, Active, Duplicated
using Reactant: @compile, to_rarray, ConcreteRArray, ReactantRNG
using Core: BFloat16

"""
Single-GPU base training. `total_batch_size` counts token slots per optimizer step;
gradients are normalized by actual non-padding targets when the optimizer reads
them; `params.δ` retains their unnormalized sum. `steps` overrides the
parameter/data horizon. Resume with the original model and schedule options.
Defaults match the RTX 4090 run: about 50M parameters with an 8,192-token
tokenizer, BF16 compute, and 32,768 token slots per update. `config` overrides
the model dimensions below; vocabulary size otherwise comes from the tokenizer.
Evaluations and FP8 are intentionally omitted.
"""
function train(;
    backend="gpu", steps=12_288, micro_steps=4, batch_size=4, compute_type=BFloat16,
    sequence_len=2048, n_layer=8, n_embed=512, n_head=8, n_kv_head=n_head,
    window_pattern="L", config=nothing,
    attention_options=(; backward=(; warps=Val(4), query_tile=Val(64))),
    total_batch_size=nothing, target_param_data_ratio=12, target_flops=nothing,
    warmup_steps=40, warmdown_ratio=.65, final_lr_frac=.05,
    embedding_lr=.3f0, unembedding_lr=.008f0, matrix_lr=.02f0,
    scalar_lr=.5f0, weight_decay=.28f0, seed=123,
    root=normpath(joinpath(@__DIR__, "..")),
    data=joinpath(root, "data", "climbmix-400b-shuffle"),
    tokenizer_path=joinpath(root, "my_tok.bin"),
    output=joinpath(root, "checkpoints", "train"), save_every=4096, resume=nothing,
    log_every=128, logger=identity, stop_after=nothing,
)
    Reactant.set_default_backend(backend)
    tokenizer = Tokenizer.load(BPETokenizer, tokenizer_path)
    tokenizer_hash = bytes2hex(sha256(read(tokenizer_path)))
    checkpoint = isnothing(resume) ? nothing : load_checkpoint(resume)
    saved = isnothing(checkpoint) ? nothing : checkpoint.metadata
    config = isnothing(config) ? GPTConfig(; sequence_len, vocab_size=length(tokenizer.vocab),
        n_layer, n_head, n_kv_head, n_embed, window_pattern) : config

    layout = parameter_layout(config)
    tokens_per_batch = batch_size * config.sequence_len
    if isnothing(total_batch_size) && !isnothing(micro_steps)
        total_batch_size = micro_steps * tokens_per_batch
    end
    scaling = training_scaling(config, total_batch_size; target_param_data_ratio, weight_decay)
    (; total_batch_size, target_tokens) = scaling
    total_batch_size % tokens_per_batch == 0 || error("total_batch_size must divide into whole microbatches")
    micro_steps = total_batch_size ÷ tokens_per_batch
    flops_per_token = estimate_flops(config, layout)
    steps = isnothing(steps) ? (isnothing(target_flops) ? max(1, target_tokens ÷ total_batch_size) :
        round(Int, target_flops / (flops_per_token * total_batch_size))) : steps
    options = (; steps, batch_size, micro_steps, total_batch_size, target_param_data_ratio,
        warmup_steps, warmdown_ratio, final_lr_frac, embedding_lr, unembedding_lr,
        matrix_lr, scalar_lr, weight_decay, seed, compute_type)
    if !isnothing(saved)
        all(getfield(saved.config, k) == getfield(config, k) for k in fieldnames(GPTConfig)) &&
            saved.options == options || error("Resume requires the original model and training schedule")
        saved.tokenizer_hash == tokenizer_hash || error("Checkpoint tokenizer differs")
    end

    𝓡 = Random.seed!(ReactantRNG(), seed)
    params = Params(ConcreteRArray{Float32}(undef, layout.nparams))
    if isnothing(saved)
        initialize!(params, layout, 𝓡)
    else
        params.Θ .= to_rarray(checkpoint.parameters)
        𝓡.seed .= to_rarray(saved.rng)
    end
    model = 🤖(params, config, layout; compute_type)
    (; Θ, δ) = params
    ω = MuonAdamW(params, layout; embedding_lr=embedding_lr * scaling.η,
        unembedding_lr=unembedding_lr * scaling.η, matrix_lr=matrix_lr * scaling.η,
        scalar_lr=scalar_lr * scaling.η, weight_decay=scaling.λ,
        t=to_rarray(0; track_numbers=Int), compute_type)
    isnothing(saved) || restore_optimizer!(ω, checkpoint.optimizer)

    loader = DataLoader(data, :train; max_document_tokens=config.max_document_tokens)
    if !isnothing(saved)
        basename.(loader.files) == basename.(saved.data_state.files) || error("Checkpoint data shards differ")
    end
    batches = eachbatch(loader, tokenizer, batch_size, config.sequence_len;
        state=isnothing(saved) ? nothing : saved.data_state)
    task = Ref{Task}()
    ready = Channel(2; spawn=true, taskref=task) do queue
        try
            for batch in batches
                # Save the cursor for THIS batch, not the producer's later position.
                put!(queue, (; batch..., Nₜ=count(!=(-1), batch.targets), state=batch_state(batches)))
            end
        catch err
            err isa InvalidStateException && !isopen(queue) || rethrow()
        finally
            close(loader)
        end
    end

    losses = Float32[]
    rows = NamedTuple[]
    step₀ = isnothing(saved) ? 0 : checkpoint.step
    smooth_loss = isnothing(saved) ? 0.0 : saved.smooth_loss
    training_seconds = isnothing(saved) ? 0.0 : saved.training_seconds
    trained_tokens = isnothing(saved) ? 0 : saved.trained_tokens
    last_step = isnothing(stop_after) ? steps : min(steps, stop_after)
    mkpath(output)
    try
        batch = take!(ready)
        tokens, targets, positions = to_rarray.((batch.tokens, batch.targets, batch.positions))
        extension = Base.get_extension(FemtoChat, :FemtoChatReactantExt)
        extension.prepare_attention(config, tokens; compute_type, attention_options...)

        started = time()
        ∇ℒ₀!, ∇ℒ! = let model=model, params=params, attention_options=attention_options
            ℒ = (tokens, targets, positions) -> model(tokens, targets; positions, reduction=:sum, attention_options)
            map((true, false)) do reset
                # Overwrite on the first microbatch; later calls accumulate.
                @compile sync=false ((tokens, targets, positions) -> begin
                    # Keep accumulation outside AD so XLA can reuse δ without copying it back.
                    δ₀ = reset ? nothing : copy(params.δ)
                    fill!(params.δ, 0f0)
                    result = Enzyme.autodiff(ReverseWithPrimal, Duplicated(ℒ, params), Active,
                        Const(tokens), Const(targets), Const(positions))
                    reset || (params.δ .+= δ₀)
                    result
                end)(tokens, targets, positions)
            end
        end
        η, μ, λ, Nₜ = to_rarray.((1f0, .85f0, scaling.λ, Float32(tokens_per_batch)); track_numbers=Number)
        update! = @compile sync=false ((η, μ, λ, Nₜ) -> ω(; η, μ, λ, Nₜ))(η, μ, λ, Nₜ)
        @info "Compiled training" seconds=time()-started parameters=layout.nparams compute_type micro_steps total_batch_size attention_options

        open(joinpath(output, "training-$(step₀).csv"), "w") do log
            for step in step₀+1:last_step
                Reactant.synchronize(Θ)
                started = time()
                ℒₜ = 0f0
                Nₜ = 0
                for micro_step in 1:micro_steps
                    if step != step₀+1 || micro_step != 1
                        batch = take!(ready)
                        tokens, targets, positions = to_rarray.((batch.tokens, batch.targets, batch.positions))
                    end
                    Nₜ += batch.Nₜ
                    _, ℒₛ = (micro_step == 1 ? ∇ℒ₀! : ∇ℒ!)(tokens, targets, positions)
                    ℒₜ += ℒₛ
                end
                Nₜ > 0 || error("Batch has no training targets")
                (; η, μ, λ) = training_schedule(step-1, steps; warmup_steps, warmdown_ratio, final_lr_frac)
                λ *= scaling.λ
                update!(to_rarray.((η, μ, λ, Float32(Nₜ)); track_numbers=Number)...)
                ℒₜ = Float32(ℒₜ) / Nₜ
                Reactant.synchronize(Θ)
                extension.ReactantCUDACall.check()
                seconds = time() - started
                isfinite(ℒₜ) || error("Non-finite loss at step $step")
                push!(losses, ℒₜ)
                smooth_loss = .9smooth_loss + .1ℒₜ
                trained_tokens += Nₜ
                training_seconds += seconds
                row = (; step, loss=ℒₜ, smooth_loss=smooth_loss / (1-.9^step),
                    seconds, tokens=Nₜ, slots=total_batch_size, tokens_per_second=Nₜ/seconds,
                    slots_per_second=total_batch_size/seconds,
                    tflops=flops_per_token*total_batch_size/seconds/1e12, η, μ, λ,
                    epoch=batch.epoch, trained_tokens, training_seconds)
                isempty(rows) && println(log, join(keys(row), ','))
                println(log, join(values(row), ',')); flush(log)
                push!(rows, row)
                if step % log_every == 0
                    @info (; step, ℒ=ℒₜ, η, seconds, tokens_per_second=row.tokens_per_second, epoch=batch.epoch)
                    logger(row)
                end
                if step == last_step || (save_every > 0 && step % save_every == 0)
                    path = save_checkpoint(output, step, Θ, ω,
                        (; config, options, tokenizer_hash, data_state=batch.state,
                         rng=Array(𝓡.seed), smooth_loss, training_seconds, trained_tokens))
                    @info "Saved checkpoint" path
                end
            end
        end
    finally
        close(ready)
        wait(task[])
    end
    (; model, params, optimizer=ω, losses, rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Run with --threads=6 and OPENBLAS_NUM_THREADS=1 on a modern NVIDIA GPU.
    train()
end
