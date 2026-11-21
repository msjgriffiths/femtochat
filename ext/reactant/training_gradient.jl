using PrecompileTools: @compile_workload, @setup_workload

function compile_training_gradient(model, params, tokens, targets, positions; reset, attention_options)
    loss = (tokens, targets, positions) -> model(tokens, targets; positions, reduction=:sum, attention_options)
    function gradient!(tokens, targets, positions)
        # Accumulate outside AD so XLA can donate the gradient buffer.
        previous = reset ? nothing : copy(params.δ)
        fill!(params.δ, 0f0)
        result = Enzyme.autodiff(ReverseWithPrimal, Enzyme.Duplicated(loss, params), Enzyme.Active,
            Const(tokens), Const(targets), Const(positions))
        reset || (params.δ .+= previous)
        return result
    end
    return Reactant.@compile sync=false gradient!(tokens, targets, positions)
end

if FemtoChat._precompile_training_gradient
    @setup_workload begin
        Reactant.set_default_backend("cpu")
        config = FemtoChat.GPTConfig(sequence_len=4, max_document_tokens=16, vocab_size=16,
            n_layer=2, n_head=2, n_kv_head=2, n_embed=32, window_pattern="L")
        layout = FemtoChat.parameter_layout(config)
        weights = zeros(Float32, layout.nparams)
        params = FemtoChat.Params(Reactant.to_rarray(weights), Reactant.to_rarray(copy(weights)))
        rope = Reactant.to_rarray.(rotary_embeddings(Vector{Float32}, config.max_document_tokens,
            config.n_embed ÷ config.n_head))
        model = FemtoChat.🤖(params, config, layout; compute_type=Core.BFloat16, rope_sin_cos=rope)
        tokens, targets, positions = Reactant.to_rarray.((ones(Int,4,2), ones(Int,4,2), zeros(Int,4,2)))
        attention_options = (; backward=(; warps=Val(4), query_tile=Val(64)))
        @compile_workload begin
            compile_training_gradient(model, params, tokens, targets, positions; reset=true, attention_options)
        end
    end
end
