module Scaling

using ..Parameters: GPTConfig, parameter_layout, window_sizes

function num_scaling_params(layout)
    (; blocks, embedding) = layout.transformer
    n(p) = isnothing(p) ? 0 : length(p)

    wte = length(embedding)
    lm_head = length(layout.lm_head)
    value_embeds = 0
    transformer_matrices = 0
    scalars = length(layout.smear_gate) + length(layout.λₛ) + length(layout.λᵧ)

    for block in blocks
        value_embeds += n(block.🍰.𝔼)
        transformer_matrices += sum(n, values(block.👀)) + sum(n, values(block.🧠))
        scalars += n(block.λᵦ) + n(block.λx₀)
    end

    total = wte + value_embeds + lm_head + transformer_matrices + scalars

    return (; wte, value_embeds, lm_head, transformer_matrices, scalars, total)
end

function num_matmul_params(layout)
    counts = num_scaling_params(layout)
    return counts.transformer_matrices + counts.lm_head + length(layout.smear_gate)
end

function estimate_flops(config::GPTConfig, layout=parameter_layout(config))
    t = config.sequence_len
    attn_flops = 0

    for (window, _) in window_sizes(config)
        effective_seq = window < 0 ? t : min(window, t)
        attn_flops += 12 * config.n_embed * effective_seq
    end

    return 6 * num_matmul_params(layout) + attn_flops
end

end