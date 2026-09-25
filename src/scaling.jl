module Scaling

using ..Parameters: GPTConfig, parameter_layout, window_sizes

export num_scaling_params, num_matmul_params, estimate_flops, training_scaling, training_schedule

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

# Nanochat's reference is depth 12, width 768, and 2^19 tokens per step.
function training_scaling(config, total_batch_size=nothing; target_param_data_ratio=12, weight_decay=.28f0)
    counts = num_scaling_params(parameter_layout(config))
    reference = num_scaling_params(parameter_layout(GPTConfig(vocab_size=config.vocab_size)))
    D = floor(Int, target_param_data_ratio * (counts.transformer_matrices + counts.lm_head))
    Dᵣ = target_param_data_ratio * (reference.transformer_matrices + reference.lm_head)
    Bᵣ = 2^19
    B = isnothing(total_batch_size) ? 2^round(Int, log2(Bᵣ * (D / Dᵣ)^.383)) : total_batch_size
    η = Float32(√(B / Bᵣ))
    λ = Float32(weight_decay * η * Dᵣ / D)
    (; total_batch_size=B, target_tokens=D, η, λ)
end

# `step` counts completed updates, starting at zero, as in nanochat.
function training_schedule(step, steps; warmup_steps=40, warmdown_ratio=.65, final_lr_frac=.05)
    warmdown = round(Int, warmdown_ratio * steps)
    start = steps - warmdown
    η = if step < warmup_steps
        (step + 1) / warmup_steps
    elseif step <= start
        1
    else
        p = (steps - step) / warmdown
        p + (1 - p) * final_lr_frac
    end
    μ = if step < 400
        .85 + (.97 - .85) * step / 400
    elseif step >= start && warmdown > 0
        .97 + (.90 - .97) * (step - start) / warmdown
    else
        .97
    end
    (; η=Float32(η), μ=Float32(μ), λ=Float32((1 + cospi(step / steps)) / 2))
end

end
