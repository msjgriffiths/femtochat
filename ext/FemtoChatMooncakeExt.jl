module FemtoChatMooncakeExt

using CUDA
using FemtoChat
using LinearAlgebra: mul!
using Mooncake

import FemtoChat: gradient_state, loss_and_gradient!
import FemtoChat.GPT: apply_rotary_embedding, cross_entropy, norm, rotary_factors, rotary_view
import FemtoChat.Kernels: attention, attention_state, Δattention!, cross_entropy_gradient!
using FemtoChat.Parameters: Embedding, GPTConfig, Linear, Params, paramview, 🤖
using Mooncake: CoDual,
                MinimalCtx,
                NoFData,
                NoRData,
                @is_primitive,
                arrayify,
                primal,
                tangent,
                zero_fcodual

model_loss(model, tokens, targets, positions) = model(tokens, targets; positions)

Mooncake.@zero_derivative MinimalCtx Tuple{typeof(rotary_factors), Tuple, Any}

struct MooncakeGradientState{M,C}
    model::M
    cache::C
end

function gradient_state(
    ::Type{Mooncake.DefaultCtx},
    params::Params{T,P},
    config::GPTConfig,
    layout,
    tokens,
    targets;
    positions=nothing,
) where {T,P<:CuArray}
    model = 🤖(params, config, layout)
    cache = prepare_gradient_cache(model_loss, model, tokens, targets, positions)
    MooncakeGradientState(model, cache)
end

tangent_fields(tangent) = tangent.fields

function flatten_gradient!(δ, gradient, layout)
    fill!(δ, 0f0)
    model = tangent_fields(gradient)
    transformer = tangent_fields(model.transformer)
    paramview(δ, layout.transformer.embedding) .=
        tangent_fields(transformer.embed).𝔼

    for (block, block_layout) in zip(transformer.blocks, layout.transformer.blocks)
        block = tangent_fields(block)
        attention = tangent_fields(block.👀)
        paramview(δ, block_layout.👀.𝕎) .= tangent_fields(attention.𝕎).𝕎
        paramview(δ, block_layout.👀.𝕂) .= tangent_fields(attention.𝕂).𝕎
        paramview(δ, block_layout.👀.𝕍) .= tangent_fields(attention.𝕍).𝕎
        paramview(δ, block_layout.👀.ℙ) .= tangent_fields(attention.ℙ).𝕎

        if !isnothing(block_layout.👀.𝕧𝕖)
            paramview(δ, block_layout.👀.𝕧𝕖) .=
                tangent_fields(attention.𝕧𝕖).𝕎
        end

        mlp = tangent_fields(block.🧠)
        paramview(δ, block_layout.🧠.𝔽) .= tangent_fields(mlp.𝔽).𝕎
        paramview(δ, block_layout.🧠.ℙ) .= tangent_fields(mlp.ℙ).𝕎

        if !isnothing(block_layout.🍰.𝔼)
            paramview(δ, block_layout.🍰.𝔼) .= tangent_fields(block.🍰).𝔼
        end

        paramview(δ, block_layout.λᵦ) .= block.λᵦ
        paramview(δ, block_layout.λx₀) .= block.λx₀
    end

    paramview(δ, layout.lm_head) .= tangent_fields(model.lm_head).𝕎
    paramview(δ, layout.smear_gate) .= tangent_fields(model.smear_gate).𝕎
    paramview(δ, layout.λₛ) .= model.λₛ
    paramview(δ, layout.λᵧ) .= model.λᵧ
    return nothing
end

function loss_and_gradient!(
    params::Params{T,P},
    state::MooncakeGradientState,
    layout,
    tokens,
    targets;
    positions=nothing,
) where {T,P<:CuArray}
    loss, gradients = value_and_gradient!!(
        state.cache,
        model_loss,
        state.model,
        tokens,
        targets,
        positions,
    )
    flatten_gradient!(params.δ, gradients[2], layout)
    return loss
end


@is_primitive MinimalCtx Tuple{
    Linear{T,A},
    CuArray{T,N},
} where {T<:AbstractFloat,A<:CuArray{T,2},N}

function linear_pullback!(dweights, dx, output_gradient, weights, x)
    X = reshape(x, size(x, 1), :)
    dX = reshape(dx, size(x, 1), :)
    dY = reshape(output_gradient, size(weights, 1), :)
    mul!(dweights, dY, X', one(eltype(x)), one(eltype(x)))
    mul!(dX, weights', dY, one(eltype(x)), one(eltype(x)))
    return nothing
end

function Mooncake.rrule!!(
    linear::CoDual{<:Linear},
    x::CoDual{<:CuArray{T,N},<:CuArray{T,N}},
) where {T<:AbstractFloat,N}
    primal_linear = primal(linear)
    primal_weights = primal_linear.𝕎
    weight_gradient = tangent(linear).data.𝕎
    primal_x, x_gradient = arrayify(x)
    result = zero_fcodual(primal_linear(primal_x))

    function linear_pullback(::NoRData)
        linear_pullback!(
            weight_gradient,
            x_gradient,
            tangent(result),
            primal_weights,
            primal_x,
        )
        return NoRData(), NoRData()
    end

    return result, linear_pullback
end

@is_primitive MinimalCtx Tuple{
    typeof(apply_rotary_embedding),
    CuArray{T,4},
    CuArray{T,N},
    CuArray{T,N},
} where {T<:AbstractFloat,N}

function rotary_pullback!(gradient, output_gradient, cos, sin)
    d = size(gradient, 1) ÷ 2
    T = size(gradient, 3)

    x₁ = @view gradient[1:d, :, :, :]
    x₂ = @view gradient[d+1:end, :, :, :]
    y₁ = @view output_gradient[1:d, :, :, :]
    y₂ = @view output_gradient[d+1:end, :, :, :]
    c = rotary_view(cos, T)
    s = rotary_view(sin, T)

    @. x₁ += y₁ * c - y₂ * s
    @. x₂ += y₁ * s + y₂ * c
    return nothing
end

function Mooncake.rrule!!(
    ::CoDual{typeof(apply_rotary_embedding)},
    x::CoDual{<:CuArray{T,4},<:CuArray{T,4}},
    cos::CoDual{<:CuArray{T,N},<:CuArray{T,N}},
    sin::CoDual{<:CuArray{T,N},<:CuArray{T,N}},
) where {T<:AbstractFloat,N}
    primal_x, gradient = arrayify(x)
    result = zero_fcodual(apply_rotary_embedding(
        primal_x,
        primal(cos),
        primal(sin),
    ))

    function rotary_pullback(::NoRData)
        rotary_pullback!(
            gradient,
            tangent(result),
            primal(cos),
            primal(sin),
        )
        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return result, rotary_pullback
end

@is_primitive MinimalCtx Tuple{
    typeof(norm),
    CuArray{T,N},
} where {T<:AbstractFloat,N}

function norm_pullback!(gradient, output_gradient, x)
    D = size(x, 1)
    scale = sqrt.(sum(abs2, x; dims=1) ./ D .+ eps(Float32))
    column_dot = sum(output_gradient .* x; dims=1)
    @. gradient += output_gradient / scale -
        x * column_dot / (D * scale^3)
    return nothing
end

function Mooncake.rrule!!(
    ::CoDual{typeof(norm)},
    x::CoDual{<:CuArray{T,N},<:CuArray{T,N}},
) where {T<:AbstractFloat,N}
    primal_x, gradient = arrayify(x)
    result = zero_fcodual(norm(primal_x))

    function norm_pullback(::NoRData)
        norm_pullback!(gradient, tangent(result), primal_x)
        return NoRData(), NoRData()
    end

    return result, norm_pullback
end

@is_primitive MinimalCtx Tuple{
    Embedding{T,A},
    CuArray{I,N},
} where {T<:AbstractFloat,A<:CuArray{T,2},I<:Integer,N}

function embedding_pullback_kernel!(dweights, output_gradient, tokens)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(output_gradient)
        D = size(dweights, 1)
        row = mod1(i, D)
        token_position = cld(i, D)
        token = @inbounds tokens[token_position]
        CUDA.@atomic dweights[row, token] += @inbounds output_gradient[i]
    end
    return nothing
end

function embedding_pullback!(dweights, output_gradient, tokens)
    threads = min(length(output_gradient), 256)
    blocks = cld(length(output_gradient), threads)
    @cuda threads=threads blocks=blocks embedding_pullback_kernel!(
        dweights,
        output_gradient,
        tokens,
    )
    return nothing
end

function Mooncake.rrule!!(
    embedding::CoDual{<:Embedding},
    tokens::CoDual{<:CuArray{I,N},NoFData},
) where {I<:Integer,N}
    primal_embedding = primal(embedding)
    weight_gradient = tangent(embedding).data.𝔼
    result = zero_fcodual(primal_embedding(primal(tokens)))

    function embedding_pullback(::NoRData)
        embedding_pullback!(
            weight_gradient,
            tangent(result),
            primal(tokens),
        )
        return NoRData(), NoRData()
    end

    return result, embedding_pullback
end

@is_primitive MinimalCtx Tuple{
    typeof(attention),
    CuArray{T,4},
    CuArray{T,4},
    CuArray{T,4},
    Tuple{Int,Int},
} where {T<:AbstractFloat}

function Mooncake.rrule!!(
    f::CoDual{typeof(attention)},
    Q::CoDual{<:CuArray{T,4},<:CuArray{T,4}},
    K::CoDual{<:CuArray{T,4},<:CuArray{T,4}},
    V::CoDual{<:CuArray{T,4},<:CuArray{T,4}},
    window::CoDual{<:Tuple{Int,Int},NoFData},
) where {T<:AbstractFloat}
    pQ, dQ = primal(Q), tangent(Q)
    pK, dK = primal(K), tangent(K)
    pV, dV = primal(V), tangent(V)
    pwindow = primal(window)

    saved = attention_state(primal(f),pQ,pK,pV,pwindow)
    result = zero_fcodual(saved.O)

    function attention_pullback(::NoRData)
        Δattention!(
            saved.𝒜,
            dQ,
            dK,
            dV,
            tangent(result),
            saved.Q,
            saved.K,
            saved.V,
            saved.O,
            saved.ℓ,
            saved.m,
            pwindow,
        )
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end

    return result, attention_pullback
end

@is_primitive MinimalCtx Tuple{
    typeof(cross_entropy),
    CuArray{T,3},
    CuArray{I,2},
    Int,
    Symbol,
} where {T<:AbstractFloat,I<:Integer}


function Mooncake.rrule!!(
    ::CoDual{typeof(cross_entropy)},
    logits::CoDual{<:CuArray{T,3},<:CuArray{T,3}},
    targets::CoDual{<:CuArray{I,2},NoFData},
    ignore_index::CoDual{Int,NoFData},
    reduction::CoDual{Symbol,NoFData},
) where {T<:AbstractFloat,I<:Integer}
    primal_logits, gradient = arrayify(logits)
    result = zero_fcodual(cross_entropy(
        primal_logits,
        primal(targets),
        primal(ignore_index),
        primal(reduction),
    ))

    function cross_entropy_pullback(dy)
        cross_entropy_gradient!(
            gradient,
            primal_logits,
            primal(targets),
            dy isa NoRData ? tangent(result) : dy,
            primal(ignore_index),
            primal(reduction),
        )
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end

    return result, cross_entropy_pullback
end

end
