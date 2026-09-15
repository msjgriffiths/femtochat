module FemtoChatEnzymeExt

using Enzyme
using FemtoChat

import FemtoChat: loss_and_gradient!
import FemtoChat.Kernels: attention, attention_state, Δattention!
import FemtoChat.GPT: softcap, relu², norm, cross_entropy, apply_rotary_embedding
using FemtoChat.Parameters: Params, GPTConfig, Linear, 🤖
using LinearAlgebra: mul!

const Rules = Enzyme.EnzymeRules

function augmented_return(config, RT, y, dy, tape)
    Rules.augmented_rule_return_type(config, RT)(
        Rules.needs_primal(config) ? y : nothing,
        Rules.needs_shadow(config) ? dy : nothing, tape)
end

function Rules.augmented_primal(config::Rules.RevConfigWidth{1},
    linear::Duplicated{<:Linear}, ::Type{RT},
    x::Duplicated{<:AbstractArray}) where {RT<:Union{Duplicated,DuplicatedNoNeed}}
    y = linear.val(x.val)
    dy = zero(y)
    overwritten = Rules.overwritten(config)
    W = overwritten[1] ? copy(linear.val.𝕎) : linear.val.𝕎
    X = overwritten[2] ? copy(x.val) : x.val
    augmented_return(config, RT, y, dy, (; W, X, dy))
end

function Rules.reverse(::Rules.RevConfigWidth{1}, linear::Duplicated{<:Linear},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}}, tape, x::Duplicated)
    (; W, X, dy) = tape
    X = reshape(X, size(W,2), :)
    dX = reshape(x.dval, size(W,2), :)
    dY = reshape(dy, size(W,1), :)
    mul!(linear.dval.𝕎, dY, X', true, true)
    mul!(dX, W', dY, true, true)
    (nothing,)
end

function Rules.augmented_primal(config::Rules.RevConfigWidth{1},
    f::Const{typeof(apply_rotary_embedding)}, ::Type{RT},
    x::Duplicated{<:AbstractArray}, cos::Annotation, sin::Annotation) where {RT<:Union{Duplicated,DuplicatedNoNeed}}
    y = f.val(x.val, cos.val, sin.val)
    dy = zero(y)
    saved = Rules.overwritten(config)[2] ? copy(x.val) : x.val
    c = Rules.overwritten(config)[3] ? copy(cos.val) : cos.val
    s = Rules.overwritten(config)[4] ? copy(sin.val) : sin.val
    augmented_return(config, RT, y, dy, (; x=saved, c, s, dy))
end

function Rules.reverse(config::Rules.RevConfigWidth{1}, ::Const{typeof(apply_rotary_embedding)},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}}, tape,
    x::Duplicated, cos::Annotation, sin::Annotation)
    d, T = size(x.val,1) ÷ 2, size(x.val,3)
    (; dy) = tape
    @views begin
        c = reshape(tape.c[:,1:T], d,1,T,1)
        s = reshape(tape.s[:,1:T], d,1,T,1)
        if cos isa Duplicated && !(Rules.runtime_activity(config) && cos.val === cos.dval)
            cos.dval[:,1:T] .+= reshape(sum(dy[1:d,:,:,:] .* tape.x[1:d,:,:,:] .+
                dy[d+1:end,:,:,:] .* tape.x[d+1:end,:,:,:]; dims=(2,4)),d,T)
        end
        if sin isa Duplicated && !(Rules.runtime_activity(config) && sin.val === sin.dval)
            sin.dval[:,1:T] .+= reshape(sum(dy[1:d,:,:,:] .* tape.x[d+1:end,:,:,:] .-
                dy[d+1:end,:,:,:] .* tape.x[1:d,:,:,:]; dims=(2,4)),d,T)
        end
        x.dval[1:d,:,:,:] .+= dy[1:d,:,:,:] .* c .- dy[d+1:end,:,:,:] .* s
        x.dval[d+1:end,:,:,:] .+= dy[1:d,:,:,:] .* s .+ dy[d+1:end,:,:,:] .* c
    end
    nothing, nothing, nothing
end

# These mathematical boundaries also keep GPU broadcasts out of the AD tape.
function Rules.augmented_primal(config::Rules.RevConfigWidth{1},
    f::Const{typeof(softcap)}, ::Type{RT},
    x::Duplicated{<:AbstractArray}, c::Const) where {RT<:Union{Duplicated,DuplicatedNoNeed}}
    y = f.val(x.val, c.val)
    dy = zero(y)
    augmented_return(config, RT, y, dy, (; y, dy))
end

function Rules.reverse(::Rules.RevConfigWidth{1}, ::Const{typeof(softcap)},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}}, tape,
    x::Duplicated, c::Const)
    (; y, dy) = tape
    @. x.dval += dy * (1 - (y / c.val)^2)
    nothing, nothing
end

function Rules.augmented_primal(config::Rules.RevConfigWidth{1},
    f::Const{typeof(relu²)}, ::Type{RT},
    x::Duplicated{<:AbstractArray}) where {RT<:Union{Duplicated,DuplicatedNoNeed}}
    y = f.val(x.val)
    dy = zero(y)
    saved = Rules.overwritten(config)[2] ? copy(x.val) : x.val
    augmented_return(config, RT, y, dy, (; x=saved, dy))
end

function Rules.reverse(::Rules.RevConfigWidth{1}, ::Const{typeof(relu²)},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}}, tape, x::Duplicated)
    @. x.dval += tape.dy * 2 * max(tape.x, 0)
    (nothing,)
end

function Rules.augmented_primal(config::Rules.RevConfigWidth{1},
    ::Const{typeof(norm)}, ::Type{RT},
    x::Duplicated{<:AbstractArray}, ϵ::Const) where {RT<:Union{Duplicated,DuplicatedNoNeed}}
    D = size(x.val, 1)
    r = sum(abs2, x.val; dims=1)
    @. r = inv(sqrt(r / D + ϵ.val))
    y = x.val .* r
    dy = zero(y)
    saved = Rules.overwritten(config)[2] ? copy(x.val) : x.val
    augmented_return(config, RT, y, dy, (; x=saved, r, dy))
end

function Rules.reverse(::Rules.RevConfigWidth{1}, ::Const{typeof(norm)},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}}, tape,
    x::Duplicated, ϵ::Const)
    (; r, dy) = tape
    D = size(x.val, 1)
    dot = sum(dy .* tape.x; dims=1)
    @. x.dval += r * dy - tape.x * r^3 * dot / D
    nothing, nothing
end

function Rules.augmented_primal(config::Rules.RevConfigWidth{1},
    f::Const{typeof(cross_entropy)}, ::Type{RT},
    x::Duplicated{<:AbstractArray}, targets::Const, ignore::Const, reduction::Const) where {RT<:Union{Duplicated,DuplicatedNoNeed}}
    y = f.val(x.val, targets.val, ignore.val, reduction.val)
    dy = zero(y)
    saved = Rules.overwritten(config)[2] ? copy(x.val) : x.val
    augmented_return(config, RT, y, dy, (; x=saved, dy))
end

function Rules.reverse(::Rules.RevConfigWidth{1}, ::Const{typeof(cross_entropy)},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}}, tape,
    x::Duplicated, targets::Const, ignore::Const, reduction::Const)
    V, T, B = size(x.val)
    valid = targets.val .!= ignore.val
    dy = reduction.val == :mean ? tape.dy ./ sum(valid; dims=(1,2)) : tape.dy
    scale = reshape(dy, 1, size(dy)...)
    target = reshape(targets.val, 1, T, B)
    valid = reshape(valid, 1, T, B)
    vocabulary = reshape(1:V, V, 1, 1)
    maximum_logit = maximum(tape.x; dims=1)
    normalizer = sum(exp.(tape.x .- maximum_logit); dims=1)
    @. x.dval += ifelse(valid, scale *
        (exp(tape.x - maximum_logit) / normalizer - (vocabulary == target)), 0)
    nothing, nothing, nothing, nothing
end

Enzyme.Duplicated(params::Params) =
    Enzyme.Duplicated(params.Θ, params.δ)

loss(Θ, config, layout, tokens, targets) =
    sum(🤖(Θ, config, layout)(tokens, targets))

function Enzyme.EnzymeRules.augmented_primal(
    config::Enzyme.EnzymeRules.RevConfigWidth{1},
    ::Enzyme.Const{typeof(attention)},
    ::Type{<:Union{Enzyme.Duplicated,Enzyme.DuplicatedNoNeed}},
    Q::Enzyme.Duplicated{AQ},
    K::Enzyme.Duplicated{AK},
    V::Enzyme.Duplicated{AV},
    window::Enzyme.Const{Tuple{Int,Int}},
) where {
    F<:AbstractFloat,
    AQ<:AbstractArray{F,4},
    AK<:AbstractArray{F,4},
    AV<:AbstractArray{F,4},
}
    saved = attention_state(attention, Q.val, K.val, V.val, window.val)
    dO = similar(saved.O)
    fill!(dO, zero(eltype(dO)))

    # Device dispatch may already have made separate, rounded forward inputs.
    overwritten = Enzyme.EnzymeRules.overwritten(config)
    pQ = overwritten[2] && Base.mightalias(saved.Q, Q.val) ? copy(saved.Q) : saved.Q
    pK = overwritten[3] && Base.mightalias(saved.K, K.val) ? copy(saved.K) : saved.K
    pV = overwritten[4] && Base.mightalias(saved.V, V.val) ? copy(saved.V) : saved.V
    saved = (; saved..., Q=pQ, K=pK, V=pV)
    primal = Enzyme.EnzymeRules.needs_primal(config) ? saved.O : nothing
    tape = (; saved, dO)

    return Enzyme.EnzymeRules.AugmentedReturn(primal, dO, tape)
end

function Enzyme.EnzymeRules.reverse(
    ::Enzyme.EnzymeRules.RevConfigWidth{1},
    ::Enzyme.Const{typeof(attention)},
    ::Type{<:Union{Enzyme.Duplicated,Enzyme.DuplicatedNoNeed}},
    tape,
    Q::Enzyme.Duplicated,
    K::Enzyme.Duplicated,
    V::Enzyme.Duplicated,
    window::Enzyme.Const{Tuple{Int,Int}},
)
    (; saved, dO) = tape
    Δattention!(
        saved.𝒜,
        Q.dval,
        K.dval,
        V.dval,
        dO,
        saved.Q,
        saved.K,
        saved.V,
        saved.O,
        saved.ℓ,
        saved.m,
        window.val,
    )

    return nothing, nothing, nothing, nothing
end

"""
Compute the loss and accumulate its gradient directly into `params.δ`.

"""
function loss_and_gradient!(
    params::Params{T,P},
    config::GPTConfig,
    layout,
    tokens,
    targets,
) where {T,P<:Vector}
    fill!(params.δ, 0f0)

    Enzyme.API.strictAliasing!(false)
    Enzyme.API.looseTypeAnalysis!(true)
    mode = Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal)

    _, primal = Enzyme.autodiff(
        mode,
        loss,
        Enzyme.Active,
        Enzyme.Duplicated(params),
        Enzyme.Const(config),
        Enzyme.Const(layout),
        Enzyme.Const(tokens),
        Enzyme.Const(targets),
    )

    return primal
end

end
