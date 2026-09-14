module FemtoChatEnzymeExt

using Enzyme
using FemtoChat

import FemtoChat: loss_and_gradient!
import FemtoChat.Kernels: attention, attention_state, Δattention!
using FemtoChat.Parameters: Params, GPTConfig, 🤖

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
