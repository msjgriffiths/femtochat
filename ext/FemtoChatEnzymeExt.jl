module FemtoChatEnzymeExt

using Enzyme
using FemtoChat

import FemtoChat: loss_and_gradient!
import FemtoChat.Kernels: attention, flash_attention₁!, Δflash_attention₁!
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
    _, H, N, B = size(Q.val)
    O = similar(Q.val)
    dO = similar(Q.val)
    ℓ = similar(Q.val, 1, N, H, B)
    m = similar(Q.val, 1, N, H, B)
    fill!(dO, zero(F))

    flash_attention₁!(O, ℓ, m, Q.val, K.val, V.val, window.val)

    pQ = Enzyme.EnzymeRules.overwritten(config)[2] ? copy(Q.val) : Q.val
    pK = Enzyme.EnzymeRules.overwritten(config)[3] ? copy(K.val) : K.val
    pV = Enzyme.EnzymeRules.overwritten(config)[4] ? copy(V.val) : V.val
    primal = Enzyme.EnzymeRules.needs_primal(config) ? O : nothing
    tape = (; pQ, pK, pV, O, dO, ℓ, m)

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
    (; pQ, pK, pV, O, dO, ℓ, m) = tape
    Δflash_attention₁!(
        Q.dval,
        K.dval,
        V.dval,
        dO,
        pQ,
        pK,
        pV,
        O,
        ℓ,
        m,
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
