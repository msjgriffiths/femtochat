module FemtoChatEnzymeExt

using Enzyme
using FemtoChat

import FemtoChat: loss_and_gradient!, gradient_state
import FemtoChat.Kernels: attention, attention_state, Δattention!
import FemtoChat.GPT: softcap, relu², rescale, norm, cross_entropy, apply_rotary_embedding
import FemtoChat.GPT: value_residual!, residual_mix
import FemtoChat.GPT: embedding_gradient!, leading_channels, smear_input, smear
import FemtoChat.GPT: norm_state, norm_gradient!, cross_entropy_gradient!
using FemtoChat.Parameters: Params, GPTConfig, Linear, Embedding, MLP, 🤖
using LinearAlgebra: mul!

const Rules = Enzyme.EnzymeRules

model_loss(model, tokens, targets) = model(tokens, targets)

struct EnzymeGradientState{M,F,R}
    model::M
    forward::F
    reverse::R
end

"""Prepare Enzyme once; parameter tangents are views into `params.δ`."""
function gradient_state(params::Params{Float32}, config::GPTConfig, layout, tokens, targets)
    model = 🤖(params,config,layout)
    # Shared read-only RoPE buffers are inactive under Enzyme's runtime activity.
    shadow = 🤖(params.δ,config,layout; rope_sin_cos=model.rope_sin_cos)
    model = Duplicated(model,shadow)
    mode = Enzyme.set_runtime_activity(Enzyme.ReverseSplitWithPrimal)
    forward, reverse = Enzyme.autodiff_thunk(mode,Const{typeof(model_loss)},Duplicated,
        typeof(model),typeof(Const(tokens)),typeof(Const(targets)))
    EnzymeGradientState(model,forward,reverse)
end

function loss_and_gradient!(params::Params, state::EnzymeGradientState, layout, tokens, targets)
    fill!(params.δ,0f0)
    args = (Const(model_loss),state.model,Const(tokens),Const(targets))
    tape, loss, dloss = state.forward(args...)
    fill!(dloss,one(eltype(dloss)))
    state.reverse(args...,tape)
    # Only the scalar metric leaves the GPU, after differentiation has finished.
    sum(loss)
end

function augmented_return(config, RT, y, dy, tape)
    Rules.augmented_rule_return_type(config, RT)(
        Rules.needs_primal(config) ? y : nothing,
        Rules.needs_shadow(config) ? dy : nothing, tape)
end

# Reshape aliases values and gradients. Construct both aliases normally so GPU
# reference counts are retained; there is no numerical work to undo in reverse.
function Rules.augmented_primal(config::Rules.RevConfigWidth{1},
    ::Const{typeof(reshape)}, ::Type{RT}, x::Duplicated{<:AbstractArray}, dims::Const...) where {RT<:Union{Duplicated,DuplicatedNoNeed}}
    sizes = map(d -> d.val,dims)
    y = reshape(x.val,sizes...)
    dy = Rules.runtime_activity(config) && x.val === x.dval ? y : reshape(x.dval,sizes...)
    augmented_return(config,RT,y,dy,nothing)
end

function Rules.reverse(::Rules.RevConfigWidth{1}, ::Const{typeof(reshape)},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}}, ::Nothing,
    x::Duplicated{<:AbstractArray}, dims::Const...)
    (nothing,map(_ -> nothing,dims)...)
end

function Rules.augmented_primal(config::Rules.RevConfigWidth{1},
    f::Const{typeof(value_residual!)}, ::Type{<:Const},
    V::Duplicated, gate::Duplicated, VE::Duplicated)
    G = Rules.overwritten(config)[3] ? copy(gate.val) : gate.val
    E = Rules.overwritten(config)[4] ? copy(VE.val) : VE.val
    f.val(V.val,gate.val,VE.val)
    Rules.AugmentedReturn(nothing,nothing,(; G,E))
end

function Rules.reverse(::Rules.RevConfigWidth{1}, ::Const{typeof(value_residual!)},
    ::Type{<:Const}, tape, V::Duplicated, gate::Duplicated, VE::Duplicated)
    # V's derivative is the identity: retain its incoming cotangent.
    gate.dval .+= mapreduce(*,+,V.dval,tape.E; dims=1)
    VE.dval .+= V.dval .* tape.G
    nothing,nothing,nothing
end

function Rules.augmented_primal(config::Rules.RevConfigWidth{1},
    f::Const{typeof(residual_mix)}, ::Type{RT}, x::Duplicated, y::Duplicated,
    α::Union{Const,Duplicated}, β::Union{Const,Duplicated}) where {RT<:Union{Duplicated,DuplicatedNoNeed}}
    z = f.val(x.val,y.val,α.val,β.val)
    dz = zero(z)
    saved = map((x,overwritten) -> overwritten ? copy(x.val) : x.val,
        (x,y,α,β),Rules.overwritten(config)[2:5])
    augmented_return(config,RT,z,dz,(;saved,dz))
end

function Rules.reverse(::Rules.RevConfigWidth{1}, ::Const{typeof(residual_mix)},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}}, tape,
    x::Duplicated, y::Duplicated, α::Union{Const,Duplicated}, β::Union{Const,Duplicated})
    X,Y,a,b = tape.saved
    (;dz) = tape
    @. x.dval += a*dz
    @. y.dval += b*dz
    # Learned coefficients are one-element arrays. Reduce their gradients once,
    # instead of making every activation atomically update the same scalar.
    dims = ntuple(identity,ndims(dz))
    α isa Duplicated && (α.dval .+= reshape(mapreduce(*,+,dz,X;dims),size(α.dval)))
    β isa Duplicated && (β.dval .+= reshape(mapreduce(*,+,dz,Y;dims),size(β.dval)))
    nothing,nothing,nothing,nothing
end

function Rules.augmented_primal(config::Rules.RevConfigWidth{1},
    embedding::Duplicated{<:Embedding}, ::Type{RT}, tokens::Const) where {RT<:Union{Duplicated,DuplicatedNoNeed}}
    y = embedding.val(tokens.val)
    dy = zero(y)
    augmented_return(config,RT,y,dy,dy)
end

function Rules.reverse(::Rules.RevConfigWidth{1}, embedding::Duplicated{<:Embedding},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}}, dy, tokens::Const)
    embedding_gradient!(embedding.dval.𝔼,dy,tokens.val)
    (nothing,)
end

input_region(::typeof(leading_channels), x, n) = @view x[1:n,:,:]
input_region(::typeof(smear_input), x) = @view x[1:24,2:end,:]

function Rules.augmented_primal(config::Rules.RevConfigWidth{1},
    f::Const{F}, ::Type{RT}, x::Duplicated{<:AbstractArray}, args::Const...) where {
    F<:Union{typeof(leading_channels),typeof(smear_input)}, RT<:Union{Duplicated,DuplicatedNoNeed}}
    y = f.val(x.val,map(a -> a.val,args)...)
    dy = zero(y)
    augmented_return(config,RT,y,dy,dy)
end

function Rules.reverse(::Rules.RevConfigWidth{1}, f::Const{F},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}}, dy, x::Duplicated, args::Const...) where {
    F<:Union{typeof(leading_channels),typeof(smear_input)}}
    region = input_region(f.val,x.dval,map(a -> a.val,args)...)
    region .+= dy
    (nothing, map(_ -> nothing,args)...)
end

function Rules.augmented_primal(config::Rules.RevConfigWidth{1},
    f::Const{typeof(smear)}, ::Type{RT}, x::Duplicated, gate::Duplicated) where {RT<:Union{Duplicated,DuplicatedNoNeed}}
    y = f.val(x.val,gate.val)
    dy = zero(y)
    X = Rules.overwritten(config)[2] ? copy(x.val) : x.val
    G = Rules.overwritten(config)[3] ? copy(gate.val) : gate.val
    augmented_return(config,RT,y,dy,(; X,G,dy))
end

function Rules.reverse(::Rules.RevConfigWidth{1}, ::Const{typeof(smear)},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}}, tape, x::Duplicated, gate::Duplicated)
    (; X,G,dy) = tape
    x.dval .+= dy
    @views begin
        x.dval[:,1:end-1,:] .+= dy[:,2:end,:] .* G
        gate.dval .+= mapreduce(*,+,dy[:,2:end,:],X[:,1:end-1,:]; dims=1)
    end
    nothing,nothing
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

# The MLP owns u/a: retain them directly, while respecting mutation of its inputs.
function Rules.augmented_primal(config::Rules.RevConfigWidth{1},
    m::Duplicated{<:MLP}, ::Type{RT},
    x::Duplicated{<:AbstractArray}) where {RT<:Union{Duplicated,DuplicatedNoNeed}}
    u = m.val.𝔽(x.val)
    a = relu²(u)
    y = m.val.ℙ(a)
    dy = zero(y)
    overwritten = Rules.overwritten(config)
    W₁ = overwritten[1] ? copy(m.val.𝔽.𝕎) : m.val.𝔽.𝕎
    W₂ = overwritten[1] ? copy(m.val.ℙ.𝕎) : m.val.ℙ.𝕎
    X = overwritten[2] ? copy(x.val) : x.val
    augmented_return(config,RT,y,dy,(;W₁,W₂,X,u,a,dy))
end

function Rules.reverse(::Rules.RevConfigWidth{1}, m::Duplicated{<:MLP},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}}, tape, x::Duplicated)
    (;W₁,W₂,X,u,a,dy) = tape
    X = reshape(X,size(W₁,2),:)
    A = reshape(a,size(W₂,2),:)
    dY = reshape(dy,size(W₂,1),:)
    du = similar(u) # Private scratch: its first writer assigns, not adds.
    dU = reshape(du,size(W₁,1),:)
    mul!(m.dval.ℙ.𝕎,dY,A',true,true)
    mul!(dU,W₂',dY)
    @. du *= 2max(u,0)
    mul!(m.dval.𝔽.𝕎,dU,X',true,true)
    mul!(reshape(x.dval,size(W₁,2),:),W₁',dU,true,true)
    (nothing,)
end

function Rules.augmented_primal(config::Rules.RevConfigWidth{1},
    f::Const{typeof(apply_rotary_embedding)}, ::Type{RT},
    x::Duplicated{<:AbstractArray}, cos::Annotation, sin::Annotation) where {RT<:Union{Duplicated,DuplicatedNoNeed}}
    y = f.val(x.val, cos.val, sin.val)
    dy = zero(y)
    # Only derivatives of the angles need x; the input derivative needs just c/s.
    needs_x = any(a -> a isa Duplicated &&
        !(Rules.runtime_activity(config) && a.val === a.dval),(cos,sin))
    saved = needs_x && Rules.overwritten(config)[2] ? copy(x.val) : x.val
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
    f::Const{typeof(rescale)}, ::Type{RT}, x::Duplicated, c::Const) where {RT<:Union{Duplicated,DuplicatedNoNeed}}
    y = f.val(x.val,c.val)
    dy = zero(y)
    augmented_return(config,RT,y,dy,dy)
end

function Rules.reverse(::Rules.RevConfigWidth{1}, ::Const{typeof(rescale)},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}}, dy, x::Duplicated, c::Const)
    @. x.dval += c.val * dy
    nothing,nothing
end

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
    y, r = norm_state(x.val,ϵ.val)
    dy = zero(y)
    saved = Rules.overwritten(config)[2] ? copy(x.val) : x.val
    augmented_return(config, RT, y, dy, (; x=saved, r, dy))
end

function Rules.reverse(::Rules.RevConfigWidth{1}, ::Const{typeof(norm)},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}}, tape,
    x::Duplicated, ϵ::Const)
    norm_gradient!(x.dval,tape.dy,tape.x,tape.r)
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
    cross_entropy_gradient!(x.dval,tape.x,targets.val,tape.dy,ignore.val,reduction.val)
    nothing, nothing, nothing, nothing
end

Enzyme.Duplicated(params::Params) =
    Enzyme.Duplicated(params.Θ, params.δ)

function Enzyme.EnzymeRules.augmented_primal(
    config::Enzyme.EnzymeRules.RevConfigWidth{1},
    ::Enzyme.Const{typeof(attention)},
    ::Type{RT},
    Q::Enzyme.Duplicated{AQ},
    K::Enzyme.Duplicated{AK},
    V::Enzyme.Duplicated{AV},
    window::Enzyme.Annotation{Tuple{Int,Int}},
) where {
    RT<:Union{Enzyme.Duplicated,Enzyme.DuplicatedNoNeed},
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
    tape = (; saved, dO)

    return augmented_return(config, RT, saved.O, dO, tape)
end

function Enzyme.EnzymeRules.reverse(
    ::Enzyme.EnzymeRules.RevConfigWidth{1},
    ::Enzyme.Const{typeof(attention)},
    ::Type{<:Union{Enzyme.Duplicated,Enzyme.DuplicatedNoNeed}},
    tape,
    Q::Enzyme.Duplicated,
    K::Enzyme.Duplicated,
    V::Enzyme.Duplicated,
    window::Enzyme.Annotation{Tuple{Int,Int}},
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
Convenience call; keep `gradient_state` outside the loop for repeated batches.
"""
function loss_and_gradient!(
    params::Params{Float32},
    config::GPTConfig,
    layout,
    tokens,
    targets,
)
    state = gradient_state(params,config,layout,tokens,targets)
    loss_and_gradient!(params,state,layout,tokens,targets)
end

end
