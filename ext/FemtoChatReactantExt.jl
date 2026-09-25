module FemtoChatReactantExt

using FemtoChat, Reactant, CUDA
import Enzyme
using Enzyme: ReverseWithPrimal, Const
using FemtoChat.Parameters: window_sizes
import FemtoChat.Parameters: rotary_embeddings
import FemtoChat: initialize!, gradient_state, loss_and_gradient!, ℒ
import FemtoChat.Kernels: attention
import FemtoChat.GPT: cross_entropy
import FemtoChat.Optimizer: muon_group, muon_workspace, gather!, scatter!, batched_mul!
using FemtoChat.Optimizer: muon_direction!, adam_moments!
using FemtoChat.Optimizer: MuonGroup, ParameterUpdate, Muon

include("reactant/cuda_call.jl")

# One final writeback lets XLA donate Θ instead of assembling and copying a new Θ.
struct ParameterStep{Adam,NMuon} end
function (::ParameterStep{Adam,NMuon})(Θ,previous,args...) where {Adam,NMuon}
    GPU = Base.get_extension(FemtoChat,:FemtoChatCUDAExt)
    # XLA aliases Θ to previous. The foreign-call operands are flattened by group.
    for (i,meta) in enumerate(Adam)
        𝓂,𝓋,α,β,η = args[5i-4:5i]
        @cuda threads=256 blocks=cld(length(𝓂),256) GPU.adam_update_kernel!(Θ,𝓂,𝓋,α,β,η,Val(meta))
    end
    start = 5length(Adam)
    for i in 1:NMuon
        X,η,λ,offsets = args[start+4i-3:start+4i]
        @cuda threads=256 blocks=(cld(size(X,1)*size(X,2),256),size(X,3)) GPU.muon_update_kernel!(Θ,X,η,λ,offsets)
    end
    nothing
end

function prepare_parameter_step(adam,n,inputs)
    f = ParameterStep{adam,n}()
    outputs = (inputs[1],)
    haskey(ReactantCUDACall.registry,(f,outputs,inputs)) && return f
    GPU = Base.get_extension(FemtoChat,:FemtoChatCUDAExt)
    θ,scalar = CUDA.ones(Float32,1),CUDA.ones(Float32)
    X,offsets = CUDA.ones(Float32,1,1,1),CUDA.zeros(Int,1)
    for meta in adam
        @cuda launch=false GPU.adam_update_kernel!(θ,θ,θ,scalar,scalar,scalar,Val(meta))
    end
    @cuda launch=false GPU.muon_update_kernel!(θ,X,scalar,scalar,offsets)
    foreach(CUDA.unsafe_free!,(θ,scalar,X,offsets))
    ReactantCUDACall.register(f,outputs,inputs)
    f
end
Reactant.@skip_rewrite_func prepare_parameter_step

function (ω::MuonAdamW{A,<:Tuple{MuonGroup{U,P,I},Vararg}})(;
    η=1f0, μ=nothing, λ=nothing, Nₜ=1f0,
) where {A,U,P<:Params{Float32,<:Reactant.TracedRArray},I}
    if cpu_backend()
        for update in ω.adamw
            update(; α=η*update.ω.α,Nₜ)
        end
        for group in ω.muon
            group(; α=η*group.update.ω.α,Nₜ,
                μ=isnothing(μ) ? group.update.ω.μ : μ,λ=isnothing(λ) ? group.update.ω.λ : λ)
        end
        return nothing
    end
    args,adam = (),()
    for update in ω.adamw
        a = update.ω
        adam_moments!(a,update.δ; Nₜ)
        rate = η*a.α
        α,β,rate = Reactant.promote_to.(Reactant.TracedRNumber{Float32},
            (rate/(1f0-a.β₁^a.t),1f0-a.β₂^a.t,rate))
        args = (args...,Reactant.materialize_traced_array(a.𝓂ₜ),Reactant.materialize_traced_array(a.𝓋ₜ),
            α,β,rate)
        adam = (adam...,(update.offset,a.λ,a.ϵ))
    end
    for group in ω.muon
        δ = similar(group.update.δ)
        gather!(δ,group.params.δ,group.offsets)
        m = group.update.ω
        X = muon_direction!(m,δ; μ=isnothing(μ) ? m.μ : μ,Nₜ)
        rate = η*m.α*√max(1f0,Float32(size(X,1))/Float32(size(X,2)))
        decay = isnothing(λ) ? m.λ : λ
        rate,decay = Reactant.promote_to.(Reactant.TracedRNumber{Float32},(rate,decay))
        args = (args...,Reactant.materialize_traced_array(X),rate,decay,Reactant.Ops.constant(collect(group.offsets)))
    end
    Θ = first(ω.muon).params.Θ
    inputs = map(x -> (Reactant.unwrapped_eltype(typeof(x)),size(x)),(Θ,args...))
    f = prepare_parameter_step(adam,length(ω.muon),inputs)
    Θ .= only(ReactantCUDACall.call(f,(inputs[1],),Reactant.materialize_traced_array(Θ),args...;alias_input=0))
    nothing
end

# Offsets are static layout metadata, not an array of runtime scalar indices.
function muon_group(params::Params{Float32,<:ConcreteRArray}, specs; kwargs...)
    shape = (first(specs).shape..., length(specs))
    θ, δ = similar(params.Θ, shape), similar(params.δ, shape)
    offsets = Tuple(first(spec.range)-1 for spec in specs)
    MuonGroup(ParameterUpdate(Muon(δ; kwargs...), θ, δ), params, offsets)
end

# Scratch belongs to this compiled call, not to the state returned between steps.
muon_workspace(ω::Muon{F,M}) where {F<:AbstractFloat,M<:Reactant.AnyTracedRArray} =
    muon_workspace(ω.𝓂ₜ, ω.𝓋ₜ, Reactant.unwrapped_eltype(typeof(ω.work.polar.X)))

function (group::MuonGroup{U,P,<:Tuple})(;
    α=group.update.ω.α, μ=group.update.ω.μ, λ=group.update.ω.λ, Nₜ=1f0,
) where {U,P<:Params{Float32,<:Reactant.AnyTracedRArray}}
    (; update, params, offsets) = group
    δ = similar(update.δ)
    gather!(δ, params.δ, offsets)
    X = muon_direction!(update.ω, δ; μ, Nₜ)
    m, n = size(X,1), size(X,2)
    η = α * √max(1f0, Float32(m) / Float32(n))
    for (k, offset) in enumerate(offsets)
        @views begin
            θ = reshape(params.Θ[offset+1:offset+m*n], m, n)
            direction = X[:,:,k]
            @. θ -= η * direction + η * λ * θ * ((direction * θ) ≥ 0)
        end
    end
    nothing
end

function gather!(stack::Reactant.AnyTracedRArray, vector, offsets::Tuple)
    width = size(stack,1) * size(stack,2)
    for (k, offset) in enumerate(offsets)
        @views stack[:,:,k] .= reshape(vector[offset+1:offset+width], size(stack,1), size(stack,2))
    end
    stack
end

function scatter!(vector::Reactant.AnyTracedRArray, stack, offsets::Tuple)
    width = size(stack,1) * size(stack,2)
    for (k, offset) in enumerate(offsets)
        @views vector[offset+1:offset+width] .= vec(stack[:,:,k])
    end
    vector
end

function batched_mul!(C::Reactant.AnyTracedRArray, A, B, transpose_A=false, transpose_B=false)
    product = Reactant.Ops.dot_general(Reactant.materialize_traced_array(A),
        Reactant.materialize_traced_array(B);
        contracting_dimensions=([transpose_A ? 1 : 2], [transpose_B ? 2 : 1]),
        batching_dimensions=([3], [3]))
    C .= permutedims(product, (2,3,1))
    C
end

# Reactant 0.2.285 collapses a one-element parameter view to a scalar index.
# Preserve its range when writing traced data back into the flat vector.
function Reactant.TracedUtils.set_mlir_data!(
    x::SubArray{Reactant.TracedRNumber{T},1,<:Reactant.TracedRArray,
                Tuple{UnitRange{Int}},L}, data,
) where {T,L}
    parent(x)[only(parentindices(x))] = Reactant.TracedRArray{T}(data)
    x
end

# Generate weights on the device; only the layer specifications are static.
function initialize!(params::Params{Float32,<:Reactant.ConcreteRArray}, layout,
                     ℛ=Reactant.ReactantRNG())
    layout = (;layout...,transformer=(;layout.transformer...,blocks=Tuple(layout.transformer.blocks)))
    Reactant.@jit initialize!(params.Θ,layout,ℛ)
    nothing
end

# Trace range construction too, so RoPE is generated on the selected device.
function rotary_embeddings(::Type{<:Reactant.ConcreteRArray{T}}, seq_len::Int,
                           head_dim::Int, base::AbstractFloat=100_000f0) where T
    Reactant.@jit rotary_embeddings(Reactant.TracedRArray{T,1},seq_len,head_dim,base)
end

struct Forward{A,Window,Options} end
struct Backward{A,Window,Options} end
(::Forward{A,W,Options})(O,ℓ,m,Q,K,V) where {A,W,Options} =
    FemtoChat.Kernels.attention!(A(),O,ℓ,m,Q,K,V,W;Options...)
function (::Backward{A,W,Options})(dQ,dK,dV,dO,Q,K,V,O,ℓ,m) where {A,W,Options}
    GPU = Base.get_extension(FemtoChat,:FemtoChatCUDAExt)
    if A == GPU.TensorCoreInstruction
        FemtoChat.Kernels.Δattention!(A(),dQ,dK,dV,dO,Q,K,V,O,ℓ,m,W;accumulate=false,Options...)
    else
        foreach(d -> fill!(d,0f0),(dQ,dK,dV))
        FemtoChat.Kernels.Δattention!(A(),dQ,dK,dV,dO,Q,K,V,O,ℓ,m,W;Options...)
    end
    nothing
end

function instruction(F,D)
    GPU = Base.get_extension(FemtoChat,:FemtoChatCUDAExt)
    GPU.instruction(F,CUDA.device(),Val(D))
end
Reactant.@skip_rewrite_func instruction

cpu_backend() = Reactant.XLA.platform_name(Reactant.XLA.default_backend()) == "cpu"
Reactant.@skip_rewrite_func cpu_backend

# JIT CUDA launchers before tracing and before XLA invokes a foreign callback.
function prepare_attention(config,tokens; compute_type=Float32, forward=(;), backward=(;))
    cpu_backend() && return nothing
    D,H,Hkv = config.n_embed ÷ config.n_head,config.n_head,config.n_kv_head
    T,B = size(tokens)
    𝒜 = instruction(compute_type,D)
    GPU = Base.get_extension(FemtoChat,:FemtoChatCUDAExt)
    F = 𝒜 isa GPU.TensorCoreInstruction && compute_type == Float32 ? Float16 : compute_type
    Q = CUDA.ones(F,D,H,T,B)
    K,V = ntuple(_ -> CUDA.ones(F,D,Hkv,T,B),2)
    O,dO = ntuple(_ -> CUDA.zeros(compute_type,D,H,T,B),2)
    # Accumulation stays FP32 inside attention; store gradients in the input type.
    dQ = CUDA.zeros(F,D,H,T,B)
    dK,dV = ntuple(_ -> CUDA.zeros(F,D,Hkv,T,B),2)
    ℓ,m = ntuple(_ -> CUDA.zeros(Float32,1,T,H,B),2)
    for window in unique(window_sizes(config))
        f,b = Forward{typeof(𝒜),window,forward}(),Backward{typeof(𝒜),window,backward}()
        f(O,ℓ,m,Q,K,V)
        b(dQ,dK,dV,dO,Q,K,V,O,ℓ,m)
        specs(xs) = map(x -> (eltype(x),size(x)),xs)
        ReactantCUDACall.register(f,specs((O,ℓ,m)),specs((Q,K,V)))
        ReactantCUDACall.register(b,specs((dQ,dK,dV)),specs((dO,Q,K,V,O,ℓ,m)))
    end
    CUDA.synchronize()
    foreach(CUDA.unsafe_free!,(Q,K,V,O,dO,dQ,dK,dV,ℓ,m))
    nothing
end

function attention(Q::Reactant.AnyTracedRArray{F,4}, K::Reactant.AnyTracedRArray{F,4},
                   V::Reactant.AnyTracedRArray{F,4}, window; forward=(;), backward=(;)) where F
    cpu_backend() && return FemtoChat.Kernels.naive_attention(Q,K,V,window)
    D,H,T,B = size(Q)
    𝒜 = instruction(F,D)
    GPU = Base.get_extension(FemtoChat,:FemtoChatCUDAExt)
    storage = 𝒜 isa GPU.TensorCoreInstruction && F == Float32 ? Float16 : F
    q,k,v = map(x -> Reactant.materialize_traced_array(storage.(x)),(Q,K,V))
    outputs = ((F,size(Q)),(Float32,(1,T,H,B)),(Float32,(1,T,H,B)))
    O,_,_ = ReactantCUDACall.call(Forward{typeof(𝒜),window,forward}(),outputs,q,k,v;
                                vjp=Backward{typeof(𝒜),window,backward}())
    O
end

struct EmbeddingForward end
struct EmbeddingBackward end
function (::EmbeddingForward)(y,weights,tokens)
    Base.get_extension(FemtoChat,:FemtoChatCUDAExt).embedding!(y,weights,tokens)
end
function (::EmbeddingBackward)(δ,dy,weights,tokens,y)
    fill!(δ,0f0)
    FemtoChat.Kernels.embedding_gradient!(δ,dy,tokens)
end

# Compile launchers with tiny buffers; register their actual shapes for XLA.
function prepare_embedding(shape,I,token_shape)
    inputs = ((Float32,shape),(I,token_shape))
    outputs = ((Float32,(shape[1],token_shape...)),)
    f,b = EmbeddingForward(),EmbeddingBackward()
    haskey(ReactantCUDACall.registry,(f,outputs,inputs)) && return nothing
    weights = CUDA.ones(Float32,1,1)
    tokens = CUDA.ones(I,ntuple(_ -> 1,length(token_shape)))
    y,dy = ntuple(_ -> CUDA.ones(Float32,1,size(tokens)...),2)
    δ = CUDA.zeros(Float32,1,1)
    f(y,weights,tokens)
    b(δ,dy,weights,tokens,y)
    ReactantCUDACall.register(f,outputs,inputs)
    ReactantCUDACall.register(b,(inputs[1],),(outputs[1],inputs...,outputs...))
    CUDA.synchronize()
    foreach(CUDA.unsafe_free!,(weights,tokens,y,dy,δ))
    nothing
end
Reactant.@skip_rewrite_func prepare_embedding

function (𝔼::FemtoChat.Parameters.Embedding)(tokens::Reactant.AnyTracedRVecOrMat{<:Integer})
    F = Reactant.unwrapped_eltype(typeof(𝔼.𝔼))
    (cpu_backend() || F != Float32) && return 𝔼.𝔼[:,tokens]
    weights,tokens = Reactant.materialize_traced_array.((𝔼.𝔼,tokens))
    prepare_embedding(size(weights),Reactant.unwrapped_eltype(typeof(tokens)),size(tokens))
    outputs = ((F,(size(weights,1),size(tokens)...)),)
    only(ReactantCUDACall.call(EmbeddingForward(),outputs,weights,tokens;vjp=EmbeddingBackward()))
end

# Elementwise target selection lets XLA fuse the loss adjoint instead of scattering it.
function cross_entropy(logits::Reactant.AnyTracedRArray, targets, ignore_index, reduction)
    V,T,B = size(logits)
    valid = targets .!= ignore_index
    # A per-token shift leaves softmax unchanged; its derivative cancels out.
    maximum_logit = Enzyme.ignore_derivatives(maximum(logits; dims=1))
    normalizer = sum(exp.(logits .- maximum_logit); dims=1)
    selected = reshape(1:V,V,1,1) .== reshape(targets,1,T,B)
    target_logit = sum(ifelse.(selected,logits,zero(eltype(logits))); dims=1)
    losses = reshape(maximum_logit .+ log.(normalizer) .- target_logit,T,B)
    losses = ifelse.(valid,losses,zero(eltype(losses)))

    if reduction == :mean
        sum(losses) / sum(valid)
    elseif reduction == :sum
        sum(losses)
    else
        losses
    end
end

ℒ(Θ,config,layout,rope,tokens,targets,positions=nothing) =
    🤖(Θ,config,layout;rope_sin_cos=rope)(tokens,targets;positions)

function gradient!(Θ,δ,config,layout,rope,tokens,targets,positions)
    result = Enzyme.gradient(ReverseWithPrimal,ℒ,Θ,Const(config),Const(layout),
                             Const(rope),Const(tokens),Const(targets),Const(positions))
    δ .= result.derivs[1]
    result.val
end

struct ReactantGradientState{F,C,L,R}
    compiled::F
    config::C
    layout::L
    rope::R
end

"""Compile the GPU loss/gradient once for this token/batch shape; reuse `params.δ`."""
function gradient_state(params::Params{Float32,<:Reactant.ConcreteRArray},
                        config::GPTConfig, layout, tokens, targets; positions=nothing)
    prepare_attention(config,tokens)
    # Layer metadata is static; weights and batches remain runtime arguments.
    layout = (;layout...,transformer=(;layout.transformer...,blocks=Tuple(layout.transformer.blocks)))
    rope = rotary_embeddings(Vector{Float32},config.max_document_tokens,config.n_embed ÷ config.n_head)
    rope = Reactant.to_rarray(rope)
    (; Θ,δ) = params
    compiled = Reactant.@compile sync=true gradient!(Θ,δ,config,layout,rope,tokens,targets,positions)
    ReactantGradientState(compiled,config,layout,rope)
end

function loss_and_gradient!(params::Params, state::ReactantGradientState, layout, tokens, targets; positions=nothing)
    (; compiled,config,rope) = state
    loss = compiled(params.Θ,params.δ,config,state.layout,rope,tokens,targets,positions)
    ReactantCUDACall.check()
    Float32(loss)
end

function __init__()
    # These mutate Reactant's registry, so restore them when loading a precompiled extension.
    Reactant.@skip_rewrite_func prepare_parameter_step
    Reactant.@skip_rewrite_func prepare_embedding
    Reactant.@skip_rewrite_func instruction
    Reactant.@skip_rewrite_func cpu_backend
end

end
