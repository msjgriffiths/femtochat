module FemtoChatReactantExt

using FemtoChat, Reactant, CUDA
import Enzyme
using Enzyme: ReverseWithPrimal, Const
using FemtoChat.Parameters: window_sizes
import FemtoChat.Parameters: rotary_embeddings
import FemtoChat: initialize!, gradient_state, loss_and_gradient!, ℒ
import FemtoChat.Kernels: attention

include("reactant/cuda_call.jl")

# Reactant 0.2.285 collapses a one-element parameter view to a scalar index.
# Preserve its range when writing traced data back into the flat vector.
function Reactant.TracedUtils.set_mlir_data!(
    x::SubArray{Reactant.TracedRNumber{T},1,<:Reactant.TracedRArray,
                Tuple{UnitRange{Int}},true}, data,
) where T
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

struct Forward{A,Window} end
struct Backward{A,Window} end
(::Forward{A,W})(O,ℓ,m,Q,K,V) where {A,W} =
    FemtoChat.Kernels.attention!(A(),O,ℓ,m,Q,K,V,W)
function (::Backward{A,W})(dQ,dK,dV,dO,Q,K,V,O,ℓ,m) where {A,W}
    GPU = Base.get_extension(FemtoChat,:FemtoChatCUDAExt)
    if A == GPU.TensorCoreInstruction
        FemtoChat.Kernels.Δattention!(A(),dQ,dK,dV,dO,Q,K,V,O,ℓ,m,W;accumulate=false)
    else
        foreach(d -> fill!(d,0f0),(dQ,dK,dV))
        FemtoChat.Kernels.Δattention!(A(),dQ,dK,dV,dO,Q,K,V,O,ℓ,m,W)
    end
    nothing
end

function instruction(F,D)
    GPU = Base.get_extension(FemtoChat,:FemtoChatCUDAExt)
    GPU.instruction(F,CUDA.device(),Val(D))
end
Reactant.@skip_rewrite_func instruction

# JIT CUDA launchers before tracing and before XLA invokes a foreign callback.
function prepare_attention(config,tokens)
    D,H,Hkv = config.n_embed ÷ config.n_head,config.n_head,config.n_kv_head
    T,B = size(tokens)
    𝒜 = instruction(Float32,D)
    GPU = Base.get_extension(FemtoChat,:FemtoChatCUDAExt)
    F = 𝒜 isa GPU.TensorCoreInstruction ? Float16 : Float32
    Q = CUDA.ones(F,D,H,T,B)
    K,V = ntuple(_ -> CUDA.ones(F,D,Hkv,T,B),2)
    O,dO,dQ = ntuple(_ -> CUDA.zeros(Float32,D,H,T,B),3)
    dK,dV = ntuple(_ -> CUDA.zeros(Float32,D,Hkv,T,B),2)
    ℓ,m = ntuple(_ -> CUDA.zeros(Float32,1,T,H,B),2)
    for window in unique(window_sizes(config))
        forward,backward = Forward{typeof(𝒜),window}(),Backward{typeof(𝒜),window}()
        forward(O,ℓ,m,Q,K,V)
        backward(dQ,dK,dV,dO,Q,K,V,O,ℓ,m)
        specs(xs) = map(x -> (eltype(x),size(x)),xs)
        ReactantCUDACall.register(forward,specs((O,ℓ,m)),specs((Q,K,V)))
        ReactantCUDACall.register(backward,specs((dQ,dK,dV)),specs((dO,Q,K,V,O,ℓ,m)))
    end
    CUDA.synchronize()
    foreach(CUDA.unsafe_free!,(Q,K,V,O,dO,dQ,dK,dV,ℓ,m))
    nothing
end

function attention(Q::Reactant.AnyTracedRArray{F,4}, K::Reactant.AnyTracedRArray{F,4},
                   V::Reactant.AnyTracedRArray{F,4}, window) where F
    D,H,T,B = size(Q)
    𝒜 = instruction(F,D)
    GPU = Base.get_extension(FemtoChat,:FemtoChatCUDAExt)
    storage = 𝒜 isa GPU.TensorCoreInstruction && F == Float32 ? Float16 : F
    q,k,v = map(x -> Reactant.materialize_traced_array(storage.(x)),(Q,K,V))
    outputs = ((Float32,size(Q)),(Float32,(1,T,H,B)),(Float32,(1,T,H,B)))
    O,_,_ = ReactantCUDACall.call(Forward{typeof(𝒜),window}(),outputs,q,k,v;
                                vjp=Backward{typeof(𝒜),window}())
    O
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

end
