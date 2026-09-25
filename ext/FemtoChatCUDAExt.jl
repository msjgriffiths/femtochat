module FemtoChatCUDAExt

import FemtoChat.Optimizer: batched_mul!, gather!, scatter!
import FemtoChat.Kernels: embedding_gradient!, cross_entropy_gradient!
import FemtoChat.GPT: cross_entropy

using CUDA

@inline function reduce_block(op, value, neutral)
    lane, warp = (threadIdx().x-1) % 32, (threadIdx().x-1) ÷ 32
    partial = CuStaticSharedArray(Float32,32)
    value = reduce_lanes(op,value,Val(32))
    lane == 0 && (@inbounds partial[warp+1] = value)
    sync_threads()
    value = lane < blockDim().x÷32 ? (@inbounds partial[lane+1]) : neutral
    value = reduce_lanes(op,value,Val(32))
    sync_threads()
    value
end

# A block owns one token's vocabulary column. No vocabulary-sized probabilities.
function cross_entropy_kernel!(losses, δ, logits, targets, scale, ignore, ::Val{Reverse}) where Reverse
    token, V = blockIdx().x, size(logits,1)
    offset = (token-1)*V
    maximum_logit = -Inf32
    @inbounds for v in threadIdx().x:blockDim().x:V
        maximum_logit = max(maximum_logit,logits[offset+v])
    end
    maximum_logit = reduce_block(max,maximum_logit,-Inf32)
    normalizer = 0f0
    @inbounds for v in threadIdx().x:blockDim().x:V
        normalizer += exp(logits[offset+v] - maximum_logit)
    end
    normalizer = reduce_block(+,normalizer,0f0)
    @inbounds target = targets[token]
    if Reverse
        @inbounds weight = scale[length(scale) == 1 ? 1 : token]
        @inbounds for v in threadIdx().x:blockDim().x:V
            δ[offset+v] += target == ignore ? 0f0 : weight *
                (exp(logits[offset+v] - maximum_logit) / normalizer - (v == target))
        end
    elseif threadIdx().x == 1
        @inbounds losses[token] = target == ignore ? 0f0 :
            maximum_logit + log(normalizer) - logits[offset+target]
    end
    nothing
end

Base.@constprop :aggressive function cross_entropy(logits::CuArray{Float32,3}, targets::CuArray, ignore, reduction)
    losses = similar(logits,size(targets))
    @cuda threads=256 blocks=length(targets) cross_entropy_kernel!(losses,nothing,logits,targets,nothing,ignore,Val(false))
    reduction in (:sum,:mean) || return losses
    total = sum(losses)
    reduction == :mean ? total / sum(t -> t != ignore,targets) : total
end

function cross_entropy_gradient!(δ::CuArray{Float32}, logits::CuArray{Float32,3}, targets::CuArray, dy, ignore, reduction)
    scale = reduction == :mean ? dy ./ sum(t -> t != ignore,targets; dims=(1,2)) : dy
    @cuda threads=256 blocks=length(targets) cross_entropy_kernel!(nothing,δ,logits,targets,scale,ignore,Val(true))
    nothing
end

function embedding_kernel!(y, weights, tokens)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= length(y)
        token,d = divrem(i-1,size(weights,1))
        @inbounds y[i] = weights[d+1,tokens[token+1]]
    end
    nothing
end

function embedding!(y::CuArray, weights::CuArray, tokens::CuArray)
    @cuda threads=256 blocks=cld(length(y),256) embedding_kernel!(y,weights,tokens)
    nothing
end

function embedding_gradient_kernel!(δ, dy, tokens)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= length(dy)
        D = size(δ,1)
        token, d = divrem(i-1,D)
        @inbounds value = dy[i]
        # Masked targets produce zero adjoints. Avoid contended atomics for padding.
        if !iszero(value)
            @inbounds CUDA.@atomic δ[d+1,tokens[token+1]] += value
        end
    end
    nothing
end

function embedding_gradient!(δ::CuArray, dy::CuArray, tokens::CuArray)
    @cuda threads=256 blocks=cld(length(dy),256) embedding_gradient_kernel!(δ,dy,tokens)
    nothing
end

# Final optimizer writes. Moments and Muon's direction are computed by Reactant.
# α includes the first-moment bias correction; β is the second-moment correction.
# η is the uncorrected learning rate used for weight decay.
function adam_update_kernel!(Θ,𝓂,𝓋,α,β,η,::Val{Meta}) where Meta
    offset,λ,ϵ = Meta
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if i <= length(𝓂)
        @inbounds Θ[offset+i] = Θ[offset+i]*(1f0-η[1]*λ) - α[1]*𝓂[i]/(sqrt(𝓋[i]/β[1])+ϵ)
    end
    nothing
end

function muon_update_kernel!(Θ,X,η,λ,offsets)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    k, width = blockIdx().y, size(X,1)*size(X,2)
    if i <= width
        @inbounds begin
            index = offsets[k]+i
            x = X[i+(k-1)*width]
            Θ[index] -= η[1]*x + η[1]*λ[1]*Θ[index]*(x*Θ[index] ≥ 0)
        end
    end
    nothing
end

function batched_mul!(C::CuArray{F,3}, A::CuArray{F,3}, B::CuArray{F,3},
                      transpose_A=false, transpose_B=false) where F
    CUDA.CUBLAS.gemmStridedBatchedEx!(transpose_A ? 'T' : 'N', transpose_B ? 'T' : 'N',
        1f0, A, B, 0f0, C)
end

function parameter_copy_kernel!(stack, vector, offsets, ::Val{Gather}) where Gather
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= length(stack)
        width = size(stack,1) * size(stack,2)
        k, entry = divrem(i-1, width)
        @inbounds if Gather
            stack[i] = vector[offsets[k+1] + entry+1]
        else
            vector[offsets[k+1] + entry+1] = stack[i]
        end
    end
    nothing
end

function gather!(stack::CuArray, vector::CuArray, offsets::CuArray)
    @cuda threads=256 blocks=cld(length(stack),256) parameter_copy_kernel!(stack,vector,offsets,Val(true))
    stack
end

function scatter!(vector::CuArray, stack::CuArray, offsets::CuArray)
    @cuda threads=256 blocks=cld(length(stack),256) parameter_copy_kernel!(stack,vector,offsets,Val(false))
    vector
end
using CUDA: i32
using Core: BFloat16
using FemtoChat
using Base.Cartesian: @ntuple

# CUDA 6 moved compiler utilities into CUDACore.
const CUDACompiler = parentmodule(CuDevice)
using .CUDACompiler: @loopinfo
const LLVM = CUDACompiler.LLVM
using .LLVM.Interop: create_function, call_function

import FemtoChat.Kernels: attention, attention!, Δattention!, attention_state,
                         attention_inputs, accumulator_type, attention_mask!,
                         flash_attention₁, flash_attention₁!, Δflash_attention₁!

# ── Configuration ───────────────────────────────────────────────────────────

# CUDA 5.x stores the assembler target as `cap`; newer CUDA stores an SMVersion
# in `sm`, preserving its architecture/family feature set. LLVM may target less.
function compiler_targets(config)
    (; params, target) = config
    arch = hasproperty(params, :sm) ? params.sm : params.cap
    feature_set = hasproperty(target, :feature_set) ? target.feature_set : :baseline
    return (; target=(; arch, ptx=params.ptx),
              llvm=(; compute=target.cap, feature_set, ptx=target.ptx))
end

"""
    capabilities(dev=CUDA.device())
    capabilities(Q::CuArray)

Report hardware compute capability, CUDA's default compilation target, LLVM's
target, and device limits. `target.arch` retains CUDA's native SMVersion (including
`a`/`f`) when available; CUDA 5.x returns a baseline VersionNumber instead.
Hardware capability is not a promise that the installed toolchain supports it.
Shared-memory limits are bytes; register limits count 32-bit registers. Opt-in
shared memory requires a separate launch configuration; it is not enabled here.
This does not enumerate individual instructions or change the active device.
"""
function capabilities(dev::CuDevice=CUDA.device())
    # Compiler configuration moved from CUDA to CUDACore in CUDA 6. This is the
    # one internal API dependency: use its target selection, not our own GPU table.
    (; target, llvm) = compiler_targets(CUDACompiler.compiler_config(dev))
    attr(code) = CUDA.attribute(dev, code)
    limits = (
        warp_size=CUDA.warpsize(dev),
        multiprocessors=attr(CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        threads_per_block=attr(CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
        threads_per_sm=attr(CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR),
        shared_bytes_per_block=attr(CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK),
        shared_bytes_per_block_optin=attr(CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN),
        shared_bytes_per_sm=attr(CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR),
        registers_per_block=attr(CUDA.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK),
        registers_per_sm=attr(CUDA.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR),
    )
    return (; name=CUDA.name(dev), compute=CUDA.capability(dev), target, llvm, limits)
end

capabilities(Q::CuArray) = capabilities(CUDA.device(Q))

struct SIMTInstruction end
struct TensorCoreInstruction end
const TensorFloat = Union{Float16,BFloat16}

instruction(::Type, dev::CuDevice, ::Val) = SIMTInstruction()
function instruction(::Type{F}, dev::CuDevice, ::Val{D}) where {F<:Union{TensorFloat,Float32},D}
    0 < D <= 128 && D % 16 == 0 || return SIMTInstruction()
    CUDA.capability(dev) >= v"8.0" || return SIMTInstruction()
    target = compiler_targets(CUDACompiler.compiler_config(dev)).llvm.compute
    target >= v"8.0" || return SIMTInstruction()
    budget = CUDA.attribute(dev,CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    shared_bytes(D,1,1,64) <= budget && backward_layout(D,16,16).bytes <= budget ?
        TensorCoreInstruction() : SIMTInstruction()
end

attention_state(::typeof(attention), Q::CuArray, K::CuArray, V::CuArray, window) =
    attention_state(instruction(eltype(Q),CUDA.device(Q),Val(size(Q,1))),Q,K,V,window)

accumulator_type(::TensorCoreInstruction, ::Type) = Float32
attention_inputs(::TensorCoreInstruction, Q::CuArray{Float32}, K, V) =
    Float16.(Q),Float16.(K),Float16.(V)

attention!(::SIMTInstruction, args...; kwargs...) = flash_attention₁!(args...;kwargs...)
Δattention!(::SIMTInstruction, args...; kwargs...) = Δflash_attention₁!(args...;kwargs...)

function attention(Q::CuArray{F,4}, K::CuArray{F,4}, V::CuArray{F,4}, window) where {F<:AbstractFloat}
    attention_state(attention,Q,K,V,window).O
end

# Explicit instruction selection is useful for tests and tile benchmarks.
attention(𝒜::Union{SIMTInstruction,TensorCoreInstruction}, Q, K, V, window; kwargs...) =
    attention_state(𝒜,Q,K,V,window;kwargs...).O

# One compile-time description of the FP32 SIMT multiply and its storage.
const SIMT = (thread_tile=(8,4), lanes=(4,8), stage=16, vector=4, swizzle=(bits=3, base=2))

function attention_layout(D, warps, spec=SIMT)
    (; thread_tile, lanes, stage, vector, swizzle) = spec
    warp_tile = thread_tile .* lanes
    Bᵣ, Bᶜ = warp_tile .* warps
    threads = 32prod(warps)
    Dᵥ = Bᶜ * cld(D, Bᶜ)

    # Q/K use two stages. P̃ reuses those stages; row statistics reuse V.
    stage_size = stage * (Bᵣ + Bᶜ)
    V_offset = max(2stage_size, Bᵣ * Bᶜ)
    shared = V_offset + max(2stage * Dᵥ, Bᵣ * warps[2])
    return (; D, thread_tile, lanes, warp_tile, warps, tile=(Bᵣ,Bᶜ),
            stage, vector, swizzle, threads, Dᵥ, stage_size, V_offset, shared)
end


# ── Host entry points ───────────────────────────────────────────────────────

"""
    flash_attention₁!(𝕆, ℓ, m, Q, K, V, window, ::Val{Warps}=Val((2,2)); simt=Val(SIMT))

Configure the Float32 kernel's query/key warp arrangement. The default thread tile
and lane shapes give 32×32 entries per warp, so `(2,2)` gives a 64×64 tile with
128 threads. `simt=Val((; SIMT..., stage=8))` specializes the complete multiply
and staging code on first use. Arrays must have compatible nonempty shapes and
live on the active CUDA device; custom configurations must satisfy the kernel's
layout assumptions. This internal launcher does not validate those requirements.
Only Float32 uses this fused launcher; other types use the generic implementation.
"""
function flash_attention₁!(
    𝕆::CuArray{Float32,4}, ℓ::CuArray{Float32,4}, m::CuArray{Float32,4},
    Q::CuArray{Float32,4}, K::CuArray{Float32,4}, V::CuArray{Float32,4},
    window, ::Val{Warps}=Val((2,2));
    simt::Val{Spec}=Val(SIMT),
) where {Warps,Spec}
    D, H, T, B = size(Q)
    layout = attention_layout(D, Warps, Spec)
    Bᵣ, Bᶜ = layout.tile
    threads = layout.threads
    shmem = sizeof(Float32) * layout.shared
    blocks = (cld(T, Bᵣ), H, B)

    @cuda threads=threads blocks=blocks shmem=shmem fastmath=true flash_attention₁_kernel!(
        𝕆, ℓ, m, Q, K, V, Val(window), Val(layout))

    return nothing
end

# ── Register notation ───────────────────────────────────────────────────────

"""
    Tile{Shape}(value)
An immutable thread-local matrix tile, backed by a tuple of register values.
Tensor Core operations distribute a larger matrix tile across a warp's lanes.
"""
struct Tile{Shape,T,N}
    values::NTuple{N,T}
end

@inline Tile{Shape}(x::T) where {Shape,T<:Number} =
    Tile{Shape}(ntuple(Returns(x), Val(prod(Shape))))
@inline Tile{Shape}(values::NTuple{N,T}) where {Shape,N,T} =
    Tile{Shape,T,N}(values)

Base.size(::Tile{Shape}) where Shape = Shape
Base.length(::Tile{Shape,T,N}) where {Shape,T,N} = N
Base.eltype(::Type{<:Tile{Shape,T}}) where {Shape,T} = T
Base.Tuple(S::Tile) = S.values
Base.iterate(S::Tile, state...) = iterate(S.values, state...)

# Slices are tuple snapshots. Integer axes have length one in the slice.
@inline slice_shape(Shape,I) = map((n,i)->i isa Colon ? n : 1,Shape,I)

@inline function Base.getindex(S::Tile{Shape},I::Vararg{Union{Integer,Colon},N}) where {Shape,N}
    length(Shape) == N || error("use one index per tile dimension")
    dimensions = slice_shape(Shape,I)
    values = ntuple(Val(prod(dimensions))) do j
        position = Tuple(CartesianIndices(dimensions)[j])
        coordinate = map((i,p)->i isa Colon ? p : i,I,position)
        @inbounds S.values[LinearIndices(Shape)[coordinate...]]
    end
    any(i->i isa Colon,I) ? values : only(values)
end

@inline function replaced(S::Tile{Shape,T,N},x,I::Vararg{Union{Integer,Colon},M}) where {Shape,T,N,M}
    dimensions = slice_shape(Shape,I)
    ntuple(Val(N)) do j
        coordinate = Tuple(CartesianIndices(Shape)[j])
        selected = all(map((i,c)->i isa Colon || i == c,I,coordinate))
        position = map((i,c)->i isa Colon ? c : 1,I,coordinate)
        value = any(i->i isa Colon,I) ? x[LinearIndices(dimensions)[position...]] : x
        ifelse(selected,convert(T,value),S.values[j])
    end
end

@inline Base.setindex(S::Tile{Shape}, x, I...) where Shape =
    Tile{Shape}(replaced(S,x,I...))

# Mutating updates return nothing; Tile updates return a replacement.
Base.@propagate_inbounds assignindex!(A::Tile, x, I::Vararg{Any,N}) where N = Base.setindex(A,x,I...)
Base.@propagate_inbounds function assignindex!(A, x, I::Vararg{Any,N}) where N
    setindex!(A,x,I...)
    nothing
end

# Unroll literal loop bounds, leaving arithmetic and indexing to Julia.
function unroll(x, indices=Dict{Symbol,Int}())
    x isa Symbol && return get(indices,x,x)
    x isa Expr || return x
    x.head in (:quote, :function, :->, :let) && return x
    if x.head == :for
        binding, loop = x.args
        if Meta.isexpr(binding,:block)
            return unroll(foldr((b,tail)->Expr(:for,b,tail),binding.args;init=loop),indices)
        end
        name, range = binding.args
        range = unroll(range,indices)
        if Meta.isexpr(range,:call) && range.args[1] == :(:) && all(i->i isa Int,range.args[2:end])
            return Expr(:block,[unroll(loop,merge(indices,Dict(name=>i))) for i in (:)(range.args[2:end]...)]...)
        end
    end
    Expr(x.head,map(a->unroll(a,indices),x.args)...)
end

function assignments(x)
    x isa Expr || return x
    x.head in (:quote,:function,:->,:let) && return x
    if x.head in (:(=),:(+=),:(-=),:(*=),:(/=)) && Meta.isexpr(x.args[1],:ref)
        ref, rhs = x.args
        A, I = ref.args[1], ref.args[2:end]
        A isa Symbol || error("@tile assignment requires a local variable")
        object, indices, value, replacement = gensym.((:object,:indices,:value,:replacement))
        if x.head != :(=)
            op = Symbol(chop(string(x.head)))
            rhs = :($op(getindex($object,$indices...),$rhs))
        end
        return quote
            local $object = $A
            local $indices = ($(I...),)
            local $value = $rhs
            local $replacement = $(GlobalRef(@__MODULE__,:assignindex!))($object,$value,$indices...)
            isnothing($replacement) || ($A = $replacement)
            $value
        end
    end
    Expr(x.head,map(assignments,x.args)...)
end

"""
    @tile begin ... end

Unroll literal loops and rebind immutable tiles on indexed assignment.
Tiles must already be constructed explicitly. Other arrays retain mutation.
No variable names, tile shapes, reductions, or multiply names are recognized.
"""
macro tile(body)
    esc(assignments(unroll(body)))
end


# ── Shared memory and warp operations ───────────────────────────────────────

"""
    Swizzled(A, ::Val{S}=Val(SIMT.swizzle))

Logical matrix view of the SIMT multiply's XOR-swizzled storage. By default,
rows are permuted within groups of 32; each four-column group shares a permutation.
"""
struct Swizzled{F,Storage<:AbstractMatrix{F},S} <: AbstractMatrix{F}
    data::Storage
    @inline function Swizzled(data::AbstractMatrix{F}, ::Val{S}=Val(SIMT.swizzle)) where {F,S}
        new{F,typeof(data),S}(data)
    end
end

Base.parent(A::Swizzled) = A.data
Base.size(A::Swizzled) = size(parent(A))

function swizzle_mask(spec)
    (; bits, base) = spec
    return ((1 << bits)-1) << base
end

@inline swizzle(r, c, ::Val{S}=Val(SIMT.swizzle)) where S =
    ((r-1) ⊻ ((c-1) & swizzle_mask(S))) + 1

Base.@propagate_inbounds function Base.getindex(A::Swizzled{F,Storage,S}, r::Integer, c::Integer) where {F,Storage,S}
    @boundscheck checkbounds(A, r, c)
    @inbounds return parent(A)[swizzle(r,c,Val(S)), c]
end

Base.@propagate_inbounds function Base.setindex!(A::Swizzled{F,Storage,S}, x, r::Integer, c::Integer) where {F,Storage,S}
    @boundscheck checkbounds(A, r, c)
    @inbounds parent(A)[swizzle(r,c,Val(S)), c] = x
    return A
end

# Explicit logical cell → physical address for the register multiply. Adjacent
# logical cells need not be adjacent in storage; do not expose strides/pointer(A).
Base.@propagate_inbounds function Base.pointer(A::Swizzled{F,Storage,S}, r::Integer, c::Integer) where {F,Storage,S}
    @boundscheck checkbounds(A, r, c)
    return pointer(parent(A), swizzle(r,c,Val(S)) + size(A,1)*(c-1))
end

const Float32ᵛ{N} = NTuple{N,Base.VecElement{Float32}}

# These aligned vector operations are only for contiguous global channels.
# Strided shared-memory copies use ordinary views and copyto! in the kernel.
@inline function load(::Type{NTuple{W,VecElement{F}}}, A::CuDeviceArray{T,N,CUDA.AS.Global}, index) where {F,W,T,N}
    p = reinterpret(Core.LLVMPtr{NTuple{W,VecElement{F}},CUDA.AS.Global}, pointer(A,index))
    return CUDA.unsafe_cached_load(p,1,Val(sizeof(F)*W))
end

@inline function load(::Type{NTuple{W,VecElement{F}}}, A::CuDeviceArray{T,N,CUDA.AS.Shared}, index) where {F,W,T,N}
    p = reinterpret(Core.LLVMPtr{NTuple{W,VecElement{F}},CUDA.AS.Shared}, pointer(A,index))
    return unsafe_load(p,1,Val(sizeof(F)*W))
end

# Width is the Q/K staging width or padded V pitch. Whole stages elide the
# channel guard; uneven head widths use scalar loads to pad the final vector.
@inline function load(::Type{Float32ᵛ{W}}, A::CuDeviceArray, index, valid::Bool, d, ::Val{D}, ::Val{Width}) where {W,D,Width}
    if D % W == 0
        return valid && (D % Width == 0 || d <= D) ? load(Float32ᵛ{W}, A, index) :
            ntuple(_ -> VecElement(0f0), Val(W))
    end
    return ntuple(Val(W)) do j
        VecElement(valid && d+j-1 <= D ? (@inbounds A[index+j-1]) : 0f0)
    end
end

@inline function store!(A::CuDeviceArray{T,N,AS}, index, value::NTuple{W,VecElement{F}}) where {T,N,AS,F,W}
    p = reinterpret(Core.LLVMPtr{NTuple{W,VecElement{F}},AS}, pointer(A,index))
    unsafe_store!(p,value,1,Val(sizeof(F)*W))
    return nothing
end


# All active warp lanes must reach this call, with each aligned W-wide subgroup
# present in full, with W a power of two ≤ 32. The XOR butterfly uses offsets
# 1, 2, 4, ...; op must be
# associative and commutative (up to rounding).
@inline function reduce_lanes(op::F, x, ::Val{W}) where {F,W}
    offset = 1
    while offset < W
        x = op(x, CUDA.shfl_xor_sync(CUDA.FULL_MASK, x, offset, W))
        offset *= 2
    end
    return x
end

# Fold contributions from different column warps, after a block barrier.
@inline function reduce_row(op::F, x, A, row, ::Val{N}) where {F,N}
    @loopinfo unroll for column in 1:N
        @inbounds x = op(x, A[row,column])
    end
    return x
end


# ── Register matrix multiply ────────────────────────────────────────────────

@inline function shared_group_pointer(A, row, channel, ::Val{Pitch}, ::Val{Width}, ::Val{Mask}) where {Pitch,Width,Mask}
    address = (reinterpret(UInt,A) % UInt32) + UInt32(sizeof(Float32)) * ((row + Pitch*channel) % UInt32)
    address ⊻= UInt32(sizeof(Float32)) * ((channel & Mask) % UInt32)
    return reinterpret(Core.LLVMPtr{Float32ᵛ{Width},CUDA.AS.Shared}, UInt(address))
end

# See https://github.com/NVIDIA/cutlass/blob/v2.11.0/include/cutlass/gemm/thread/mma_sm50.h
@inline function Base.muladd(
    A::Core.LLVMPtr{Float32,CUDA.AS.Shared}, B::Core.LLVMPtr{Float32,CUDA.AS.Shared},
    x::NTuple{N,Float32},
    ::Val{LDA}, ::Val{LDB}, ::Val{Layout}, ::Val{Spec},
) where {N,LDA,LDB,Layout,Spec}
    x = VecElement.(x)
    R, C = Spec.thread_tile
    W, stage = Spec.vector, Spec.stage
    mask = swizzle_mask(Spec.swizzle)
    group = min(stage, 1 << Spec.swizzle.base)

    # One XOR per vector/group; the unrolled inner loads use constant offsets.
    @inbounds @loopinfo unroll=false for d in Int32(0):Int32(group):Int32(stage-1)
        ap = ntuple(j -> shared_group_pointer(A,Int32(W*(j-1)),d,Val(LDA),Val(W),Val(mask)), Val(R÷W))
        bp = ntuple(j -> shared_group_pointer(B,Int32(W*(j-1)),d,Val(LDB),Val(W),Val(Layout == :swizzled ? mask : 0)), Val(C÷W))
        @loopinfo unroll for offset in Int32(0):Int32(group-1)
            q = ntuple(j -> unsafe_load(ap[j],1+Int32(LDA÷W)*offset,Val(sizeof(Float32)*W)), Val(R÷W))
            k = ntuple(j -> unsafe_load(bp[j],1+Int32(LDB÷W)*offset,Val(sizeof(Float32)*W)), Val(C÷W))
            previous = x # Capture a value, not the reassigned loop variable.
            x = ntuple(Val(N)) do i
                r, c = Tuple(CartesianIndices((R,C))[i])
                VecElement(muladd(q[cld(r,W)][mod1(r,W)].value, k[cld(c,W)][mod1(c,W)].value, previous[i].value))
            end
        end
    end
    return getfield.(x,:value)
end


# Preserve the tile's shape around the tuple-level register multiply.
@inline function Base.muladd(a::A, b::B, x::Tile{Shape}, rest::Vararg{Any,N}) where {A,B,Shape,N}
    Tile{Shape}(muladd(a,b,x.values,rest...))
end

# ── Cooperative staging ────────────────────────────────────────────────────

# Q and K share one double-buffered channel stage.
@inline function qk_buffers(buffer_offset, ::Val{L}) where L
    Bᵣ, Bᶜ = L.tile
    Qᵢ = Swizzled(CuDynamicSharedArray(Float32, (Bᵣ,L.stage), buffer_offset), Val(L.swizzle))
    Kⱼ = Swizzled(CuDynamicSharedArray(Float32, (Bᶜ,L.stage),
        buffer_offset + sizeof(Float32)*L.stage*Bᵣ), Val(L.swizzle))
    return Qᵢ, Kⱼ
end

@inline v_buffer(buffer_offset, ::Val{L}) where L =
    CuDynamicSharedArray(Float32, (L.Dᵥ,L.stage), sizeof(Float32)*L.V_offset + buffer_offset)

# Register vectors → strided rows in a shared tile. 
@inline function stage_rows!(A, x::NTuple{N,Float32ᵛ{W}}, row, row_step, d, ::Val{Rows}) where {N,W,Rows}
    @loopinfo unroll for j in 1:N
        r = row + (j-1)*row_step
        if Rows % row_step == 0 || r <= Rows
            @inbounds @views copyto!(A[r,d:d+W-1], getfield.(x[j], :value))
        end
    end
    return nothing
end

# V's channels are contiguous; slots carry (channel, row, global address).
@inline function stage_slots!(A, x::NTuple{N,Float32ᵛ{W}}, slots::NTuple{N,S},
    ::Val{Rows}, ::Val{Pitch}, ::Val{Threads}) where {N,W,S,Rows,Pitch,Threads}
    full = Rows*Pitch % (W*Threads) == 0
    # Val-sized expansion avoids the local-memory traffic of tuple foreach here.
    ntuple(Val(N)) do j
        d, row, _ = slots[j]
        if full || row <= Rows
            @inbounds store!(A, d+Pitch*(row-1), x[j])
        end
        nothing
    end
    return nothing
end

# ── FP32 FlashAttention ─────────────────────────────────────────────────────

@generated function flash_attention₁_kernel!(
    𝕆, ℓ, m, Q, K, V, ::Val{Window}, ::Val{Layout},
) where {Window,Layout}
    # Expand only the register counts and cooperative-load counts here.
    (; D, threads, stage, vector, Dᵥ) = Layout
    Bᵣ, Bᶜ = Layout.tile
    R, C = Layout.thread_tile
    Nq, Nk = cld(stage*Bᵣ, vector*threads), cld(stage*Bᶜ, vector*threads)
    No = cld(D, Bᶜ)
    Nv = cld(stage*Dᵥ, vector*threads)
    return :(@tile begin
        window = Window
        causal = window == (-1,0)
        left, right = window
        layout = Layout
        D = layout.D
        Bᵣ, Bᶜ = layout.tile
        Wᶜ = layout.warps[2]
        stage, width = layout.stage, layout.vector

        τ = threadIdx().x - 1i32
        lane, warp = τ % 32i32, τ ÷ 32i32
        i, head, document = blockIdx()
        T, H, Hkv = Int32(size(Q,3)), Int32(size(Q,2)), Int32(size(K,2))
        kv_head = cld(head, H ÷ Hkv)
        blockᵢ = Int32(Bᵣ) * (i - 1)

        # Column varies fastest within the lane layout, then the grid of warps.
        laneᵣ, laneᶜ = lane ÷ Int32(layout.lanes[2]), lane % Int32(layout.lanes[2])
        warpᵣ, warpᶜ = warp ÷ Int32(Wᶜ), warp % Int32(Wᶜ)
        row₀ = Int32(layout.warp_tile[1]) * warpᵣ + Int32(layout.thread_tile[1]) * laneᵣ
        col₀ = Int32(layout.warp_tile[2]) * warpᶜ + Int32(layout.thread_tile[2]) * laneᶜ
        query₀ = blockᵢ + row₀
        log₂e = 1.4426950408889634f0
        scale = log₂e / sqrt(Float32(D))

        # Output channels may require several register tiles per thread.
        padded_D = Int32(cld(D,stage)*stage)
        Dᵥ = layout.Dᵥ
        qk_stage_bytes = sizeof(Float32) * layout.stage_size
        v_stage_bytes = sizeof(Float32) * stage * Dᵥ
        V_offset_bytes = sizeof(Float32) * layout.V_offset
        P̃ᵢⱼ = Swizzled(CuDynamicSharedArray(Float32, (Bᵣ, Bᶜ), 0), Val(layout.swizzle))
        stats = CuDynamicSharedArray(Float32, (Bᵣ, Wᶜ), V_offset_bytes)
        𝕆ᵢ = Tile{($R,$C,$No)}(0f0)
        mᵢ = Tile{($R,)}(-floatmax(Float32))
        ℓᵢ = Tile{($R,)}(0f0)

        # Each thread transfers one contiguous channel vector.
        vectors_per_row = Int32(stage ÷ width)
        d = Int32(width) * (τ % vectors_per_row) + 1
        row = τ ÷ vectors_per_row + 1
        row_step = Int32(layout.threads) ÷ vectors_per_row
        query = blockᵢ + row
        qstride, kvstride = D*H, D*Hkv
        # Anchor inside each head/document
        Q₀ = (@inbounds LinearIndices(Q)[1,head,1,document]) + (d-1) + qstride*(query-1)
        K₀ = (@inbounds LinearIndices(K)[1,kv_head,1,document]) + (d-1) + kvstride*(row-1)
        V₀ = @inbounds LinearIndices(V)[1,kv_head,1,document]
        vslots = @ntuple $Nv j -> begin
            t = τ + Int32((j-1)*layout.threads)
            dᵥ = Int32(width) * (t % Int32(Dᵥ÷width)) + 1
            rowᵥ = t ÷ Int32(Dᵥ÷width) + 1
            (dᵥ, rowᵥ, V₀+(dᵥ-1)+kvstride*(rowᵥ-1))
        end
        first_key = causal ? 0i32 : Int32(max(0, fld(blockᵢ-left, Bᶜ)*Bᶜ))
        last_key = causal ? min(blockᵢ+Bᵣ-1, T-1) :
            min(blockᵢ+Bᵣ-1+right, T-1)

        for blockⱼ in first_key:Int32(Bᶜ):last_key
            qᵥ = @ntuple $Nq j ->
                load(Float32ᵛ{width}, Q, Q₀+(j-1)*qstride*row_step, query+(j-1)*row_step <= T, d, Val(D), Val(stage))
            kᵥ = @ntuple $Nk j ->
                load(Float32ᵛ{width}, K, K₀+kvstride*(blockⱼ+(j-1)*row_step), blockⱼ+row+(j-1)*row_step <= T, d, Val(D), Val(stage))
            Qᵢ, Kⱼ = qk_buffers(0, Val(layout))
            stage_rows!(Qᵢ, qᵥ, row, row_step, d, Val(Bᵣ))
            stage_rows!(Kⱼ, kᵥ, row, row_step, d, Val(Bᶜ))
            sync_threads()

            # Sᵢⱼ = QᵢKⱼ'. 
            S = Tile{($R,$C)}(0f0)
            for channel₀ in 0i32:Int32(stage):(padded_D-1i32)
                next_channel = (channel₀ + Int32(stage)) % padded_D
                qᵥ = @ntuple $Nq j ->
                    load(Float32ᵛ{width}, Q, Q₀+next_channel+(j-1)*qstride*row_step, query+(j-1)*row_step <= T, d+next_channel, Val(D), Val(stage))
                kᵥ = @ntuple $Nk j ->
                    load(Float32ᵛ{width}, K, K₀+next_channel+kvstride*(blockⱼ+(j-1)*row_step), blockⱼ+row+(j-1)*row_step <= T, d+next_channel, Val(D), Val(stage))
                @inbounds S = muladd(
                    pointer(Qᵢ,row₀+1,1), pointer(Kⱼ,col₀+1,1), S,
                    Val(Bᵣ), Val(Bᶜ), Val(:swizzled), Val(layout),
                )
                buffer_offset = ((channel₀ ÷ stage + 1) % 2) * qk_stage_bytes
                Qᵢ, Kⱼ = qk_buffers(buffer_offset, Val(layout))
                stage_rows!(Qᵢ, qᵥ, row, row_step, d, Val(Bᵣ))
                stage_rows!(Kⱼ, kᵥ, row, row_step, d, Val(Bᶜ))
                sync_threads()
            end

            # Online softmax; maxima use base-2 units until saved to m.
            @inbounds begin
                for r in 1:$R
                    queryᵣ = query₀ + (r-1)
                    for c in 1:$C
                        key = blockⱼ + col₀ + (c-1)
                        valid = causal ? queryᵣ < T && key <= queryᵣ :
                            queryᵣ < T && key < T && queryᵣ-left <= key <= queryᵣ+right
                        # Fast-math compilation assumes finite operands.
                        S[r,c] = valid ? S[r,c] * scale : -floatmax(Float32)
                    end
                    m̃ᵢⱼ = reduce_lanes(max, maximum(S[r,:]), Val(layout.lanes[2]))
                    if laneᶜ == 0
                        stats[row₀+r, warpᶜ+1] = m̃ᵢⱼ
                    end
                end
                sync_threads()
                for r in 1:$R
                    mᵢⁿᵉʷ = reduce_row(max, mᵢ[r], stats, row₀+r, Val(Wᶜ))
                    αᵢ = exp2(mᵢ[r] - mᵢⁿᵉʷ)
                    mᵢ[r] = mᵢⁿᵉʷ
                    S[r,:] = exp2.(S[r,:] .- mᵢⁿᵉʷ)
                    𝕆ᵢ[r,:,:] = 𝕆ᵢ[r,:,:] .* αᵢ
                    for c in 1:$C
                        P̃ᵢⱼ[row₀+r, col₀+c] = S[r,c]
                    end
                    ℓᵢ[r] = αᵢ * ℓᵢ[r] + foldl(+, S[r,:])
                end
                sync_threads()
            end

            # 𝕆ᵢ += P̃ᵢⱼVⱼ'. 
            Vⱼ = v_buffer(0, Val(layout))
            tile_offset = kvstride*blockⱼ
            vᵥ = @ntuple $Nv j -> begin
                dᵥ, rowᵥ, address = vslots[j]
                load(Float32ᵛ{width}, V, address+tile_offset, blockⱼ+rowᵥ <= T, dᵥ, Val(D), Val(Dᵥ))
            end
            stage_slots!(Vⱼ, vᵥ, vslots, Val(stage), Val(Dᵥ), Val(layout.threads))
            sync_threads()
            for key₀ in 0i32:Int32(stage):Int32(Bᶜ-1)
                next_key = (key₀+Int32(stage)) % Int32(Bᶜ)
                # Share the tile offset
                tile_offset = kvstride*(blockⱼ+next_key)
                vᵥ = @ntuple $Nv j -> begin
                    dᵥ, rowᵥ, address = vslots[j]
                    load(Float32ᵛ{width}, V, address+tile_offset, blockⱼ+rowᵥ+next_key <= T, dᵥ, Val(D), Val(Dᵥ))
                end
                for n in 1:$No
                    @inbounds 𝕆ᵢ[:,:,n] = muladd(
                        pointer(P̃ᵢⱼ,row₀+1,key₀+1),
                        pointer(Vⱼ,1+col₀+(n-1)*Bᶜ), 𝕆ᵢ[:,:,n], Val(Bᵣ), Val(Dᵥ), Val(:linear), Val(layout),
                    )
                end
                buffer_offset = ((key₀ ÷ stage + 1) % 2) * v_stage_bytes
                Vⱼ = v_buffer(buffer_offset, Val(layout))
                stage_slots!(Vⱼ, vᵥ, vslots, Val(stage), Val(Dᵥ), Val(layout.threads))
                sync_threads()
            end
        end

        # Combine row sums across column warps, normalize, and save backward state.
        @inbounds begin
            for r in 1:$R
                ℓᵢ[r] = reduce_lanes(+, ℓᵢ[r], Val(layout.lanes[2]))
                if laneᶜ == 0
                    stats[row₀+r, warpᶜ+1] = ℓᵢ[r]
                end
            end
            sync_threads()
            for r in 1:$R
                queryᵣ = query₀+r
                ℓᵣ = reduce_row(+, 0f0, stats, row₀+r, Val(Wᶜ))
                inverse = inv(ℓᵣ)
                for n in 1:$No, c in 1:$C
                    channel = col₀+c+(n-1)*Bᶜ
                    if queryᵣ <= T && (D % Bᶜ == 0 || channel <= D)
                        𝕆[channel,head,queryᵣ,document] = 𝕆ᵢ[r,c,n] * inverse
                    end
                end
                if laneᶜ == 0 && warpᶜ == 0 && queryᵣ <= T
                    ℓ[1,queryᵣ,head,document] = ℓᵣ
                    m[1,queryᵣ,head,document] = mᵢ[r] / log₂e
                end
            end
        end
        return nothing
    end)
end


# ── FP32 FlashAttention backward ────────────────────────────────────────────

"""
    Δflash_attention₁!(dQ, dK, dV, dO, Q, K, V, O, ℓ, m, window, ::Val{Warps}=Val((2,2)); simt=Val(SIMT))

Accumulate gradients using the forward pass's saved O, ℓ, and m. 
Here the warp layout describes key × query tiles
"""
function Δflash_attention₁!(
    dQ::CuArray{Float32,4}, dK::CuArray{Float32,4}, dV::CuArray{Float32,4},
    dO::CuArray{Float32,4}, Q::CuArray{Float32,4}, K::CuArray{Float32,4},
    V::CuArray{Float32,4}, O::CuArray{Float32,4},
    ℓ::CuArray{Float32,4}, m::CuArray{Float32,4}, window::Tuple{Int,Int},
    ::Val{Warps}=Val((2,2)); simt::Val{Spec}=Val(SIMT),
) where {Warps,Spec}
    D, H, T, B = size(Q)
    layout = attention_layout(D, Warps, Spec)
    Bᶜ, Bᵣ = layout.tile
    query_warps = (Bᵣ, Bᶜ) .÷ layout.warp_tile
    query_layout = attention_layout(D, query_warps, Spec)
    shared = max(layout.stage_size,
        Bᶜ*Bᵣ + layout.stage*max(layout.Dᵥ, query_layout.Dᵥ))
    Δ = similar(ℓ)

    @cuda threads=layout.threads blocks=(cld(T,layout.threads÷32),H,B) flash_attention₁_rows!(Δ, dO, O)
    @cuda threads=layout.threads blocks=(cld(T,Bᶜ),size(K,2),B) shmem=sizeof(Float32)*shared fastmath=true Δflash_attention₁_kernel!(
        dQ, dK, dV, dO, Q, K, V, ℓ, m, Δ, Val(window), Val(layout), Val(query_layout))
    CUDA.unsafe_free!(Δ)
    return nothing
end

# One warp per query: Dᵢ = rowsum(dOᵢ ∘ Oᵢ)
function flash_attention₁_rows!(Δ, dO, O)
    τ = threadIdx().x - 1i32
    lane, warp = τ % 32i32, τ ÷ 32i32
    i, head, document = blockIdx()
    query = (i-1i32)*(blockDim().x÷32i32) + warp + 1i32
    value = 0f0
    if query <= size(O,3)
        @inbounds for d in lane+1i32:32i32:Int32(size(O,1))
            value = muladd(dO[d,head,query,document], O[d,head,query,document], value)
        end
    end
    value = reduce_lanes(+, value, Val(32))
    if lane == 0 && query <= size(O,3)
        @inbounds Δ[1,query,head,document] = value
    end
    return nothing
end

# Channel vectors from HBM → token rows in a swizzled shared stage.
@inline function stage_channels!(tile, A, head, document, token₀, channel₀, ::Val{Rows}, ::Val{L}) where {Rows,L}
    τ = threadIdx().x - 1i32
    width, stage, D = L.vector, L.stage, L.D
    d = Int32(width)*(τ % Int32(stage÷width)) + 1i32
    row = τ ÷ Int32(stage÷width) + 1i32
    row_step = Int32(L.threads÷(stage÷width))
    stride = Int32(D*size(A,2))
    address = (@inbounds LinearIndices(A)[1,head,1,document]) + d-1 + channel₀ + stride*(token₀+row-1)
    values = ntuple(Val(cld(stage*Rows,width*L.threads))) do j
        load(Float32ᵛ{width}, A, address+(j-1)*stride*row_step,
            token₀+row+(j-1)*row_step <= size(A,3), d+channel₀, Val(D), Val(stage))
    end
    stage_rows!(tile, values, row, row_step, d, Val(Rows))
    return nothing
end

# Token stage from HBM → contiguous, padded channels for dS·Q / P·dO / dS'·K.
@inline function stage_values!(tile, A, head, document, token₀, ::Val{L}) where L
    τ = threadIdx().x - 1i32
    width, pitch = L.vector, L.Dᵥ
    stride = Int32(L.D*size(A,2))
    address = @inbounds LinearIndices(A)[1,head,1,document]
    @loopinfo unroll for j in 0:cld(L.stage*pitch,width*L.threads)-1
        index = τ + Int32(j*L.threads)
        d = Int32(width)*(index % Int32(pitch÷width)) + 1i32
        row = index ÷ Int32(pitch÷width) + 1i32
        if L.stage*pitch % (width*L.threads) == 0 || row <= L.stage
            value = load(Float32ᵛ{width}, A, address+d-1+stride*(token₀+row-1),
                token₀+row <= size(A,3), d, Val(L.D), Val(pitch))
            @inbounds store!(tile, d+pitch*(row-1), value)
        end
    end
    return nothing
end

@generated function Δflash_attention₁_kernel!(
    dQ, dK, dV, dO, Q, K, V, ℓ, m, Δ, ::Val{Window}, ::Val{L}, ::Val{LQ},
) where {Window,L,LQ}
    R, C = L.thread_tile
    Bᶜ, Bᵣ = L.tile
    Nk, Nq = cld(L.D,Bᵣ), cld(L.D,Bᶜ)
    return :(@tile begin
        layout, query_layout = L, LQ
        D, stage = layout.D, layout.stage
        Bᶜ, Bᵣ = layout.tile
        τ = threadIdx().x - 1i32
        lane, warp = τ % 32i32, τ ÷ 32i32
        laneᵣ, laneᶜ = lane ÷ Int32(layout.lanes[2]), lane % Int32(layout.lanes[2])
        row₀ = Int32(layout.warp_tile[1])*(warp÷Int32(layout.warps[2])) + Int32($R)*laneᵣ
        col₀ = Int32(layout.warp_tile[2])*(warp%Int32(layout.warps[2])) + Int32($C)*laneᶜ
        query₀ = Int32(query_layout.warp_tile[1])*(warp÷Int32(query_layout.warps[2])) + Int32($R)*laneᵣ
        channel₀ = Int32(query_layout.warp_tile[2])*(warp%Int32(query_layout.warps[2])) + Int32($C)*laneᶜ
        j, kv_head, document = blockIdx()
        blockⱼ = Int32(Bᶜ)*(j-1i32)
        T, H, Hkv = Int32(size(Q,3)), Int32(size(Q,2)), Int32(size(K,2))
        heads_per_kv = H÷Hkv
        left, right = Window
        causal = Window == (-1,0)
        scale = inv(sqrt(Float32(D)))
        log₂e = 1.4426950408889634f0

        Kⱼ, Qᵢ = qk_buffers(0, Val(layout))
        Pᵢⱼ = Swizzled(CuDynamicSharedArray(Float32,(Bᶜ,Bᵣ),0), Val(layout.swizzle))
        dSᵢⱼ = Pᵢⱼ
        dSᵀᵢⱼ = Swizzled(CuDynamicSharedArray(Float32,(Bᵣ,Bᶜ),0), Val(layout.swizzle))
        operand = CuDynamicSharedArray(Float32,(layout.Dᵥ,stage),sizeof(Float32)*Bᶜ*Bᵣ)
        key_operand = CuDynamicSharedArray(Float32,(query_layout.Dᵥ,stage),sizeof(Float32)*Bᶜ*Bᵣ)
        dKⱼ = Tile{($R,$C,$Nk)}(0f0)
        dVⱼ = Tile{($R,$C,$Nk)}(0f0)
        first_query = causal ? fld(blockⱼ,Bᵣ)*Bᵣ : max(0i32,fld(blockⱼ-right,Bᵣ)*Bᵣ)
        last_query = causal ? T-1i32 : min(T-1i32,blockⱼ+Bᶜ-1+left)

        for head in (kv_head-1)*heads_per_kv+1:kv_head*heads_per_kv
            for blockᵢ in Int32(first_query):Int32(Bᵣ):Int32(last_query)
                S = Tile{($R,$C)}(0f0)
                dP = Tile{($R,$C)}(0f0)
                for d₀ in 0i32:Int32(stage):Int32(D-1)
                    # S = KQ'; dP = VdO'. Shared stages are reused after each product.
                    stage_channels!(Kⱼ,K,kv_head,document,blockⱼ,d₀,Val(Bᶜ),Val(layout))
                    stage_channels!(Qᵢ,Q,head,document,blockᵢ,d₀,Val(Bᵣ),Val(layout))
                    sync_threads()
                    @inbounds S = muladd(pointer(Kⱼ,row₀+1,1),pointer(Qᵢ,col₀+1,1),S,
                        Val(Bᶜ),Val(Bᵣ),Val(:swizzled),Val(layout))
                    sync_threads()
                    stage_channels!(Kⱼ,V,kv_head,document,blockⱼ,d₀,Val(Bᶜ),Val(layout))
                    stage_channels!(Qᵢ,dO,head,document,blockᵢ,d₀,Val(Bᵣ),Val(layout))
                    sync_threads()
                    @inbounds dP = muladd(pointer(Kⱼ,row₀+1,1),pointer(Qᵢ,col₀+1,1),dP,
                        Val(Bᶜ),Val(Bᵣ),Val(:swizzled),Val(layout))
                    sync_threads()
                end

                @inbounds for c in 1:$C
                    query = blockᵢ+col₀+c
                    mᵢ = query <= T ? m[1,query,head,document] : 0f0
                    inverse = query <= T ? inv(ℓ[1,query,head,document]) : 0f0
                    Dᵢ = query <= T ? Δ[1,query,head,document] : 0f0
                    for r in 1:$R
                        key = blockⱼ+row₀+r
                        valid = key <= T && query <= T &&
                            (causal ? key <= query : query-left <= key <= query+right)
                        p = valid ? exp2((S[r,c]*scale-mᵢ)*log₂e)*inverse : 0f0
                        Pᵢⱼ[row₀+r,col₀+c] = p
                        # dP's registers now hold the scaled score derivative dS.
                        dP[r,c] = scale*p*(dP[r,c]-Dᵢ)
                    end
                end
                sync_threads()

                # dV += P·dO. Keep dS in registers until P's last use.
                for q₀ in 0i32:Int32(stage):Int32(Bᵣ-1)
                    stage_values!(operand,dO,head,document,blockᵢ+q₀,Val(layout))
                    sync_threads()
                    for n in 1:$Nk
                        @inbounds dVⱼ[:,:,n] = muladd(pointer(Pᵢⱼ,row₀+1,q₀+1),
                            pointer(operand,col₀+1+(n-1)*Bᵣ),dVⱼ[:,:,n],
                            Val(Bᶜ),Val(layout.Dᵥ),Val(:linear),Val(layout))
                    end
                    sync_threads()
                end
                @inbounds for c in 1:$C, r in 1:$R
                    dSᵢⱼ[row₀+r,col₀+c] = dP[r,c]
                end
                sync_threads()

                # dK += dS·Q. dS already contains the softmax scale.
                for q₀ in 0i32:Int32(stage):Int32(Bᵣ-1)
                    stage_values!(operand,Q,head,document,blockᵢ+q₀,Val(layout))
                    sync_threads()
                    for n in 1:$Nk
                        @inbounds dKⱼ[:,:,n] = muladd(pointer(dSᵢⱼ,row₀+1,q₀+1),
                            pointer(operand,col₀+1+(n-1)*Bᵣ),dKⱼ[:,:,n],
                            Val(Bᶜ),Val(layout.Dᵥ),Val(:linear),Val(layout))
                    end
                    sync_threads()
                end

                # Transpose from registers into the same tile; dQ += dS'·K.
                @inbounds for c in 1:$C, r in 1:$R
                    dSᵀᵢⱼ[col₀+c,row₀+r] = dP[r,c]
                end
                sync_threads()
                dQᵢ = Tile{($R,$C,$Nq)}(0f0)
                for k₀ in 0i32:Int32(stage):Int32(Bᶜ-1)
                    stage_values!(key_operand,K,kv_head,document,blockⱼ+k₀,Val(query_layout))
                    sync_threads()
                    for n in 1:$Nq
                        @inbounds dQᵢ[:,:,n] = muladd(pointer(dSᵀᵢⱼ,query₀+1,k₀+1),
                            pointer(key_operand,channel₀+1+(n-1)*Bᶜ),dQᵢ[:,:,n],
                            Val(Bᵣ),Val(query_layout.Dᵥ),Val(:linear),Val(query_layout))
                    end
                    sync_threads()
                end
                @inbounds for n in 1:$Nq, c in 1:$C, r in 1:$R
                    query, channel = blockᵢ+query₀+r, channel₀+c+(n-1)*Bᶜ
                    if query <= T && channel <= D
                        index = LinearIndices(dQ)[channel,head,query,document]
                        CUDA.atomic_add!(pointer(dQ,index), dQᵢ[r,c,n])
                    end
                end
                sync_threads()
            end
        end
        @inbounds for n in 1:$Nk, c in 1:$C, r in 1:$R
            key, channel = blockⱼ+row₀+r, col₀+c+(n-1)*Bᵣ
            if key <= T && channel <= D
                dK[channel,kv_head,key,document] += dKⱼ[r,c,n]
                dV[channel,kv_head,key,document] += dVⱼ[r,c,n]
            end
        end
        return nothing
    end)
end

# ── Tensor Core attention ──────────────────────────────────────────────────

# Four raw words: eight 16-bit values moved without numeric conversion.
const Word4 = NTuple{4,VecElement{Int32}}

function copy! end

# Packed operands retain their numeric format across word loads and slices.
struct Packed{F,N}
    words::NTuple{N,UInt32}
end
@inline Packed{F}(words::NTuple{N,UInt32}) where {F,N} = Packed{F,N}(words)
Base.eltype(::Packed{F}) where F = F
Base.Tuple(A::Packed) = A.words
@inline Base.getindex(A::Packed,i::Integer) = A.words[i]
@inline Base.getindex(A::Packed{F},I::Tuple) where F = Packed{F}(map(i->A[i],I))

@inline pack(x::F,y::F) where {F<:TensorFloat} = UInt32(reinterpret(UInt16,x)) | (UInt32(reinterpret(UInt16,y)) << 16)
@inline pack(::Type{F},x,y) where {F<:TensorFloat} = pack(F(x),F(y))

@inline shared_pointer(A,row,column) = @inbounds pointer(A,row+(column-1)*size(A,1))
@inline shared_pointer(A::Swizzled,row,column) = @inbounds pointer(A,row,column)

# ldmatrix consumes a 32-bit byte address in the shared-memory address space.
@inline shared_address(A,row,column) = reinterpret(UInt,shared_pointer(A,row,column)) % UInt32

# Convergent native instructions return an LLVM struct; bridge it explicitly to
# Julia's tuple representation so whole kernels do not acquire C-ABI helper calls.
function tuple_intrinsic(prepare,name,Return,Arguments,input_types,argument_values...)
    LLVM.Context() do _
        result_type = convert(LLVM.LLVMType,Return)
        f,_ = create_function(result_type,[convert(LLVM.LLVMType,T) for T in Arguments])
        signature = LLVM.FunctionType(LLVM.StructType([convert(LLVM.LLVMType,T) for T in Return.parameters]),input_types())
        instruction = LLVM.Function(LLVM.parent(f),name,signature)
        push!(LLVM.function_attributes(instruction),LLVM.EnumAttribute("convergent"))
        LLVM.IRBuilder() do builder
            LLVM.position!(builder,LLVM.BasicBlock(f,"entry"))
            values = LLVM.call!(builder,signature,instruction,prepare(builder,LLVM.parameters(f)))
            push!(LLVM.function_attributes(values),LLVM.EnumAttribute("convergent"))
            result = LLVM.UndefValue(result_type)
            for i in 0:fieldcount(Return)-1
                result = LLVM.insert_value!(builder,result,LLVM.extract_value!(builder,values,i),i)
            end
            LLVM.ret!(builder,result)
        end
        call_function(f,Return,Tuple{Arguments...},argument_values...)
    end
end

@inline @generated function load_matrix(address::UInt32,::Val{N},::Val{Transpose}) where {N,Transpose}
    name = "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x$N$(Transpose ? ".trans" : "").b16.p3"
    # Types are created within the builder's LLVM context, not at runtime.
    tuple_intrinsic(name,NTuple{N,UInt32},(UInt32,),() -> [LLVM.PointerType(LLVM.Int8Type(),3)],:address) do builder,args
        [LLVM.inttoptr!(builder,only(args),LLVM.PointerType(LLVM.Int8Type(),3))]
    end
end

@inline load_matrix(::Type{F},address,count,transpose) where {F<:TensorFloat} =
    Packed{F}(load_matrix(address,count,transpose))

# Matrix operands take zero-based token/channel offsets; shared indices are one-based.
@inline query_matrix(Q,q₀,d₀,lane) = load_matrix(eltype(Q),
    shared_address(Q,d₀+8*(lane÷16)+1,q₀+lane%16+1),Val(4),Val(false))
@inline key_matrix(K,k₀,d₀,lane) = load_matrix(eltype(K),
    shared_address(K,d₀+8*((lane÷8)%2)+1,k₀+lane%8+1),Val(2),Val(false))
@inline value_matrix(V,k₀,d₀,lane) = load_matrix(eltype(V),
    shared_address(V,d₀+1,k₀+lane%16+1),Val(2),Val(true))

# NVIDIA m16n8k16: four FP32 accumulators; packed operands retain input format.
@inline @generated function Base.muladd(A::Packed{F,4},B::Packed{F,2},C::NTuple{4,Float32}) where {F<:TensorFloat}
    suffix = F == Float16 ? "f32.f32" : "bf16"
    tuple_intrinsic("llvm.nvvm.mma.m16n8k16.row.col.$suffix",NTuple{4,Float32},
        (NTuple{4,UInt32},NTuple{2,UInt32},NTuple{4,Float32}),
        () -> [fill(F == Float16 ? LLVM.VectorType(LLVM.HalfType(),2) : LLVM.Int32Type(),6);fill(LLVM.FloatType(),4)],
        :(Tuple(A)),:(Tuple(B)),:C) do builder,args
        a,b,c = args
        operand = F == Float16 ? LLVM.VectorType(LLVM.HalfType(),2) : LLVM.Int32Type()
        packed = [LLVM.bitcast!(builder,LLVM.extract_value!(builder,x,i),operand) for x in (a,b) for i in 0:(x === a ? 3 : 1)]
        LLVM.Value[packed;[LLVM.extract_value!(builder,c,i) for i in 0:3]]
    end
end

# Approximate only the exponential, not the surrounding masked-tail Inf checks.
@inline exp₂(x::Float32) = ccall("llvm.nvvm.ex2.approx.ftz.f",llvmcall,Float32,(Float32,),x)

const LOG₂E = Float32(log2(exp(1.0)))

# dS stores two neighboring queries for one key in a single shared word.
@inline function store!(A,query,key,value::UInt32)
    @inbounds address = pointer(A,query+(key-1)*size(A,1))
    output = reinterpret(Core.LLVMPtr{UInt32,CUDA.AS.Shared},address)
    unsafe_store!(output,value,1,Val(4))
    nothing
end

# Physical dS is query × key. ldmatrix.trans restores the identical logical
# query-row × key-channel A tile used by the fifth tensor multiply.
@inline transposed_query_matrix(A,q₀,k₀,lane) = load_matrix(eltype(A),
    shared_address(A,q₀+8*((lane÷8)%2)+1,k₀+8*(lane÷16)+lane%8+1),Val(4),Val(true))

@inline commit_copies!() = CUDA.CG.pipeline_commit()

# Waiting completes this thread's copies; the barrier also makes the completed
# copies and synchronous zero stores visible to all threads in the block.
@inline function wait_copies!(::Val{Remaining}=Val(0)) where Remaining
    CUDA.CG.pipeline_wait_prior(Remaining)
    sync_threads()
    nothing
end


# Slices of a score tile already have the next multiply's A layout.
@inline probability_matrix(::Type{F},S,k) where F = Packed{F}((
    pack(F,S[1,2k-1],S[2,2k-1]),pack(F,S[3,2k-1],S[4,2k-1]),
    pack(F,S[1,2k],S[2,2k]),pack(F,S[3,2k],S[4,2k])))

# Zero-based lane/query-tile; one-based register/channel-chunk/head/document.
@inline function tile_index(lane,r,d,block₀,head,document,tiles,heads,::Val{D}) where D
    lane+Int32(1)+Int32(32)*(Int32(r)-Int32(1)+Int32(4)*(d-Int32(1)+
        Int32(D÷8)*(block₀+tiles*(head-Int32(1)+heads*(document-Int32(1))))))
end


# Unpadded shared storage and one query-owned forward implementation.
# Static slices in every arm: runtime selection returns four scalar registers,
# not a dynamically addressed NTuple or stack-backed probability array.
# Wrap after selection so LLVM can simplify the packed-word branch joins.
@inline probability_tile(::Type{F},P,k) where F = Packed{F}(probability_tile(P,k))

@inline @generated function probability_tile(P::Tile{Shape,UInt32},k::Int32) where Shape
    result=:(P[:,$(Shape[2])])
    for i in Shape[2]-1:-1:1
        result=:(if k==$(Int32(i)); P[:,$i]; else; $result; end)
    end
    result
end

# Reuse the existing logical matrix wrapper with an owned aligned-word spec.
# Its default SIMT specification and dispatch remain unchanged.
struct WordSwizzle{Width,Mask} end
@inline swizzle(row,column,::Val{WordSwizzle{Width,Mask}}) where {Width,Mask} =
    ((row-1) ⊻ (Width*((column-1)&Mask)))+1
# Four words contain the B operands for two adjacent eight-column MMAs.
@inline key_matrices(K::Swizzled,k₀,d₀,lane) = load_matrix(eltype(K),
    shared_address(K,d₀+8*((lane÷8)%2)+1,k₀+8*(lane÷16)+lane%8+1),Val(4),Val(false))
@inline value_matrices(V::Swizzled,k₀,d₀,lane) = load_matrix(eltype(V),
    shared_address(V,d₀+8*(lane÷16)+1,k₀+lane%16+1),Val(4),Val(true))

# One word belongs to one thread. Synchronous copies interleave load/store;
# asynchronous copies are committed and waited for by the owning algorithm.
@inline function copy!(output::Core.LLVMPtr{Word4,CUDA.AS.Shared},input,valid)
    zero = ntuple(_->VecElement(Int32(0)),Val(4))
    value = valid ? CUDA.unsafe_cached_load(input,1,Val(16)) : zero
    unsafe_store!(output,value,1,Val(16))
    nothing
end

@inline function copy_async!(output::Core.LLVMPtr{Word4,CUDA.AS.Shared},input,valid)
    if valid
        CUDA.CG.pipeline_memcpy_async(output,input)
    else
        unsafe_store!(output,ntuple(_->VecElement(Int32(0)),Val(4)),1,Val(16))
    end
    nothing
end

# A copy owns matching matrix tuples. The layouts own their shared addresses;
# all sources share (D,H,T,B), so channel/head/token/document offsets agree.
@inline matrices(A) = (A,)
@inline matrices(A::Tuple) = A

@inline function copy_rows!(copy!::F,destination,source,head,token₀,document,
    ::Val{D},::Val{Rows},::Val{W},
) where {F,D,Rows,W}
    destinations,sources = matrices(destination),matrices(source)
    thread = threadIdx().x
    first_source = first(sources)
    T = size(first_source,3)%Int32
    pitch = D*size(first_source,2)
    base = @inbounds LinearIndices((D,Base.tail(size(first_source))...))[1,head,1,document]
    @loopinfo unroll=false for step in 0i32:Int32(cld(D÷8*Rows,32W)-1)
        index = thread+step*Int32(32W)
        if D÷8*Rows % (32W) == 0 || index <= Int32(D÷8*Rows)
            channel = 8i32*((index-1i32)%Int32(D÷8))+1i32
            row = (index-1i32)÷Int32(D÷8)+1i32
            global_index = base+(channel-1i32)+(token₀+row-1i32)*pitch
            @loopinfo unroll=true for pair in 1:length(sources)
                destination,source = destinations[pair],sources[pair]
                output = reinterpret(Core.LLVMPtr{Word4,CUDA.AS.Shared},shared_pointer(destination,channel,row))
                input = reinterpret(Core.LLVMPtr{Word4,CUDA.AS.Global},@inbounds pointer(source,global_index))
                copy!(output,input,token₀+row<=T)
            end
        end
    end
    nothing
end

function forward_layout(D,W,U,C)
    R = 16W*U
    Qbytes,Kbytes = sizeof(UInt16)*D*R,sizeof(UInt16)*D*C
    (;tile=(R,C),threads=32W,bytes=Qbytes+2Kbytes,
      offsets=(Q=0,K=Qbytes,V=Qbytes+Kbytes))
end

shared_bytes(D,W,U,C) = forward_layout(D,W,U,C).bytes

# C bounds score registers per lane; shared capacity targets two resident CTAs.
# The coverage adjustment changes only query ownership, not the reduction tile.
function forward_geometry(D,H,T,B,SM,budget;
    query_subtiles=1,key_tile=64,
)
    U,C = query_subtiles,key_tile
    W = 8
    while W > 1 && shared_bytes(D,W,U,C) > budget
        W ÷= 2
    end
    while W > 1 && H*B*cld(T,16W*U) < SM
        W ÷= 2
    end
    (;warps=W,query_subtiles,key_tile)
end

# Opt in to the kernel's requested dynamic shared-memory capacity.
function shared_memory!(kernel,bytes)
    attributes = CUDA.attributes(kernel.fun)
    if bytes > attributes[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES]
        attributes[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = bytes
    end
    nothing
end

"""
    attention!(TensorCoreInstruction(), O, ℓ, m, Q, K, V, window; ...)

Float16/BFloat16 Q/K/V with Float32 accumulation and saved statistics. Output
uses the input format or Float32; the kernel converts only at its final store.
Compatible nonempty, nonaliasing buffers and valid grouped heads are expected.
"""
function attention!(𝒜::TensorCoreInstruction,
    O::CuArray{F,4},ℓ::CuArray{Float32,4},m::CuArray{Float32,4},
    Q::CuArray{E,4},K::CuArray{E,4},V::CuArray{E,4},
    window::Tuple{Int,Int};
    warps=nothing,query_subtiles::Val{U}=Val(1),key_tile::Val{C}=Val(64),order=nothing,
) where {E<:TensorFloat,F<:Union{E,Float32},U,C}
    D,H,T,B = size(Q)
    dev = device(Q)
    budget = min(attribute(dev,CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN),
        attribute(dev,CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)÷2)
    SM = attribute(dev,CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    selected = forward_geometry(D,H,T,B,SM,budget;query_subtiles=U,key_tile=C)
    warps = something(warps,Val(selected.warps))
    # Interleave heads while K/V fits in L2; otherwise keep neighboring query
    # blocks on the same K/V stream. This changes ownership, not the arithmetic.
    l2 = attribute(dev,CUDA.DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
    order = something(order,Val(sizeof(K)+sizeof(V) > l2 ? :query : :head))
    launch_attention!(O,ℓ,m,Q,K,V,window,warps,query_subtiles,key_tile,order)
end

function launch_attention!(O,ℓ,m,Q,K,V,window,warps::Val{W},subtiles::Val{U},key_tile::Val{C},
    order::Val{Order}=Val(:head),
) where {W,U,C,Order}
    D,H,T,B = size(Q)
    @assert 0 < D <= 128 && D%16 == 0 && C > 0 && C%16 == 0 && U in (1,2) && 0 < W <= 32
    layout = forward_layout(D,W,U,C)
    R = layout.tile[1]
    args = (TensorCoreInstruction(),O,ℓ,m,Q,K,V,Val(window),Val(D),warps,subtiles,key_tile,
        Val(H÷size(K,2)),Val((D,H)),Val((D,size(K,2))),order)
    kernel = @cuda launch=false maxthreads=layout.threads attention_forward!(args...)
    shared_memory!(kernel,layout.bytes)
    blocks = Order == :query ? (cld(T,R),B,H) : (H,B,cld(T,R))
    kernel(args...;threads=layout.threads,blocks,shmem=layout.bytes)
    nothing
end

@inline function static_shape(A::CuDeviceArray{F,N,AS},shape) where {F,N,AS}
    CuDeviceArray{F,N,AS}(pointer(A),shape,A.maxsize)
end

# Only channel/head strides specialize; token and batch dimensions stay runtime.
function attention_forward!(instruction::TensorCoreInstruction,O,ℓ,m,Q,K,V,window,D,warps,subtiles,key_tile,groups,
    ::Val{QChannels},::Val{KChannels},order::Val=Val(:head),
) where {QChannels,KChannels}
    O = static_shape(O,(QChannels...,size(O,3),size(O,4)))
    Q = static_shape(Q,(QChannels...,size(Q,3),size(Q,4)))
    K = static_shape(K,(KChannels...,size(K,3),size(K,4)))
    V = static_shape(V,(KChannels...,size(V,3),size(V,4)))
    ℓ = static_shape(ℓ,(1,size(ℓ,2),QChannels[2],size(ℓ,4)))
    m = static_shape(m,(1,size(m,2),QChannels[2],size(m,4)))
    @inline attention_kernel!(instruction,O,ℓ,m,Q,K,V,window,D,warps,subtiles,key_tile,groups,order)
end

@generated function attention_kernel!(::TensorCoreInstruction,O,ℓ,m,Q,K,V,
    ::Val{Window},::Val{D},::Val{W},::Val{U},::Val{C},::Val{Groups},
    ::Val{Order}=Val(:head),
) where {Window,D,W,U,C,Groups,Order}
    (;tile,offsets) = forward_layout(D,W,U,C)
    R = tile[1]
    subtiles = U
    mask = (1<<min(3,trailing_zeros(D÷8)))-1
    quote
        x,document,z = blockIdx()
        head,i = $(Order == :query) ? (z,gridDim().x-x+1i32) : (x,gridDim().z-z+1i32)
        thread = threadIdx().x
        warp,lane = (thread-1i32)÷32i32,(thread-1i32)%32i32
        lane_row,lane_pair = lane÷4i32,lane%4i32
        T = size(Q,3)%Int32
        kv_head = (head-1i32)÷$(Int32(Groups))+1i32
        blockᵢ = (i-1i32)*$(Int32(R))
        left,right = $Window
        τ = $(Float32(log2(exp(1.0))/sqrt(D)))
        @inbounds begin
            Qᵢ = Swizzled(CuDynamicSharedArray(eltype(Q),($D,$R),$(offsets.Q)),Val(WordSwizzle{8,$mask}))
            Kⱼ = Swizzled(CuDynamicSharedArray(eltype(K),($D,$C),$(offsets.K)),Val(WordSwizzle{8,$mask}))
            Vⱼ = Swizzled(CuDynamicSharedArray(eltype(V),($D,$C),$(offsets.V)),Val(WordSwizzle{8,$mask}))
        end
        copy_rows!(copy!,Qᵢ,Q,head,blockᵢ,document,Val($D),Val($R),Val($W))
        sync_threads()
        𝕆₁ = Tile{(4,$(D÷8))}(0f0)
        𝕆₂ = 𝕆₁
        𝕞 = Tile{(2,2)}(-Inf32)
        𝕝 = Tile{(2,2)}(0f0)
        first_key = $Window == (-1,0) ? 0i32 : max(0i32,blockᵢ-left)÷$(Int32(C))*$(Int32(C))
        last_key = min(T,blockᵢ+$(Int32(R))+right)
        copy_rows!(copy_async!,Kⱼ,K,kv_head,first_key,document,Val($D),Val($C),Val($W))
        commit_copies!()
        for j in 0i32:cld(last_key-first_key,$(Int32(C)))-1i32
            blockⱼ = first_key+j*$(Int32(C))
            # Current K is ready; every reader of the previous V has retired.
            wait_copies!()
            copy_rows!(copy_async!,Vⱼ,V,kv_head,blockⱼ,document,Val($D),Val($C),Val($W))
            commit_copies!()

            # Both query halves consume the same K tile before it dies.
            # Keep their register representations separate rather than one 3D tuple.
            S₁ = Tile{(4,$(C÷8))}(0f0)
            S₂ = S₁
            @tile begin
                @loopinfo unroll=false for d in 0i32:16i32:$(Int32(D-16))
                    A₁ = query_matrix(Qᵢ,warp*16i32,d,lane)
                    A₂ = $subtiles == 2 ? query_matrix(Qᵢ,warp*16i32+$(Int32(16W)),d,lane) : A₁
                    for n in 1:$(C÷16)
                        B = key_matrices(Kⱼ,16(n-1),d,lane)
                        B₁,B₂ = B[(1,2)],B[(3,4)]
                        S₁[:,2n-1] = muladd(A₁,B₁,S₁[:,2n-1])
                        S₁[:,2n] = muladd(A₁,B₂,S₁[:,2n])
                        if $subtiles == 2
                            S₂[:,2n-1] = muladd(A₂,B₁,S₂[:,2n-1])
                            S₂[:,2n] = muladd(A₂,B₂,S₂[:,2n])
                        end
                    end
                end
            end
            # Current V completes while QK runs. The barrier also retires
            # every K reader before the single K buffer receives its next tile.
            wait_copies!()
            if blockⱼ+$(Int32(C)) < last_key
                copy_rows!(copy_async!,Kⱼ,K,kv_head,blockⱼ+$(Int32(C)),document,Val($D),Val($C),Val($W))
                commit_copies!()
            end
            𝒫₁ = Tile{(4,$(C÷16))}(UInt32(0))
            𝒫₂ = 𝒫₁
            @tile for u in 1:$subtiles
                𝕆 = u == 1 ? 𝕆₁ : 𝕆₂
                S = u == 1 ? S₁ : S₂
                q = warp*16i32+$(Int32(16W))*(u-1)
                query = blockᵢ+q+lane_row+1i32
                if $(Window == (-1,0)) && blockⱼ+$(Int32(C)) <= blockᵢ
                    for n in 1:$(C÷8),r in 1:4
                        S[r,n] *= τ
                    end
                else
                    for n in 1:$(C÷8),r in 1:4
                        key = blockⱼ+8(n-1)+2lane_pair+mod(r-1,2)+1
                        row = query+8*((r-1)÷2)
                        valid = row <= T && key <= T && ($Window == (-1,0) ? key <= row : row-left <= key <= row+right)
                        S[r,n] = valid ? S[r,n]*τ : -Inf32
                    end
                end
                for r in 1:2
                    a,b = 2r-1,2r
                    m̃ = -Inf32
                    for n in 1:$(C÷8)
                        m̃ = max(m̃,S[a,n],S[b,n])
                    end
                    m̃ = reduce_lanes(max,m̃,Val(4))
                    mⁿᵉʷ = max(𝕞[r,u],m̃)
                    safe = isfinite(mⁿᵉʷ) ? mⁿᵉʷ : 0f0
                    α = exp₂(𝕞[r,u]-safe)
                    ℓ̃ = 0f0
                    for n in 1:$(C÷8)
                        S[a,n] = exp₂(S[a,n]-safe)
                        S[b,n] = exp₂(S[b,n]-safe)
                        ℓ̃ += S[a,n]+S[b,n]
                    end
                    # Row sums stay local until final normalization.
                    𝕝[r,u] = α*𝕝[r,u]+ℓ̃
                    𝕞[r,u] = mⁿᵉʷ
                    for d in 1:$(D÷8)
                        𝕆[a,d] *= α
                        𝕆[b,d] *= α
                    end
                end
                # Finish the Float32 softmax phase before multiplying V.
                # Two probabilities share each UInt32 register; S is now dead.
                𝒫 = Tile{(4,$(C÷16))}(UInt32(0))
                for k in 1:$(C÷16)
                    𝒫[:,k] = probability_matrix(eltype(Q),S,k)
                end
                if u == 1
                    𝕆₁,𝒫₁ = 𝕆,𝒫
                else
                    𝕆₂,𝒫₂ = 𝕆,𝒫
                end
            end
            # The second half reuses V's two packed words, not another load.
            @tile begin
                @loopinfo unroll=false for k in 1i32:$(Int32(C÷16))
                    P₁ = probability_tile(eltype(Q),𝒫₁,k)
                    P₂ = $subtiles == 2 ? probability_tile(eltype(Q),𝒫₂,k) : P₁
                    for d in 1:$(D÷16)
                        B = value_matrices(Vⱼ,16i32*(k-1i32),16(d-1),lane)
                        B₁,B₂ = B[(1,2)],B[(3,4)]
                        𝕆₁[:,2d-1] = muladd(P₁,B₁,𝕆₁[:,2d-1])
                        𝕆₁[:,2d] = muladd(P₁,B₂,𝕆₁[:,2d])
                        if $subtiles == 2
                            𝕆₂[:,2d-1] = muladd(P₂,B₁,𝕆₂[:,2d-1])
                            𝕆₂[:,2d] = muladd(P₂,B₂,𝕆₂[:,2d])
                        end
                    end
                end
            end
            # The next iteration's wait/barrier retires all PV readers.
        end
        @tile for u in 1:$subtiles
            𝕝[1,u] = reduce_lanes(+,𝕝[1,u],Val(4))
            𝕝[2,u] = reduce_lanes(+,𝕝[2,u],Val(4))
            # Reuse one Float32 reciprocal per completed row; save the global ℓ.
            ℓ⁻¹ = (inv(𝕝[1,u]),inv(𝕝[2,u]))
            𝕆 = u == 1 ? 𝕆₁ : 𝕆₂
            query = blockᵢ+warp*16i32+$(Int32(16W))*(u-1)+lane_row+1i32
            for d in 1:$(D÷8),r in 1:4
                channel = 8(d-1)+2lane_pair+mod(r-1,2)+1
                row = query+8*((r-1)÷2)
                if row <= T
                    @inbounds O[channel,head,row,document] = 𝕆[r,d]*ℓ⁻¹[(r-1)÷2+1]
                end
            end
            if lane_pair == 0
                for r in 1:2
                    row = query+8*(r-1)
                    if row <= T
                        @inbounds ℓ[1,row,head,document] = 𝕝[r,u]
                        @inbounds m[1,row,head,document] = 𝕞[r,u]*$(Float32(log(2)))
                    end
                end
            end
        end
        nothing
    end
end

# Unified gradient storage preparation and final stores.
# The caller's value is added before its one final storage conversion.
# The compile-time overwrite case never reads the destination.
@inline gradient_value(destination,index,update,::Val{Accumulate}) where Accumulate =
    Accumulate ? Float32(@inbounds destination[index])+update : update

@inline function store_gradient!(destination,index,update,policy::Val)
    @inbounds destination[index] = gradient_value(destination,index,update,policy)
    nothing
end

# Both storage contracts share row statistics and padded scratch clearing.
# Nothing means no conversion buffer and no scale; those branches disappear.
function prepare_backward!(dO₁₆::Union{Nothing,CuDeviceArray},δQ,Δ,L,dO,O,ℓ,m,s,::Val{D}) where D
    lane,warp = (threadIdx().x-1i32)%32i32,(threadIdx().x-1i32)÷32i32
    i,head,document = blockIdx()
    query = (i-1i32)*(blockDim().x÷32i32)+warp+1i32
    T,H = size(O,3)%Int32,size(O,2)%Int32
    padded = 16i32*cld(T,16i32)
    if query <= padded
        offset = Int32(D)*(query-1i32+padded*(head-1i32+H*(document-1i32)))
        @inbounds for d in lane+1i32:32i32:Int32(D)
            δQ[offset+d] = 0f0
        end
    end
    scale = s === nothing ? 1f0 : (@inbounds s[1])
    value = 0f0
    if query <= T
        @inbounds for d in lane+1i32:32i32:Int32(D)
            δ = Float32(dO[d,head,query,document])*scale
            dO₁₆ === nothing || (dO₁₆[d,head,query,document] = δ)
            value = muladd(δ,Float32(O[d,head,query,document]),value)
        end
    end
    value = reduce_lanes(+,value,Val(32))
    if lane == 0 && query <= T
        @inbounds begin
            Δ[1,query,head,document] = value
            L[1,query,head,document] = muladd(m[1,query,head,document],LOG₂E,log2(ℓ[1,query,head,document]))
        end
    end
    nothing
end

@inline function load(::Type{Packed{F,N}},A::CuDeviceArray,index::Integer) where {F<:TensorFloat,N}
    words = load(NTuple{N,VecElement{Int32}},A,index)
    Packed{F}(map(x->reinterpret(UInt32,x.value),words))
end
@inline load(T::Type{<:Packed},A::CuDeviceArray,index::Tuple) =
    load(T,A,@inbounds LinearIndices(A)[index...])

# Unpack the storage format; all products and accumulation use Float32.
@inline function dotadd(a::Packed{F,N},b::Packed{F,N},value::Float32=0f0) where {F<:TensorFloat,N}
    @loopinfo unroll=true for word in 1:N
        x,y=a[word],b[word]
        value=muladd(Float32(reinterpret(F,x%UInt16)),
            Float32(reinterpret(F,y%UInt16)),value)
        value=muladd(Float32(reinterpret(F,(x>>>16)%UInt16)),
            Float32(reinterpret(F,(y>>>16)%UInt16)),value)
    end
    value
end

@generated function prepare_backward!(δQ::CuDeviceArray{Float32,1},Δ,L,dO,output,ℓ,m,::Val{D},::Val{Values},::Val{Rows}) where {D,Values,Rows}
    G=D÷Values
    quote
        thread=threadIdx().x-1i32
        lane,group=thread%$(Int32(G)),thread÷$(Int32(G))
        i,head,document=blockIdx()
        groups=blockDim().x÷$(Int32(G))
        T,H=size(output,3)%Int32,size(output,2)%Int32
        padded=16i32*cld(T,16i32)
        @tile for row in 0:$(Rows-1)
            query=((i-1i32)*$(Int32(Rows))+row)*groups+group+1i32
            if query<=padded
                offset=$(Int32(D))*(query-1i32+padded*(head-1i32+H*(document-1i32)))
                # Each instruction writes one contiguous row segment across the group.
                @tile for segment in 0:$(Values÷4-1)
                    store!(δQ,offset+4i32*lane+$(Int32(4G))*segment+1i32,
                        (@ntuple 4 d->VecElement(0f0)))
                end
            end
            value=0f0
            if query<=T
                @tile for chunk in 0:$(Values÷8-1)
                    channel=8i32*lane+$(Int32(8G))*chunk+1i32
                    index=(channel,head,query,document)
                    a=load(Packed{eltype(dO),4},dO,index)
                    b=load(Packed{eltype(output),4},output,index)
                    value=dotadd(a,b,value)
                end
            end
            # Padded/inactive groups also participate, so FULL_MASK remains valid.
            value=reduce_lanes(+,value,Val($G))
            if lane==0i32 && query<=T
                @inbounds begin
                    Δ[1,query,head,document]=value
                    L[1,query,head,document]=muladd(m[1,query,head,document],LOG₂E,log2(ℓ[1,query,head,document]))
                end
            end
        end
        nothing
    end
end


function prepare_backward!(::Type{F},δQ,Δ,L,dO::CuArray{F,4},O,ℓ,m,::Val{D},::Val{W}) where {F<:TensorFloat,D,W}
    _,H,T,B = size(dO)
    if D in (64,128)
        # Eight lanes per row, two rows per subgroup:64 queries/256-thread CTA.
        @cuda threads=256 blocks=(cld(16cld(T,16),64),H,B) prepare_backward!(
            δQ,Δ,L,dO,O,ℓ,m,Val(D),Val(D÷8),Val(2))
    else
        @cuda threads=32W blocks=(cld(16cld(T,16),W),H,B) prepare_backward!(
            nothing,δQ,Δ,L,dO,O,ℓ,m,nothing,Val(D))
    end
    dO,nothing
end

function prepare_backward!(::Type{F},δQ,Δ,L,dO::CuArray{Float32,4},O,ℓ,m,::Val{D},::Val{W}) where {F<:TensorFloat,D,W}
    _,H,T,B = size(dO)
    s = mapreduce(abs,max,vec(dO);dims=1)
    @. s = ifelse(iszero(s),1f0,exp2(clamp(floor(log2(32f0/s)),-120f0,120f0)))
    dO₁₆ = similar(dO,F)
    @cuda threads=32W blocks=(cld(16cld(T,16),W),H,B) prepare_backward!(
        dO₁₆,δQ,Δ,L,dO,O,ℓ,m,s,Val(D))
    dO₁₆,s
end

# Preserve the final rounding boundary while transposing the MMA scratch.
@inline store!(S::CuDeviceArray{F},d,q,values::NTuple{2,Float32}) where {F<:TensorFloat} = store!(S,d,q,pack(F,values...))
@inline function store!(S::CuDeviceArray{Float32},d,q,values::NTuple{2,Float32})
    @inbounds S[d,q],S[d+1,q] = values
    nothing
end

function gather_query!(dQ::CuDeviceArray{F,4},partial,s,::Val{D},::Val{H},policy::Val) where {F,D,H}
    tile,head,document = blockIdx()
    tile -= 1i32
    thread = threadIdx().x
    warp,lane = (thread-1i32)÷32i32,(thread-1i32)%32i32
    T = size(dQ,3)%Int32
    offset = Int32(16D)*(tile+cld(T,16i32)*(head-1i32+Int32(H)*(document-1i32)))
    @inbounds S = CuStaticSharedArray(F,(D+8,16))
    τ = Float32(inv(sqrt(D)))
    unscale = s === nothing ? τ : (@inbounds τ/s[1])
    for d in warp:4i32:Int32(D÷8-1)
        source = offset+128i32*d+lane+1i32
        channel = 8i32*d+2i32*(lane%4i32)+1i32
        row = lane÷4i32+1i32
        @inbounds a,b,c,e = partial[source],partial[source+32i32],partial[source+64i32],partial[source+96i32]
        @tile for pair in 0:1
            q = row+8i32*pair
            query = 16i32*tile+q
            x,y = pair==0 ? (unscale*a,unscale*b) : (unscale*c,unscale*e)
            if query<=T
                @inbounds index = LinearIndices(dQ)[channel,head,query,document]
                x = gradient_value(dQ,index,x,policy)
                y = gradient_value(dQ,index+1,y,policy)
            end
            store!(S,channel,q,(x,y))
        end
    end
    sync_threads()
    width = Int32(16÷sizeof(F))
    for index in thread:128i32:Int32(16D÷width)
        channel = width*((index-1i32)%Int32(D÷width))+1i32
        row = (index-1i32)÷Int32(D÷width)+1i32
        query = 16i32*tile+row
        if query<=T
            @inbounds output = LinearIndices(dQ)[channel,head,query,document]
            store!(dQ,output,load(Word4,S,channel+(row-1i32)*Int32(D+8)))
        end
    end
    nothing
end

function reduce_heads!(dK,dV,partialK,partialV,s,::Val{Shards},::Val{D},policy::Val) where {Shards,D}
    index = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    if index <= length(dK)
        first = index+(Int32(Shards)-1i32)*Int32(D)*((index-1i32)÷Int32(D))
        k,v = 0f0,0f0
        for shard in 0i32:Int32(Shards-1)
            @inbounds k += partialK[first+shard*Int32(D)]
            @inbounds v += partialV[first+shard*Int32(D)]
        end
        unscale = s === nothing ? 1f0 : (@inbounds inv(s[1]))
        store_gradient!(dK,index,k*unscale,policy)
        store_gradient!(dV,index,v*unscale,policy)
    end
    nothing
end


# K remains live throughout. V dies after register capture; all other buffers
# begin afterwards and reuse that complete region, not just V's first C rows.
function backward_layout(D,R,C)
    Kbytes = sizeof(UInt16)*(D+8)*R
    Qbytes = sizeof(UInt16)*(D+8)*C
    Sbytes = sizeof(UInt16)*(C+8)*R
    stats = sizeof(Float32)*C
    offsets = (K=0,V=Kbytes,Q=Kbytes,dO=Kbytes+Qbytes,dS=Kbytes+2Qbytes,
               L=Kbytes+2Qbytes+Sbytes,Δ=Kbytes+2Qbytes+Sbytes+stats)
    (;bytes=Kbytes+max(Kbytes,2Qbytes+Sbytes+2stats),offsets)
end

# Couple the tile dimensions: R = 16W, C = max(16,8W).
# These are software defaults, not GPU-family or sequence-length switches.
function backward_geometry(D,budget)
    W,C = 12,96
    while backward_layout(D,16W,C).bytes > budget && W > 1
        W -= min(4,W÷2)
        C = 16max(1,W÷2)
    end
    (;warps=W,query_tile=C)
end

# Structural backward launch policy; no timing-based tuning.
# Four compact warps is a software choice, not a GPU architectural constant.
# A compact tile is square: R=C=16W. Large tiles retain the existing R=2C rule.
function initial_geometry(D,H,Hkv,T,B,budget,sms,compact_warps)
    w = backward_geometry(D,budget).warps
    while w>1 && H*B*cld(T,16w)<sms
        w -= min(4,w÷2)
    end
    c = 16max(1,w÷2)
    if w<=compact_warps && backward_layout(D,16w,16w).bytes<=budget
        c = 16w
    end
    span = H>Hkv && Hkv*B*cld(T,16w)>=sms ? H÷Hkv : 1
    (;warps=w,query_tile=c,span)
end

prefer_compact(large_blocks,large_residency,compact_blocks,compact_residency,sms) =
    large_residency>0 && compact_residency>large_residency &&
    cld(compact_blocks,sms*compact_residency)<=cld(large_blocks,sms*large_residency)

# This barrier specializes on named-tuple/Val types. It never executes GPU
# work, copies an array, initializes scratch, or constructs a heterogeneous Dict.
Base.@noinline function compile_candidate(common::NamedTuple,::Val{W},::Val{C},
    groups::Val,span::Val,policy::Val,
) where {W,C}
    args = (Tuple(common)...,Val(W),Val(C),groups,span,policy)
    kernel = @cuda launch=false maxthreads=32W attention_gradient!(args...)
    D = size(common.Q,1)
    shmem = backward_layout(D,16W,C).bytes
    shared_memory!(kernel,shmem)
    residency = CUDA.active_blocks(kernel.fun,32W;shmem)
    (;kernel,args,shmem,residency)
end

Base.@noinline function launch_candidate!(candidate,::Val{W},H,T,B,::Val{Span}) where {W,Span}
    candidate.kernel(candidate.args...;threads=32W,blocks=(H÷Span,B,cld(T,16W)),shmem=candidate.shmem)
    nothing
end

# Keep branches separate through launch: differently specialized HostKernels
# are never merged into a resource dictionary or a large dynamic result union.
Base.@noinline function select_and_launch!(common::NamedTuple,::Val{W},::Val{C},
    groups::Val,span::Val{Span},policy::Val,::Val{Compact},sms,shared_per_sm,reserved_shared,
) where {W,C,Span,Compact}
    D,H,T,B = size(common.Q)
    large = compile_candidate(common,Val(W),Val(C),groups,span,policy)
    @assert large.residency>0 "Initial attention geometry has no resident CTA"
    large_blocks = (H÷Span)*B*cld(T,16W)
    compact_shared = backward_layout(D,16Compact,16Compact).bytes
    # Ownership is fixed before allocation. Only an already-viable grouped
    # grid may trade one large CTA for multiple compact resident CTAs.
    if Span>1 && W>Compact && 2*(compact_shared+reserved_shared)<=shared_per_sm
        small = compile_candidate(common,Val(Compact),Val(16Compact),groups,span,policy)
        compact_blocks = (H÷Span)*B*cld(T,16Compact)
        if prefer_compact(large_blocks,large.residency,compact_blocks,small.residency,sms)
            launch_candidate!(small,Val(Compact),H,T,B,span)
            return (;warps=Compact,query_tile=16Compact,head_span=Span,
                    active_blocks=small.residency,compact_selected=true)
        end
    end
    launch_candidate!(large,Val(W),H,T,B,span)
    (;warps=W,query_tile=C,head_span=Span,active_blocks=large.residency,compact_selected=false)
end


"""
    Δattention!(::TensorCoreInstruction,dQ,dK,dV,dO,Q,K,V,O,ℓ,m,window;
                accumulate=true,warps=nothing,query_tile=nothing,head_span=nothing)

One tiled Tensor Core backward. Q/K/V are Float16 or BFloat16. O/dO and dQ/dK/dV
can independently use that format or Float32. Statistics, atomics, and partial KV sums remain
Float32. Float32 upstream retains GPU power-of-two scaling; 16-bit upstream is
consumed directly, with its explicit representability and underflow limits.

The default adds to caller gradients. Existing values are converted to Float32
and added before the final storage conversion. Explicit overwrite never reads
caller gradients, but always clears the internal atomic scratch. This is a
mixed-precision analytic backward, not a derivative of quantization.

A CTA owns head_span consecutive query heads sharing a KV head. The span must
divide the group size; split groups publish Float32 partials for an ordered
reduction. Defaults use shared capacity, compiled residency, and finite CTA
waves. Any explicit Val geometry/span keyword disables compact substitution;
missing values retain their structural defaults. No timed autotuning occurs.

Supports compatible nonempty, nonaliasing arrays on SM80+, ragged sequences,
causal (-1,0), and nonnegative local extents. Scratch is linear in sequence
length. dQ is numerically reproducible, not bitwise deterministic.
"""
function Δattention!(::TensorCoreInstruction,
    dQ::CuArray{F,4},dK::CuArray{F,4},dV::CuArray{F,4},dO::CuArray{G,4},
    Q::CuArray{E,4},K::CuArray{E,4},V::CuArray{E,4},
    O::CuArray{G,4},ℓ::CuArray{Float32,4},m::CuArray{Float32,4},window::Tuple{Int,Int};
    accumulate::Bool=true,warps=nothing,query_tile=nothing,head_span=nothing,
) where {E<:TensorFloat,F<:Union{E,Float32},G<:Union{E,Float32}}
    D,H,T,B = size(Q)
    dev = device(Q)
    budget = attribute(dev,CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    sms = attribute(dev,CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    shared_per_sm = attribute(dev,CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
    reserved_shared = attribute(dev,CUDA.DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK)
    geometry = initial_geometry(D,H,size(K,2),T,B,budget,sms,4)
    W = something(warps,Val(geometry.warps))
    C = something(query_tile,Val(geometry.query_tile))
    Span = something(head_span,Val(geometry.span))
    automatic = isnothing(warps) && isnothing(query_tile) && isnothing(head_span)
    launch_gradient!(dQ,dK,dV,dO,Q,K,V,O,ℓ,m,window,W,C,Span,Val(accumulate),
                     Val(automatic),sms,shared_per_sm,reserved_shared)
end

function launch_gradient!(dQ,dK,dV,dO,Q,K,V,O,ℓ,m,window,
    ::Val{W},::Val{C},::Val{Span},policy::Val,::Val{Automatic},
    sms,shared_per_sm,reserved_shared,
) where {W,C,Span,Automatic}
    D,H,T,B = size(Q)
    groups = H÷size(K,2)
    @assert 0 < D <= 128 && D%16==0 && C>0 && C%16==0 && 0<W<=32 &&
        W%min(W,C÷16)==0 && Span>0 && H%size(K,2)==0 && groups%Span==0
    Δ,L = similar(ℓ),similar(ℓ)
    δQ = similar(dO,Float32,16D*cld(T,16)*H*B)
    δK,δV = Span==groups ? (dK,dV) :
        ntuple(_ -> similar(dO,Float32,D,H÷Span,T,B),2)
    # Preparation and padded scratch clearing happen once, not per candidate.
    dO₁₆,s = prepare_backward!(eltype(Q),δQ,Δ,L,dO,O,ℓ,m,Val(D),Val(W))
    common = (;dQ=δQ,dK=δK,dV=δV,Q,K,V,dO=dO₁₆,L,Δ,s,window=Val(window),D=Val(D))
    if Automatic
        select_and_launch!(common,Val(W),Val(C),Val(groups),Val(Span),policy,
                           Val(4),sms,shared_per_sm,reserved_shared)
    else
        candidate = compile_candidate(common,Val(W),Val(C),Val(groups),Val(Span),policy)
        launch_candidate!(candidate,Val(W),H,T,B,Val(Span))
    end
    @cuda threads=128 blocks=(cld(T,16),H,B) gather_query!(dQ,δQ,s,Val(D),Val(H),policy)
    if Span!=groups
        @cuda threads=256 blocks=cld(length(dK),256) reduce_heads!(
            dK,dV,δK,δV,s,Val(groups÷Span),Val(D),policy)
        foreach(CUDA.unsafe_free!,(δK,δV))
    end
    # Release scratch on its live stream, before XLA can destroy that context.
    foreach(CUDA.unsafe_free!,(Δ,L,δQ))
    if dO₁₆ !== dO
        CUDA.unsafe_free!(dO₁₆)
        CUDA.unsafe_free!(s)
    end
    nothing
end

@generated function attention_gradient!(dQ,dK,dV,Q,K,V,dO,L,Δ,s,
    ::Val{Window},::Val{D},::Val{W},::Val{C},::Val{Groups},::Val{Span},::Val{Accumulate},
) where {Window,D,W,C,Groups,Span,Accumulate}
    R = 16W
    query_warps = min(W,C÷16)
    channel_warps = W÷query_warps
    (;offsets) = backward_layout(D,R,C)
    quote
        shard,document,j = blockIdx()
        kv_head = (shard-1i32)÷$(Int32(Groups÷Span))+1i32
        thread = threadIdx().x
        warp,lane = (thread-1i32)÷32i32,(thread-1i32)%32i32
        lane_row,lane_pair = lane÷4i32,lane%4i32
        T = size(Q,3) % Int32
        blockⱼ = (j-1i32)*$(Int32(R))
        key = blockⱼ+16i32*warp+lane_row+1i32
        left,right = $(Int32.(Window))
        τ = $(Float32(inv(sqrt(D))))
        τ₂ = $(Float32(log2(exp(1.0))/sqrt(D)))
        unscale = s === nothing ? 1f0 : (@inbounds inv(s[1]))

        # The same multiply, transposed: rows are keys, columns are queries.
        @inbounds begin
            Kⱼ = CuDynamicSharedArray(eltype(K),($D+8,$R),$(offsets.K))
            Vⱼ = CuDynamicSharedArray(eltype(V),($D+8,$R),$(offsets.V))
            Qᵢ = CuDynamicSharedArray(eltype(Q),($D+8,$C),$(offsets.Q))
            dOᵢ = CuDynamicSharedArray(eltype(dO),($D+8,$C),$(offsets.dO))
            dSᵢⱼ = CuDynamicSharedArray(eltype(Q),($C+8,$R),$(offsets.dS))
            Lᵢ = CuDynamicSharedArray(Float32,$C,$(offsets.L))
            Δᵢ = CuDynamicSharedArray(Float32,$C,$(offsets.Δ))
        end
        copy_rows!(copy!,(Kⱼ,Vⱼ),(K,V),kv_head,blockⱼ,document,Val($D),Val($R),Val($W))
        sync_threads()
        Vᵣ = Tile{(4,$(D÷16))}(UInt32(0)) # V retained in registers, not a gradient.
        @tile for d in 1:$(D÷16)
            Vᵣ[:,d] = query_matrix(Vⱼ,16warp,16(d-1),lane)
        end
        sync_threads()
        # V is no longer live; Q/dO/dS/statistics can now overwrite its storage.
        𝔾ₖ = Tile{(4,$(D÷8))}(0f0)
        𝔾ᵥ = Tile{(4,$(D÷8))}(0f0)
        first_query = max(0i32,blockⱼ-($Window == (-1,0) ? 0i32 : right))÷$(Int32(C))*$(Int32(C))
        last_query = $Window == (-1,0) ? T : min(T,blockⱼ+$(Int32(R))+left)

        # One CTA owns Span consecutive query heads from the same KV group.
        # Keep both gradients live across heads; prime each head's Q/dO tile.
        @loopinfo unroll=false for head in ((shard-1i32)*$(Int32(Span))+1i32):(shard*$(Int32(Span)))
            # Prime the first query tile after V's register capture has completed.
            copy_rows!(copy_async!,(Qᵢ,dOᵢ),(Q,dO),head,first_query,document,Val($D),Val($C),Val($W))
            commit_copies!()
            for row in thread:$(32W):$C
                query = first_query+row
                @inbounds Lᵢ[row] = query <= T ? L[1,query,head,document] : 0f0
                @inbounds Δᵢ[row] = query <= T ? Δ[1,query,head,document] : 0f0
            end
            wait_copies!()

            for i in first_query÷$(Int32(C)):cld(last_query,$(Int32(C)))-1i32
                blockᵢ = i*$(Int32(C))
                interior = blockⱼ+$(Int32(R)) <= T && blockᵢ+$(Int32(C)) <= T &&
                    ($Window == (-1,0) ? blockⱼ+$(Int32(R)) <= blockᵢ+1i32 :
                     blockᵢ+$(Int32(C))-left <= blockⱼ+1i32 &&
                     blockⱼ+$(Int32(R)) <= blockᵢ+1i32+right)
                @loopinfo unroll=false for column in 0i32:16i32:$(Int32(C-16))
                    if $Window == (-1,0) && !interior &&
                        (blockⱼ+16i32*warp+1i32 > min(T,blockᵢ+column+16i32) || blockᵢ+column >= T)
                        @tile for n in 1:2
                            row=16warp+lane_row+1
                            query=column+8(n-1)+2lane_pair+1
                            store!(dSᵢⱼ,query,row,UInt32(0))
                            store!(dSᵢⱼ,query,row+8,UInt32(0))
                        end
                    else
                        P = Tile{(4,2)}(0f0)
                        dS = Tile{(4,2)}(0f0)
                        @tile for d in 1:$(D÷16)
                            A = query_matrix(Kⱼ,16warp,16(d-1),lane)
                            for n in 1:2
                                B = key_matrix(Qᵢ,column+8(n-1),16(d-1),lane)
                                P[:,n] = muladd(A,B,P[:,n])
                                B = key_matrix(dOᵢ,column+8(n-1),16(d-1),lane)
                                dS[:,n] = muladd(Packed{eltype(Q)}(Vᵣ[:,d]),B,dS[:,n])
                            end
                        end
                        if interior
                            @tile for n in 1:2, r in 1:4
                                col = column+8(n-1)+2lane_pair+mod(r-1,2)+1
                                query = blockᵢ+col
                                @inbounds p = exp₂(muladd(P[r,n],τ₂,-Lᵢ[col]))
                                @inbounds dS[r,n] = p*(dS[r,n]-Δᵢ[col])
                                P[r,n] = p
                            end
                        else
                            @tile for n in 1:2, r in 1:4
                                col = column+8(n-1)+2lane_pair+mod(r-1,2)+1
                                query = blockᵢ+col
                                row = key+8*((r-1)÷2)
                                valid = row <= T && query <= T && ($Window == (-1,0) ? row <= query : query-left <= row <= query+right)
                                if valid
                                    @inbounds p = exp₂(muladd(P[r,n],τ₂,-Lᵢ[col]))
                                    multiple = $Window == (-1,0) ? query > 1 : max(1,query-left) < min(T,query+right)
                                    @inbounds dS[r,n] = multiple ? p*(dS[r,n]-Δᵢ[col]) : 0f0
                                    P[r,n] = p
                                else
                                    P[r,n] = 0f0
                                    dS[r,n] = 0f0
                                end
                            end
                        end
                        p = probability_matrix(eltype(Q),P,1)
                        ds = probability_matrix(eltype(Q),dS,1)
                        # The same rounded words feed dK and the transposed dQ tile.
                        @tile for n in 1:2
                            row = 16warp+lane_row+1
                            query = column+8(n-1)+2lane_pair+1
                            store!(dSᵢⱼ,query,row,ds[2n-1])
                            store!(dSᵢⱼ,query,row+8,ds[2n])
                        end
                        @tile for d in 1:$(D÷8)
                            B = value_matrix(Qᵢ,column,8(d-1),lane)
                            𝔾ₖ[:,d] = muladd(ds,B,𝔾ₖ[:,d])
                            B = value_matrix(dOᵢ,column,8(d-1),lane)
                            𝔾ᵥ[:,d] = muladd(p,B,𝔾ᵥ[:,d])
                        end
                    end
                end
                sync_threads()

                # All current Q/dO reads are finished. Their next tile can arrive
                # while dQ uses only resident K and the completed dS tile.
                if blockᵢ+$(Int32(C)) < last_query
                    copy_rows!(copy_async!,(Qᵢ,dOᵢ),(Q,dO),head,blockᵢ+$(Int32(C)),document,Val($D),Val($C),Val($W))
                    commit_copies!()
                    for row in thread:$(32W):$C
                        query = blockᵢ+$(Int32(C))+row
                        @inbounds Lᵢ[row] = query <= T ? L[1,query,head,document] : 0f0
                        @inbounds Δᵢ[row] = query <= T ? Δ[1,query,head,document] : 0f0
                    end
                end

                # K stays resident in shared storage throughout the tiled query loop.
                # When C has fewer query blocks than warps, use the remaining
                # warp dimension for disjoint output-channel chunks.
                # Retained for codegen: removing this bound increases SM89 register use.
                if warp < $(Int32(query_warps*channel_warps))
                    for q₀ in (16i32*(warp%$(Int32(query_warps)))):$(Int32(16query_warps)):$(Int32(C-1))
                        @loopinfo unroll=false for d₀ in (1i32+warp÷$(Int32(query_warps))):$(Int32(2channel_warps)):$(Int32(D÷8))
                            # Two independent outputs reuse each dS matrix load.
                            𝔾q = Tile{(4,2)}(0f0)
                            second = $(D÷8 % (2channel_warps) == 0) || d₀+$(Int32(channel_warps)) <= $(Int32(D÷8))
                            ds = transposed_query_matrix(dSᵢⱼ,q₀,0i32,lane)
                            B₁ = value_matrix(Kⱼ,0i32,8i32*(d₀-1i32),lane)
                            B₂ = second ? value_matrix(Kⱼ,0i32,8i32*(d₀+$(Int32(channel_warps))-1i32),lane) : Packed{eltype(K)}((0x00000000,0x00000000))
                            # Load the next tiles while the current multiply runs.
                            # Peel the last multiply to avoid guarded lookahead loads.
                            for k₀ in 0i32:16i32:$(Int32(R-32))
                                dsⁿᵉʷ = transposed_query_matrix(dSᵢⱼ,q₀,k₀+16i32,lane)
                                @tile 𝔾q[:,1] = muladd(ds,B₁,𝔾q[:,1])
                                B₁ⁿᵉʷ = value_matrix(Kⱼ,k₀+16i32,8i32*(d₀-1i32),lane)
                                if second
                                    @tile 𝔾q[:,2] = muladd(ds,B₂,𝔾q[:,2])
                                end
                                B₂ⁿᵉʷ = second ? value_matrix(Kⱼ,k₀+16i32,8i32*(d₀+$(Int32(channel_warps))-1i32),lane) : B₂
                                ds,B₁,B₂ = dsⁿᵉʷ,B₁ⁿᵉʷ,B₂ⁿᵉʷ
                            end
                            @tile 𝔾q[:,1] = muladd(ds,B₁,𝔾q[:,1])
                            if second
                                @tile 𝔾q[:,2] = muladd(ds,B₂,𝔾q[:,2])
                            end
                            @tile for n in 1:2, r in 1:4
                                d = d₀+(n-1)*$(Int32(channel_warps))
                                query = blockᵢ+q₀+lane_row+8*((r-1)÷2)+1
                                if query <= T && ($(D÷8 % (2channel_warps) == 0) || d <= $(Int32(D÷8)))
                                    index = tile_index(lane,r,d,(blockᵢ+q₀)÷16i32,
                                        head,document,cld(T,16i32),size(Q,2)%Int32,Val($D))
                                    CUDA.atomic_add!(pointer(dQ,index),𝔾q[r,n])
                                end
                            end
                        end
                    end
                end
                # Finish next-tile copies and all dS readers before reusing dS.
                wait_copies!()
            end
        end # owned query heads
        if $Span == $Groups && eltype(dK) <: TensorFloat
            # Every input/dS reader retired at the final wait. Stage final
            # 16-bit values only after optionally adding the old value in FP32.
            @inbounds begin
                gradientK = CuDynamicSharedArray(eltype(dK),($D,$R),0)
                gradientV = CuDynamicSharedArray(eltype(dV),($D,$R),$(2D*R))
            end
            @tile for d in 1:$(D÷8), pair in 1:2
                channel = 8(d-1)+2lane_pair+1
                local_row = 16warp+lane_row+8(pair-1)+1
                row = blockⱼ+local_row
                k₁,k₂ = τ*𝔾ₖ[2pair-1,d]*unscale,τ*𝔾ₖ[2pair,d]*unscale
                v₁,v₂ = 𝔾ᵥ[2pair-1,d]*unscale,𝔾ᵥ[2pair,d]*unscale
                if row <= T
                    @inbounds index = LinearIndices(dK)[channel,kv_head,row,document]
                    k₁ = gradient_value(dK,index,k₁,Val($Accumulate))
                    k₂ = gradient_value(dK,index+1,k₂,Val($Accumulate))
                    v₁ = gradient_value(dV,index,v₁,Val($Accumulate))
                    v₂ = gradient_value(dV,index+1,v₂,Val($Accumulate))
                end
                store!(gradientK,channel,local_row,pack(eltype(dK),k₁,k₂))
                store!(gradientV,channel,local_row,pack(eltype(dV),v₁,v₂))
            end
            sync_threads()
            @tile for vector in 0:$(D÷16-1)
                index = thread+$(Int32(32W))*vector
                channel = 8i32*((index-1i32)%$(Int32(D÷8)))+1i32
                row = (index-1i32)÷$(Int32(D÷8))+1i32
                if blockⱼ+row <= T
                    @inbounds output = LinearIndices(dK)[channel,kv_head,blockⱼ+row,document]
                    store!(dK,output,load(Word4,gradientK,8index-7i32))
                    store!(dV,output,load(Word4,gradientV,8index-7i32))
                end
            end
        else
            @tile for d in 1:$(D÷8), r in 1:4
                channel = 8(d-1)+2lane_pair+mod(r-1,2)+1
                row = key+8*((r-1)÷2)
                if row <= T
                    @inbounds index = LinearIndices(dK)[channel,shard,row,document]
                    if $Span == $Groups
                        store_gradient!(dK,index,τ*𝔾ₖ[r,d]*unscale,Val($Accumulate))
                        store_gradient!(dV,index,𝔾ᵥ[r,d]*unscale,Val($Accumulate))
                    else
                        # Shards are disjoint FP32 partials. Unscale and add to
                        # the caller only after their final ordered reduction.
                        @inbounds dK[index] = τ*𝔾ₖ[r,d]
                        @inbounds dV[index] = 𝔾ᵥ[r,d]
                    end
                end
            end
        end
        nothing
    end
end

end # module FemtoChatCUDAExt
