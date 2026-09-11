module FemtoChatCUDAExt

using CUDA
using CUDA: i32
using LinearAlgebra: mul!
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
const TENSOR_CORE = (warps=4,key_tile=64)

instruction(::Type, dev::CuDevice, ::Val) = SIMTInstruction()
function instruction(::Type{F}, dev::CuDevice, ::Val{D}) where {F<:Union{Float16,Float32},D}
    D > 0 && D % 16 == 0 || return SIMTInstruction()
    Bᵣ,Bᶜ = 16TENSOR_CORE.warps,TENSOR_CORE.key_tile
    shared = sizeof(Float16)*(D+8)*(max(Bᵣ,Bᶜ)+Bᶜ)
    shared <= CUDA.attribute(dev,CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK) ||
        return SIMTInstruction()
    target = compiler_targets(CUDACompiler.compiler_config(dev)).llvm.compute
    CUDA.capability(dev) >= v"8.0" && target >= v"8.0" ?
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
const SIMT = (fragment=(8,4), lanes=(4,8), stage=16, vector=4, swizzle=(bits=3, base=2))

function attention_layout(D, warps, spec=SIMT)
    (; fragment, lanes, stage, vector, swizzle) = spec
    warp_tile = fragment .* lanes
    Bᵣ, Bᶜ = warp_tile .* warps
    threads = 32prod(warps)
    Dᵥ = Bᶜ * cld(D, Bᶜ)

    # Q/K use two stages. P̃ reuses those stages; row statistics reuse V.
    stage_size = stage * (Bᵣ + Bᶜ)
    V_offset = max(2stage_size, Bᵣ * Bᶜ)
    shared = V_offset + max(2stage * Dᵥ, Bᵣ * warps[2])
    return (; D, fragment, lanes, warp_tile, warps, tile=(Bᵣ,Bᶜ),
            stage, vector, swizzle, threads, Dᵥ, stage_size, V_offset, shared)
end


# ── Host entry points ───────────────────────────────────────────────────────

"""
    flash_attention₁!(𝕆, ℓ, m, Q, K, V, window, ::Val{Warps}=Val((2,2)); simt=Val(SIMT))

Configure the Float32 kernel's query/key warp arrangement. The default fragment
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
    Fragment{Shape}(value)
See: https://docs.nvidia.com/cuda/parallel-thread-execution/?utm_source=chatgpt.com#warp-level-matrix-fragment
"""
struct Fragment{Shape,T,N}
    values::NTuple{N,T}
end

@inline Fragment{Shape}(x::T) where {Shape,T<:Number} =
    Fragment{Shape}(ntuple(Returns(x), Val(prod(Shape))))
@inline Fragment{Shape}(values::NTuple{N,T}) where {Shape,N,T} =
    Fragment{Shape,T,N}(values)

Base.size(::Fragment{Shape}) where Shape = Shape
Base.length(::Fragment{Shape,T,N}) where {Shape,T,N} = N
Base.eltype(::Type{<:Fragment{Shape,T}}) where {Shape,T} = T
Base.Tuple(S::Fragment) = S.values
Base.iterate(S::Fragment, state...) = iterate(S.values, state...)

# A slice is an ordinary tuple, so reductions and broadcasts use ordinary Julia.
# It is a snapshot, not a view. Use one integer or ':' per dimension.
function coordinates(Shape, I)
    length(Shape) == length(I) || error("use one index per fragment dimension")
    ranges = ntuple(d -> I[d] <: Colon ? (1:Shape[d]) : (1:1), length(Shape))
    positions = CartesianIndices(ranges)
    coordinates = [ntuple(d -> I[d] <: Colon ? p[d] : :(I[$d]), length(Shape)) for p in positions]
    vec(coordinates)
end

function linear_index(Shape, coordinate)
    terms = [:(($(coordinate[d]) - 1) * $(prod(Shape[1:d-1]))) for d in eachindex(Shape)]
    :(1 + $(Expr(:call, :+, terms...)))
end

@inline @generated function Base.getindex(S::Fragment{Shape}, I::Vararg{Union{Integer,Colon},N}) where {Shape,N}
    values = [:(S.values[$(linear_index(Shape,c))]) for c in coordinates(Shape,I)]
    result = any(t -> t <: Colon, I) ? Expr(:tuple,values...) : only(values)
    :(@inbounds $result)
end

# Rebuild a tuple with the requested scalar/slice replaced. Static indices allow
# the compiler to discard all the unchanged tuple copies.
@inline @generated function replaced(S::Fragment{Shape,T}, x, I::Vararg{Union{Integer,Colon},N}) where {Shape,T,N}
    axes = findall(t -> t <: Colon,I)
    sliced = LinearIndices(Tuple(Shape[d] for d in axes))
    values = map(enumerate(CartesianIndices(Shape))) do (i,c)
        conditions = [:($(c[d]) == I[$d]) for d in eachindex(Shape) if d ∉ axes]
        selected = foldl((a,b)->:($a && $b),conditions;init=true)
        value = isempty(axes) ? :x : :(x[$(sliced[Tuple(c[d] for d in axes)...])])
        :(ifelse($selected, convert(T,$value), S.values[$i]))
    end
    :(@inbounds $(Expr(:tuple,vec(values)...)))
end

@inline Base.setindex(S::Fragment{Shape}, x, I...) where Shape =
    Fragment{Shape}(replaced(S,x,I...))

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
        A isa Symbol || error("@fragment assignment requires a local variable")
        object, indices, value = gensym.((:object,:indices,:value))
        if x.head != :(=)
            op = Symbol(chop(string(x.head)))
            rhs = :($op(getindex($object,$indices...),$rhs))
        end
        return quote
            local $object = $A
            local $indices = ($(I...),)
            local $value = $rhs
            if $object isa $(GlobalRef(@__MODULE__,:Fragment))
                $A = Base.setindex($object,$value,$indices...)
            else
                setindex!($object,$value,$indices...)
            end
            $value
        end
    end
    Expr(x.head,map(assignments,x.args)...)
end

"""
    @fragment begin ... end

Unroll literal loops and rebind immutable fragments on indexed assignment.
Fragments must already be constructed explicitly. Other arrays retain mutation.
No variable names, fragment shapes, reductions, or multiply names are recognized.
"""
macro fragment(body)
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
@inline function load(::Type{Float32ᵛ{W}}, A::CuDeviceArray{Float32,N,CUDA.AS.Global}, index) where {W,N}
    p = reinterpret(Core.LLVMPtr{Float32ᵛ{W},CUDA.AS.Global}, pointer(A, index))
    return CUDA.unsafe_cached_load(p, 1, Val(sizeof(Float32)*W))
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

@inline function store!(A::CuDeviceArray{Float32,N,CUDA.AS.Shared}, index, value::Float32ᵛ{W}) where {W,N}
    p = reinterpret(Core.LLVMPtr{Float32ᵛ{W},CUDA.AS.Shared}, pointer(A, index))
    unsafe_store!(p, value, 1, Val(sizeof(Float32)*W))
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
    R, C = Spec.fragment
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


# Preserve the fragment's shape around the tuple-level register multiply.
@inline function Base.muladd(a::A, b::B, x::Fragment{Shape}, rest::Vararg{Any,N}) where {A,B,Shape,N}
    Fragment{Shape}(muladd(a,b,x.values,rest...))
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
    R, C = Layout.fragment
    Nq, Nk = cld(stage*Bᵣ, vector*threads), cld(stage*Bᶜ, vector*threads)
    No = cld(D, Bᶜ)
    Nv = cld(stage*Dᵥ, vector*threads)
    return :(@fragment begin
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
        row₀ = Int32(layout.warp_tile[1]) * warpᵣ + Int32(layout.fragment[1]) * laneᵣ
        col₀ = Int32(layout.warp_tile[2]) * warpᶜ + Int32(layout.fragment[2]) * laneᶜ
        query₀ = blockᵢ + row₀
        log₂e = 1.4426950408889634f0
        scale = log₂e / sqrt(Float32(D))

        # Output channels may require several register fragments per thread.
        padded_D = Int32(cld(D,stage)*stage)
        Dᵥ = layout.Dᵥ
        qk_stage_bytes = sizeof(Float32) * layout.stage_size
        v_stage_bytes = sizeof(Float32) * stage * Dᵥ
        V_offset_bytes = sizeof(Float32) * layout.V_offset
        P̃ᵢⱼ = Swizzled(CuDynamicSharedArray(Float32, (Bᵣ, Bᶜ), 0), Val(layout.swizzle))
        stats = CuDynamicSharedArray(Float32, (Bᵣ, Wᶜ), V_offset_bytes)
        𝕆ᵢ = Fragment{($R,$C,$No)}(0f0)
        mᵢ = Fragment{($R,)}(-floatmax(Float32))
        ℓᵢ = Fragment{($R,)}(0f0)

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
            S = Fragment{($R,$C)}(0f0)
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
    R, C = L.fragment
    Bᶜ, Bᵣ = L.tile
    Nk, Nq = cld(L.D,Bᵣ), cld(L.D,Bᶜ)
    return :(@fragment begin
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
        dKⱼ = Fragment{($R,$C,$Nk)}(0f0)
        dVⱼ = Fragment{($R,$C,$Nk)}(0f0)
        first_query = causal ? fld(blockⱼ,Bᵣ)*Bᵣ : max(0i32,fld(blockⱼ-right,Bᵣ)*Bᵣ)
        last_query = causal ? T-1i32 : min(T-1i32,blockⱼ+Bᶜ-1+left)

        for head in (kv_head-1)*heads_per_kv+1:kv_head*heads_per_kv
            for blockᵢ in Int32(first_query):Int32(Bᵣ):Int32(last_query)
                S = Fragment{($R,$C)}(0f0)
                dP = Fragment{($R,$C)}(0f0)
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
                dQᵢ = Fragment{($R,$C,$Nq)}(0f0)
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

"""
    attention!(::TensorCoreInstruction, O, ℓ, m, Q, K, V, window;
               warps=Val(4), key_tile=Val(64))

Preallocated Tensor Core forward pass. O may be Float16 or Float32; ℓ and m must
be Float32 with shape (1,T,H,B). Each warp owns 16 queries, so the query tile is
16 × warps. The kernel overwrites O, ℓ, and m; no clearing is needed between calls.
Inputs must be nonempty, with Q's head count divisible by the KV head count.
Arrays must have compatible shapes and reside on the active device. The window
is causal (-1,0), or a pair of nonnegative left/right extents. This implementation
uses an FA2-style query-tile schedule; that scheduling choice is separate from
the Tensor Core instruction set. Requires an SM80+ device and compiler target.
"""
function attention!(::TensorCoreInstruction,
    O::CuArray{F,4}, ℓ::CuArray{Float32,4}, m::CuArray{Float32,4},
    Q::CuArray{Float16,4}, K::CuArray{Float16,4}, V::CuArray{Float16,4}, window;
    warps::Val{W}=Val(TENSOR_CORE.warps), key_tile::Val{Bᶜ}=Val(TENSOR_CORE.key_tile),
) where {F<:Union{Float16,Float32},W,Bᶜ}
    D,H,T,B = size(Q)
    @assert D > 0 && D % 16 == 0 && Bᶜ > 0 && Bᶜ % 16 == 0
    heads_per_kv = Val(H÷size(K,2))
    @cuda threads=32W blocks=(cld(T,16W),H,B) attention_kernel!(
        TensorCoreInstruction(),O,ℓ,m,Q,K,V,Val(window),Val(D),warps,key_tile,heads_per_kv)
    nothing
end

# Eight half values, moved as one aligned 128-bit vector without conversion.
const Half8 = NTuple{4,VecElement{Int32}}
@inline function copy8!(destination,source,d,q,head,token,document)
    @inbounds begin
        index = LinearIndices(source)[d,head,token,document]
        input = reinterpret(Core.LLVMPtr{Half8,CUDA.AS.Global},pointer(source,index))
        value = token <= size(source,3) ? CUDA.unsafe_cached_load(input,1,Val(16)) :
            ntuple(_ -> VecElement(Int32(0)),Val(4))
        output = reinterpret(Core.LLVMPtr{Half8,CUDA.AS.Shared},pointer(destination,d+(q-1)*size(destination,1)))
        unsafe_store!(output,value,1,Val(16))
    end
    nothing
end

@inline pack(x::Float16,y::Float16) = UInt32(reinterpret(UInt16,x)) | (UInt32(reinterpret(UInt16,y)) << 16)
@inline pack(x::Float32,y::Float32) = pack(Float16(x),Float16(y))

# ldmatrix consumes a 32-bit byte address in the shared-memory address space.
@inline function shared_address(A,row,column)
    @inbounds address = pointer(A,row+(column-1)*size(A,1))
    reinterpret(UInt,address) % UInt32
end

@inline @generated function load_matrix(address::UInt32, ::Val{N}, ::Val{Transpose}) where {N,Transpose}
    LLVM.Context() do _
        result_type = convert(LLVM.LLVMType,NTuple{N,UInt32})
        pointer_type = LLVM.PointerType(LLVM.Int8Type(),3)
        f,_ = create_function(result_type,[LLVM.Int32Type()])
        signature = LLVM.FunctionType(LLVM.StructType(fill(LLVM.Int32Type(),N)),[pointer_type])
        name = "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x$N$(Transpose ? ".trans" : "").b16.p3"
        instruction = LLVM.Function(LLVM.parent(f),name,signature)
        push!(LLVM.function_attributes(instruction),LLVM.EnumAttribute("convergent"))
        LLVM.IRBuilder() do builder
            LLVM.position!(builder,LLVM.BasicBlock(f,"entry"))
            pointer = LLVM.inttoptr!(builder,only(LLVM.parameters(f)),pointer_type)
            values = LLVM.call!(builder,signature,instruction,[pointer])
            push!(LLVM.function_attributes(values),LLVM.EnumAttribute("convergent"))
            # Return Julia's tuple representation directly: a C-ABI struct
            # return introduces out-of-line calls in the full attention kernel.
            result = LLVM.UndefValue(result_type)
            for i in 0:N-1
                result = LLVM.insert_value!(builder,result,LLVM.extract_value!(builder,values,i),i)
            end
            LLVM.ret!(builder,result)
        end
        call_function(f,NTuple{N,UInt32},Tuple{UInt32},:address)
    end
end

@inline query_matrix(Q,q,d,lane) = load_matrix(
    shared_address(Q,d+8*(lane÷16)+1,q+lane%16+1),Val(4),Val(false))
@inline key_matrix(K,k,d,lane) = load_matrix(
    shared_address(K,d+8*((lane÷8)%2)+1,k+lane%8+1),Val(2),Val(false))
@inline value_matrix(V,k,d,lane) = load_matrix(
    shared_address(V,d+1,k+lane%16+1),Val(2),Val(true))

# NVIDIA m16n8k16: packed half inputs and four FP32 accumulators per lane.
# LLVM18 needs the explicit convergent attribute before NVPTX lowering.
# The owned Fragment argument makes this Base extension local to our notation.
@inline @generated function Base.muladd(A::NTuple{4,UInt32}, B::NTuple{2,UInt32}, C::Fragment{(4,),Float32,4})
    multiply = LLVM.Context() do _
        result_type = convert(LLVM.LLVMType,NTuple{4,Float32})
        argument_types = [convert(LLVM.LLVMType,T) for T in (A,B,NTuple{4,Float32})]
        f,_ = create_function(result_type,argument_types)
        MMA = LLVM.FunctionType(LLVM.StructType(fill(LLVM.FloatType(),4)),
            [fill(LLVM.VectorType(LLVM.HalfType(),2),6);fill(LLVM.FloatType(),4)])
        instruction = LLVM.Function(LLVM.parent(f),"llvm.nvvm.mma.m16n8k16.row.col.f32.f32",MMA)
        push!(LLVM.function_attributes(instruction),LLVM.EnumAttribute("convergent"))
        LLVM.IRBuilder() do builder
            LLVM.position!(builder,LLVM.BasicBlock(f,"entry"))
            a,b,c = LLVM.parameters(f)
            packed = [LLVM.extract_value!(builder,x,i) for x in (a,b) for i in 0:(x === a ? 3 : 1)]
            inputs = LLVM.Value[LLVM.bitcast!(builder,x,LLVM.VectorType(LLVM.HalfType(),2)) for x in packed]
            append!(inputs,[LLVM.extract_value!(builder,c,i) for i in 0:3])
            product = LLVM.call!(builder,MMA,instruction,inputs)
            push!(LLVM.function_attributes(product),LLVM.EnumAttribute("convergent"))
            result = LLVM.UndefValue(result_type)
            for i in 0:3
                result = LLVM.insert_value!(builder,result,LLVM.extract_value!(builder,product,i),i)
            end
            LLVM.ret!(builder,result)
        end
        call_function(f,NTuple{4,Float32},Tuple{A,B,NTuple{4,Float32}},:A,:B,:(Tuple(C)))
    end
    :(Fragment{(4,)}($multiply))
end

# Approximate only the exponential, not the surrounding masked-tail Inf checks.
@inline exp₂(x::Float32) = ccall("llvm.nvvm.ex2.approx.ftz.f",llvmcall,Float32,(Float32,),x)

@generated function attention_kernel!(::TensorCoreInstruction,O,ℓ,m,Q,K,V,
    ::Val{Window}, ::Val{D}, ::Val{W}, ::Val{Bᶜ}, ::Val{Groups},
) where {Window,D,W,Bᶜ,Groups}
    Bᵣ = 16W
    quote
        i,head,document = blockIdx()
        thread = threadIdx().x
        warp,lane = (thread-1i32) ÷ 32i32, (thread-1i32) % 32i32
        group,part = lane ÷ 4i32, lane % 4i32
        T = size(Q,3) % Int32
        kv_head = (head-1i32) ÷ $(Int32(Groups)) + 1i32
        blockᵢ = (i-1i32)*$(Int32(Bᵣ))
        q = warp*16i32 + group + 1i32
        query = blockᵢ+q
        left,right = $Window
        τ = $(Float32(log2(exp(1.0))/sqrt(D)))

        # Padding rotates shared banks. Q's storage becomes K after loading A.
        Qᵢ = CuStaticSharedArray(Float16,($(D+8),$(max(Bᵣ,Bᶜ))))
        Kⱼ = Qᵢ
        Vⱼ = CuStaticSharedArray(Float16,($(D+8),$Bᶜ))
        for index in thread:$(32W):$(D÷8*Bᵣ)
            d,row = 8i32*((index-1i32) % $(Int32(D÷8))) + 1i32, (index-1i32) ÷ $(Int32(D÷8)) + 1i32
            copy8!(Qᵢ,Q,d,row,head,blockᵢ+row,document)
        end
        sync_threads()

        A = Fragment{(4,$(D÷16))}(UInt32(0))
        @fragment for d in 1:$(D÷16)
            A[:,d] = query_matrix(Qᵢ,warp*16,16(d-1),lane)
        end
        # No warp may overwrite Q until every warp has captured its queries.
        sync_threads()
        𝕆 = Fragment{(4,$(D÷8))}(0f0)
        m₁,m₂ = -Inf32,-Inf32
        ℓ₁,ℓ₂ = 0f0,0f0
        first_key = $Window == (-1,0) ? 0i32 : max(0i32,blockᵢ-left) ÷ $(Int32(Bᶜ)) * $(Int32(Bᶜ))
        last_key = min(T,blockᵢ+$(Int32(Bᵣ))+right)

        for blockⱼ in first_key:$(Int32(Bᶜ)):(last_key-1i32)
            for index in thread:$(32W):$(D÷8*Bᶜ)
                d,k = 8i32*((index-1i32) % $(Int32(D÷8))) + 1i32, (index-1i32) ÷ $(Int32(D÷8)) + 1i32
                copy8!(Kⱼ,K,d,k,kv_head,blockⱼ+k,document)
                copy8!(Vⱼ,V,d,k,kv_head,blockⱼ+k,document)
            end
            sync_threads()

            S = Fragment{(4,$(Bᶜ÷8))}(0f0)
            @fragment for d in 1:$(D÷16), n in 1:$(Bᶜ÷8)
                B = key_matrix(Kⱼ,8(n-1),16(d-1),lane)
                S[:,n] = muladd(A[:,d],B,Fragment{(4,)}(S[:,n]))
            end

            m̃₁,m̃₂ = -Inf32,-Inf32
            # A causal tile entirely before this query block needs no mask.
            if $(Window == (-1,0)) && blockⱼ+$(Int32(Bᶜ)) <= blockᵢ
                @fragment for n in 1:$(Bᶜ÷8), r in 1:4
                    S[r,n] *= τ
                end
            else
                @fragment for n in 1:$(Bᶜ÷8), r in 1:4
                    key = blockⱼ + 8(n-1) + 2part + mod(r-1,2) + 1
                    row = query + 8*((r-1)÷2)
                    valid = row <= T && key <= T && ($Window == (-1,0) ? key <= row : row-left <= key <= row+right)
                    S[r,n] = valid ? S[r,n]*τ : -Inf32
                end
            end
            @fragment for n in 1:$(Bᶜ÷8)
                m̃₁ = max(m̃₁,S[1,n],S[2,n])
                m̃₂ = max(m̃₂,S[3,n],S[4,n])
            end
            m̃₁,m̃₂ = reduce_lanes(max,m̃₁,Val(4)),reduce_lanes(max,m̃₂,Val(4))
            mⁿᵉʷ₁,mⁿᵉʷ₂ = max(m₁,m̃₁),max(m₂,m̃₂)
            safe₁,safe₂ = isfinite(mⁿᵉʷ₁) ? mⁿᵉʷ₁ : 0f0, isfinite(mⁿᵉʷ₂) ? mⁿᵉʷ₂ : 0f0
            α₁,α₂ = exp₂(m₁-safe₁),exp₂(m₂-safe₂)
            ℓ̃₁,ℓ̃₂ = 0f0,0f0
            @fragment for n in 1:$(Bᶜ÷8)
                S[1,n] = exp₂(S[1,n]-safe₁)
                S[2,n] = exp₂(S[2,n]-safe₁)
                S[3,n] = exp₂(S[3,n]-safe₂)
                S[4,n] = exp₂(S[4,n]-safe₂)
                ℓ̃₁ += S[1,n]+S[2,n]
                ℓ̃₂ += S[3,n]+S[4,n]
            end
            ℓ₁,ℓ₂ = α₁*ℓ₁+reduce_lanes(+,ℓ̃₁,Val(4)),α₂*ℓ₂+reduce_lanes(+,ℓ̃₂,Val(4))
            m₁,m₂ = mⁿᵉʷ₁,mⁿᵉʷ₂
            @fragment for d in 1:$(D÷8)
                𝕆[1,d] *= α₁
                𝕆[2,d] *= α₁
                𝕆[3,d] *= α₂
                𝕆[4,d] *= α₂
            end

            # Adjacent score fragments already have the next A operand's layout.
            @fragment for k in 1:$(Bᶜ÷16)
                P = (pack(S[1,2k-1],S[2,2k-1]),pack(S[3,2k-1],S[4,2k-1]),
                     pack(S[1,2k],S[2,2k]),pack(S[3,2k],S[4,2k]))
                for d in 1:$(D÷8)
                    B = value_matrix(Vⱼ,16(k-1),8(d-1),lane)
                    𝕆[:,d] = muladd(P,B,Fragment{(4,)}(𝕆[:,d]))
                end
            end
            sync_threads()
        end

        @fragment for d in 1:$(D÷8), r in 1:4
            channel = 8(d-1)+2part+mod(r-1,2)+1
            row = query+8*((r-1)÷2)
            if row <= T
                @inbounds O[channel,head,row,document] = 𝕆[r,d] / (r <= 2 ? ℓ₁ : ℓ₂)
            end
        end
        if part == 0
            if query <= T
                @inbounds ℓ[1,query,head,document] = ℓ₁
                @inbounds m[1,query,head,document] = m₁*$(Float32(log(2)))
            end
            if query+8 <= T
                @inbounds ℓ[1,query+8,head,document] = ℓ₂
                @inbounds m[1,query+8,head,document] = m₂*$(Float32(log(2)))
            end
        end
        nothing
    end
end

"""
    Δattention!(::TensorCoreInstruction,dQ,dK,dV,dO,Q,K,V,O,ℓ,m,window;
                query_tile=256,key_tile=256)

Correctness-first, tiled mixed-precision attention backward. Q/K/V are Float16;
all other arrays are Float32. Inputs use channel × head × token × batch layout;
ℓ,m have shape (1,T,H,B), with m in natural-log units as returned by the forward.
Accumulates into dQ/dK/dV without clearing them. Supports GQA, ragged sequences,
causal (-1,0), and nonnegative local window extents. All arrays must be on the
active GPU with compatible shapes; gradient outputs must not alias the inputs.

Uses conventional analytic softmax-backward equations on the rounded Q/K/V,
not the literal derivative of Float16 quantization. Forward rounds each tile's
unnormalized probabilities before P·V. Saved final ℓ,m do not recover those
exact rounded weights: this pass reconstructs Float32 P and uses Δ=sum(dO.*O)
from the saved Float32 output. Expect small mixed-precision differences versus
ideal attention, not exact finite-difference agreement across quantization.

Q/K/V conversion stays on the GPU and is limited to reusable tile buffers.
There is no saved T×T matrix; scratch is O(query_tile*key_tile +
D*(query_tile+key_tile)), independent of the number of heads and documents.
FP32 `mul!` follows the caller's CUDA math mode; use the default/pedantic mode
for correctness comparisons rather than enabling reduced-precision TF32 math.
"""
function Δattention!(::TensorCoreInstruction,
    dQ::CuArray{Float32,4}, dK::CuArray{Float32,4}, dV::CuArray{Float32,4},
    dO::CuArray{Float32,4},
    Q::CuArray{Float16,4}, K::CuArray{Float16,4}, V::CuArray{Float16,4},
    O::CuArray{Float32,4}, ℓ::CuArray{Float32,4}, m::CuArray{Float32,4},
    window::Tuple{Int,Int}; query_tile::Int=256, key_tile::Int=256,
)
    D,H,T,B = size(Q)
    Bᵣ,Bᶜ = min(T,query_tile),min(T,key_tile)
    heads_per_kv = H÷size(K,2)
    left,right = window
    τ = inv(sqrt(Float32(D)))

    Q₃₂ = similar(dO,D,Bᵣ)
    K₃₂,V₃₂ = ntuple(_ -> similar(dO,D,Bᶜ),2)
    S,dP = ntuple(_ -> similar(dO,Bᶜ,Bᵣ),2)
    M = similar(dO,Bool,Bᶜ,Bᵣ)
    dOO = similar(dO,D,Bᵣ)
    Δ = similar(dO,1,Bᵣ)

    for document in 1:B, head in 1:H, i in 1:cld(T,Bᵣ)
        kv_head = cld(head,heads_per_kv)
        blockᵢ = (i-1)*Bᵣ+1:min(i*Bᵣ,T)
        nᵢ = length(blockᵢ)
        @views begin
            Qᵢ = Q₃₂[:,1:nᵢ]
            Qᵢ .= Q[:,head,blockᵢ,document]
            dQᵢ = dQ[:,head,blockᵢ,document]
            dOᵢ = dO[:,head,blockᵢ,document]
            Oᵢ = O[:,head,blockᵢ,document]
            ℓᵢ = ℓ[:,blockᵢ,head,document]
            mᵢ = m[:,blockᵢ,head,document]
            dOOᵢ = dOO[:,1:nᵢ]
            Δᵢ = Δ[:,1:nᵢ]
        end
        @. dOOᵢ = dOᵢ*Oᵢ
        sum!(Δᵢ,dOOᵢ)

        first_key = window == (-1,0) ? 1 : max(1,first(blockᵢ)-left)
        last_key = min(T,last(blockᵢ)+right)
        for j in cld(first_key,Bᶜ):cld(last_key,Bᶜ)
            blockⱼ = (j-1)*Bᶜ+1:min(j*Bᶜ,T)
            nⱼ = length(blockⱼ)
            @views begin
                Kⱼ,Vⱼ = K₃₂[:,1:nⱼ],V₃₂[:,1:nⱼ]
                Kⱼ .= K[:,kv_head,blockⱼ,document]
                Vⱼ .= V[:,kv_head,blockⱼ,document]
                dKⱼ = dK[:,kv_head,blockⱼ,document]
                dVⱼ = dV[:,kv_head,blockⱼ,document]
                Sᵢⱼ = S[1:nⱼ,1:nᵢ]
                dPᵢⱼ = dP[1:nⱼ,1:nᵢ]
                Mᵢⱼ = M[1:nⱼ,1:nᵢ]
            end

            mul!(Sᵢⱼ,Kⱼ',Qᵢ,τ,0f0)
            attention_mask!(Mᵢⱼ,blockⱼ,blockᵢ,window)
            @. Sᵢⱼ = ifelse(Mᵢⱼ,Sᵢⱼ,-Inf32)
            @. Sᵢⱼ = exp(Sᵢⱼ-mᵢ)/ℓᵢ
            Pᵢⱼ = Sᵢⱼ

            mul!(dVⱼ,dOᵢ,Pᵢⱼ',1f0,1f0)
            mul!(dPᵢⱼ,Vⱼ',dOᵢ)
            @. dPᵢⱼ = Pᵢⱼ*(dPᵢⱼ-Δᵢ)
            dSᵢⱼ = dPᵢⱼ
            mul!(dQᵢ,Kⱼ,dSᵢⱼ,τ,1f0)
            mul!(dKⱼ,Qᵢ,dSᵢⱼ',τ,1f0)
        end
    end
    nothing
end

end # module FemtoChatCUDAExt
