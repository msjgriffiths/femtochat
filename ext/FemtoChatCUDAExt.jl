module FemtoChatCUDAExt

using CUDA
using CUDA: i32
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

instruction(::Type, dev::CuDevice, ::Val) = SIMTInstruction()
function instruction(::Type{F}, dev::CuDevice, ::Val{D}) where {F<:Union{Float16,Float32},D}
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

const LOG₂E = Float32(log2(exp(1.0)))

# K remains live throughout. V dies after register capture; all other buffers
# begin afterwards and reuse that complete region, not just V's first C rows.
function backward_layout(D,R,C)
    Kbytes = sizeof(Float16)*(D+8)*R
    Qbytes = sizeof(Float16)*(D+8)*C
    Sbytes = sizeof(Float16)*(C+8)*R
    stats = sizeof(Float32)*C
    offsets = (K=0,V=Kbytes,Q=Kbytes,dO=Kbytes+Qbytes,dS=Kbytes+2Qbytes,
               L=Kbytes+2Qbytes+Sbytes,Δ=Kbytes+2Qbytes+Sbytes+stats)
    (;bytes=Kbytes+max(Kbytes,2Qbytes+Sbytes+2stats),offsets)
end

# dS stores two neighboring queries for one key in a single shared word.
@inline function store_pair!(A,query,key,value::UInt32)
    @inbounds address = pointer(A,query+(key-1)*size(A,1))
    output = reinterpret(Core.LLVMPtr{UInt32,CUDA.AS.Shared},address)
    unsafe_store!(output,value,1,Val(4))
    nothing
end

# Physical dS is query × key. ldmatrix.trans restores the identical logical
# query-row × key-channel A fragment used by the fifth tensor multiply.
@inline transposed_query_matrix(A,q,k,lane) = load_matrix(
    shared_address(A,q+8*((lane÷8)%2)+1,k+8*(lane÷16)+lane%8+1),Val(4),Val(true))

# Couple the tile dimensions: R = 16W, C = max(16,8W).
# These are software defaults, not GPU-family or sequence-length switches.
function backward_geometry(D,budget)
    W,C = 12,96
    while backward_layout(D,16W,C).bytes > budget && W > 1
        W -= min(4,W÷2)
        C = 16max(1,W÷2)
    end
    Val(W),Val(C)
end

@inline function copy8_async!(destination,source,d,q,head,token,document)
    @inbounds begin
        output = reinterpret(Core.LLVMPtr{Half8,CUDA.AS.Shared},
            pointer(destination,d+(q-1)*size(destination,1)))
        if token <= size(source,3)
            index = LinearIndices(source)[d,head,token,document]
            input = reinterpret(Core.LLVMPtr{Half8,CUDA.AS.Global},pointer(source,index))
            CUDA.CG.pipeline_memcpy_async(output,input)
        else
            unsafe_store!(output,ntuple(_ -> VecElement(Int32(0)),Val(4)),1,Val(16))
        end
    end
    nothing
end

@inline copy8_commit!() = CUDA.CG.pipeline_commit()

# Waiting completes this thread's copies; the barrier also makes the completed
# copies and synchronous zero stores visible to all threads in the block.
@inline function copy8_wait!(::Val{Remaining}=Val(0)) where Remaining
    CUDA.CG.pipeline_wait_prior(Remaining)
    sync_threads()
    nothing
end


# Slices of a score fragment already have the next multiply's A layout.
@inline probability_matrix(S,k) = (
    pack(S[1,2k-1],S[2,2k-1]),pack(S[3,2k-1],S[4,2k-1]),
    pack(S[1,2k],S[2,2k]),pack(S[3,2k],S[4,2k]))

# Zero-based lane/query-tile; one-based register/channel-chunk/head/document.
@inline function fragment_index(lane,r,d,block,head,document,tiles,heads,::Val{D}) where D
    lane+Int32(1)+Int32(32)*(Int32(r)-Int32(1)+Int32(4)*(d-Int32(1)+
        Int32(D÷8)*(block+tiles*(head-Int32(1)+heads*(document-Int32(1))))))
end


# Unpadded shared storage and one query-owned forward implementation.
# Static slices in every arm: runtime selection returns four scalar registers,
# not a dynamically addressed NTuple or stack-backed probability array.
@inline @generated function probability_fragment(P::Fragment{Shape,UInt32},k::Int32) where Shape
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
@inline function shared_address(A::Swizzled,row,column)
    @inbounds address = pointer(A,row,column)
    reinterpret(UInt,address) % UInt32
end
@inline query_matrix(Q::Swizzled,q,d,lane) = load_matrix(
    shared_address(Q,d+8*(lane÷16)+1,q+lane%16+1),Val(4),Val(false))
# Four words contain the B operands for two adjacent eight-column MMAs.
@inline key_matrices(K::Swizzled,k,d,lane) = load_matrix(
    shared_address(K,d+8*((lane÷8)%2)+1,k+8*(lane÷16)+lane%8+1),Val(4),Val(false))
@inline value_matrices(V::Swizzled,k,d,lane) = load_matrix(
    shared_address(V,d+8*(lane÷16)+1,k+lane%16+1),Val(4),Val(true))

@inline function copy_query!(destination::Swizzled,source,d,q,head,token,document)
    @inbounds begin
        index = LinearIndices(source)[d,head,token,document]
        input = reinterpret(Core.LLVMPtr{Half8,CUDA.AS.Global},pointer(source,index))
        value = token <= size(source,3) ? CUDA.unsafe_cached_load(input,1,Val(16)) :
            ntuple(_ -> VecElement(Int32(0)),Val(4))
        output = reinterpret(Core.LLVMPtr{Half8,CUDA.AS.Shared},pointer(destination,d,q))
        unsafe_store!(output,value,1,Val(16))
    end
    nothing
end

@inline function copy8_async!(destination::Swizzled,source,d,q,head,token,document)
    @inbounds begin
        output = reinterpret(Core.LLVMPtr{Half8,CUDA.AS.Shared},pointer(destination,d,q))
        if token <= size(source,3)
            index = LinearIndices(source)[d,head,token,document]
            input = reinterpret(Core.LLVMPtr{Half8,CUDA.AS.Global},pointer(source,index))
            CUDA.CG.pipeline_memcpy_async(output,input)
        else
            unsafe_store!(output,ntuple(_ -> VecElement(Int32(0)),Val(4)),1,Val(16))
        end
    end
    nothing
end


@inline function copy_rows!(copy!,destination,source,head,first_token,document,
    ::Val{D},::Val{Rows},::Val{W},
) where {D,Rows,W}
    thread = threadIdx().x
    if (32W) % (D÷8) == 0
        # A thread keeps its channel word while advancing through token rows.
        # Both the row induction and the lane coordinates remain 32-bit.
        d = 8i32*((thread-1i32)%Int32(D÷8))+1i32
        first_row = (thread-1i32)÷Int32(D÷8)+1i32
        for row in first_row:Int32(32W÷(D÷8)):Int32(Rows)
            copy!(destination,source,d,row,head,first_token+row,document)
        end
    else
        # Preserve the original ownership for nondividing widths, e.g. D=48.
        for index in thread:32W:(D÷8*Rows)
            d,row = 8i32*((index-1i32)%Int32(D÷8))+1i32,(index-1i32)÷Int32(D÷8)+1i32
            copy!(destination,source,d,row,head,first_token+row,document)
        end
    end
    nothing
end

shared_bytes(D,W,U,C) = sizeof(Float16)*D*(16W*U+2C)

# C bounds score registers per lane; shared capacity targets two resident CTAs.
# The coverage adjustment changes only query ownership, not the reduction tile.
function forward_geometry(D,H,T,B,SM,budget;
    warps=nothing,query_subtiles::Val{U}=Val(1),key_tile::Val{C}=Val(64),
) where {U,C}
    W = 8
    if isnothing(warps)
        while W > 1 && shared_bytes(D,W,U,C) > budget
            W ÷= 2
        end
        while W > 1 && H*B*cld(T,16W*U) < SM
            W ÷= 2
        end
    end
    (;warps=something(warps,Val(W)),query_subtiles,key_tile)
end

"""
    attention!(TensorCoreInstruction(), O, ℓ, m, Q, K, V, window; ...)

Half Q/K/V with Float32 accumulation and saved statistics. The caller chooses
Float16 or Float32 output storage; the kernel converts only at its final store.
Compatible nonempty, nonaliasing buffers and valid grouped heads are expected.
"""
function attention!(𝒜::TensorCoreInstruction,
    O::CuArray{F,4},ℓ::CuArray{Float32,4},m::CuArray{Float32,4},
    Q::CuArray{Float16,4},K::CuArray{Float16,4},V::CuArray{Float16,4},
    window::Tuple{Int,Int};
    warps=nothing,query_subtiles::Val=Val(1),key_tile::Val=Val(64),
) where {F<:Union{Float16,Float32}}
    D,H,T,B = size(Q)
    dev = device(Q)
    budget = min(attribute(dev,CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN),
        attribute(dev,CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)÷2)
    SM = attribute(dev,CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    selected = forward_geometry(D,H,T,B,SM,budget;warps,query_subtiles,key_tile)
    launch_attention!(O,ℓ,m,Q,K,V,window,selected.warps,selected.query_subtiles,selected.key_tile)
end

function launch_attention!(O,ℓ,m,Q,K,V,window,warps::Val{W},subtiles::Val{U},key_tile::Val{C}) where {W,U,C}
    D,H,T,B = size(Q)
    @assert 0 < D <= 128 && D%16 == 0 && C > 0 && C%16 == 0 && U in (1,2) && 0 < W <= 32
    R = 16W*U
    args = (TensorCoreInstruction(),O,ℓ,m,Q,K,V,Val(window),Val(D),warps,subtiles,key_tile,
        Val(H÷size(K,2)),Val((D,H)),Val((D,size(K,2))))
    kernel = @cuda launch=false maxthreads=32W attention_forward!(args...)
    shmem = shared_bytes(D,W,U,C)
    attributes = CUDA.attributes(kernel.fun)
    if shmem > attributes[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES]
        attributes[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem
    end
    kernel(args...;threads=32W,blocks=(H,B,cld(T,R)),shmem)
    nothing
end

@inline function static_shape(A::CuDeviceArray{F,N,AS},shape) where {F,N,AS}
    CuDeviceArray{F,N,AS}(pointer(A),shape,A.maxsize)
end

# Only channel/head strides specialize; token and batch dimensions stay runtime.
function attention_forward!(instruction::TensorCoreInstruction,O,ℓ,m,Q,K,V,window,D,warps,subtiles,key_tile,groups,
    ::Val{QChannels},::Val{KChannels},
) where {QChannels,KChannels}
    O = static_shape(O,(QChannels...,size(O,3),size(O,4)))
    Q = static_shape(Q,(QChannels...,size(Q,3),size(Q,4)))
    K = static_shape(K,(KChannels...,size(K,3),size(K,4)))
    V = static_shape(V,(KChannels...,size(V,3),size(V,4)))
    ℓ = static_shape(ℓ,(1,size(ℓ,2),QChannels[2],size(ℓ,4)))
    m = static_shape(m,(1,size(m,2),QChannels[2],size(m,4)))
    @inline attention_kernel!(instruction,O,ℓ,m,Q,K,V,window,D,warps,subtiles,key_tile,groups)
end

@generated function attention_kernel!(::TensorCoreInstruction,O,ℓ,m,Q,K,V,
    ::Val{Window},::Val{D},::Val{W},::Val{U},::Val{C},::Val{Groups},
) where {Window,D,W,U,C,Groups}
    R = 16W*U
    subtiles = U
    mask = (1<<min(3,trailing_zeros(D÷8)))-1
    Qbytes = 2D*R
    Kbytes = 2D*C
    quote
        head,document,i = blockIdx(); i = gridDim().z-i+1i32
        thread = threadIdx().x
        warp,lane = (thread-1i32)÷32i32,(thread-1i32)%32i32
        group,part = lane÷4i32,lane%4i32
        T = size(Q,3)%Int32
        kv_head = (head-1i32)÷$(Int32(Groups))+1i32
        blockᵢ = (i-1i32)*$(Int32(R))
        left,right = $Window
        τ = $(Float32(log2(exp(1.0))/sqrt(D)))
        @inbounds begin
            Qᵢ = Swizzled(CuDynamicSharedArray(Float16,($D,$R),0),Val(WordSwizzle{8,$mask}))
            Kⱼ = Swizzled(CuDynamicSharedArray(Float16,($D,$C),$Qbytes),Val(WordSwizzle{8,$mask}))
            Vⱼ = Swizzled(CuDynamicSharedArray(Float16,($D,$C),$(Qbytes+Kbytes)),Val(WordSwizzle{8,$mask}))
        end
        copy_rows!(copy_query!,Qᵢ,Q,head,blockᵢ,document,Val($D),Val($R),Val($W))
        sync_threads()
        𝕆₁ = Fragment{(4,$(D÷8))}(0f0)
        𝕆₂ = 𝕆₁
        𝕞 = Fragment{(2,2)}(-Inf32)
        𝕝 = Fragment{(2,2)}(0f0)
        first_key = $Window == (-1,0) ? 0i32 : max(0i32,blockᵢ-left)÷$(Int32(C))*$(Int32(C))
        last_key = min(T,blockᵢ+$(Int32(R))+right)
        copy_rows!(copy8_async!,Kⱼ,K,kv_head,first_key,document,Val($D),Val($C),Val($W))
        copy8_commit!()
        for j in 0i32:cld(last_key-first_key,$(Int32(C)))-1i32
            blockⱼ = first_key+j*$(Int32(C))
            # Current K is ready; every reader of the previous V has retired.
            copy8_wait!()
            copy_rows!(copy8_async!,Vⱼ,V,kv_head,blockⱼ,document,Val($D),Val($C),Val($W))
            copy8_commit!()

            # Both query halves consume the same K fragment before it dies.
            # Keep their register representations separate rather than one 3D tuple.
            S₁ = Fragment{(4,$(C÷8))}(0f0)
            S₂ = S₁
            @fragment begin
                @loopinfo unroll=false for d in 0i32:16i32:$(Int32(D-16))
                    A₁ = query_matrix(Qᵢ,warp*16i32,d,lane)
                    A₂ = $subtiles == 2 ? query_matrix(Qᵢ,warp*16i32+$(Int32(16W)),d,lane) : A₁
                    for n in 1:$(C÷16)
                        B = key_matrices(Kⱼ,16(n-1),d,lane)
                        B₁,B₂ = (B[1],B[2]),(B[3],B[4])
                        S₁[:,2n-1] = muladd(A₁,B₁,Fragment{(4,)}(S₁[:,2n-1]))
                        S₁[:,2n] = muladd(A₁,B₂,Fragment{(4,)}(S₁[:,2n]))
                        if $subtiles == 2
                            S₂[:,2n-1] = muladd(A₂,B₁,Fragment{(4,)}(S₂[:,2n-1]))
                            S₂[:,2n] = muladd(A₂,B₂,Fragment{(4,)}(S₂[:,2n]))
                        end
                    end
                end
            end
            # Current V completes while QK runs. The barrier also retires
            # every K reader before the single K buffer receives its next tile.
            copy8_wait!()
            if blockⱼ+$(Int32(C)) < last_key
                copy_rows!(copy8_async!,Kⱼ,K,kv_head,blockⱼ+$(Int32(C)),document,Val($D),Val($C),Val($W))
                copy8_commit!()
            end
            𝒫₁ = Fragment{(4,$(C÷16))}(UInt32(0))
            𝒫₂ = 𝒫₁
            @fragment for u in 1:$subtiles
                𝕆 = u == 1 ? 𝕆₁ : 𝕆₂
                S = u == 1 ? S₁ : S₂
                q = warp*16i32+$(Int32(16W))*(u-1)
                query = blockᵢ+q+group+1i32
                if $(Window == (-1,0)) && blockⱼ+$(Int32(C)) <= blockᵢ
                    for n in 1:$(C÷8),r in 1:4
                        S[r,n] *= τ
                    end
                else
                    for n in 1:$(C÷8),r in 1:4
                        key = blockⱼ+8(n-1)+2part+mod(r-1,2)+1
                        row = query+8*((r-1)÷2)
                        valid = row <= T && key <= T && ($Window == (-1,0) ? key <= row : row-left <= key <= row+right)
                        S[r,n] = valid ? S[r,n]*τ : -Inf32
                    end
                end
                m̃₁,m̃₂ = -Inf32,-Inf32
                for n in 1:$(C÷8)
                    m̃₁ = max(m̃₁,S[1,n],S[2,n])
                    m̃₂ = max(m̃₂,S[3,n],S[4,n])
                end
                m̃₁,m̃₂ = reduce_lanes(max,m̃₁,Val(4)),reduce_lanes(max,m̃₂,Val(4))
                mⁿᵉʷ₁,mⁿᵉʷ₂ = max(𝕞[1,u],m̃₁),max(𝕞[2,u],m̃₂)
                safe₁,safe₂ = isfinite(mⁿᵉʷ₁) ? mⁿᵉʷ₁ : 0f0,isfinite(mⁿᵉʷ₂) ? mⁿᵉʷ₂ : 0f0
                α₁,α₂ = exp₂(𝕞[1,u]-safe₁),exp₂(𝕞[2,u]-safe₂)
                ℓ̃₁,ℓ̃₂ = 0f0,0f0
                for n in 1:$(C÷8)
                    S[1,n] = exp₂(S[1,n]-safe₁)
                    S[2,n] = exp₂(S[2,n]-safe₁)
                    S[3,n] = exp₂(S[3,n]-safe₂)
                    S[4,n] = exp₂(S[4,n]-safe₂)
                    ℓ̃₁ += S[1,n]+S[2,n]
                    ℓ̃₂ += S[3,n]+S[4,n]
                end
                # α is shared by the four lanes; their sums can stay local
                # until the final normalization, rather than shuffling every tile.
                𝕝[1,u] = α₁*𝕝[1,u]+ℓ̃₁
                𝕝[2,u] = α₂*𝕝[2,u]+ℓ̃₂
                𝕞[1,u] = mⁿᵉʷ₁
                𝕞[2,u] = mⁿᵉʷ₂
                for d in 1:$(D÷8)
                    𝕆[1,d] *= α₁
                    𝕆[2,d] *= α₁
                    𝕆[3,d] *= α₂
                    𝕆[4,d] *= α₂
                end
                # Finish the Float32 softmax phase before multiplying V.
                # Two probabilities share each UInt32 register; S is now dead.
                𝒫 = Fragment{(4,$(C÷16))}(UInt32(0))
                for k in 1:$(C÷16)
                    𝒫[:,k] = (pack(S[1,2k-1],S[2,2k-1]),pack(S[3,2k-1],S[4,2k-1]),
                              pack(S[1,2k],S[2,2k]),pack(S[3,2k],S[4,2k]))
                end
                if u == 1
                    𝕆₁,𝒫₁ = 𝕆,𝒫
                else
                    𝕆₂,𝒫₂ = 𝕆,𝒫
                end
            end
            # The second half reuses V's two packed words, not another load.
            @fragment begin
                @loopinfo unroll=false for k in 1i32:$(Int32(C÷16))
                    P₁ = probability_fragment(𝒫₁,k)
                    P₂ = $subtiles == 2 ? probability_fragment(𝒫₂,k) : P₁
                    for d in 1:$(D÷16)
                        B = value_matrices(Vⱼ,16i32*(k-1i32),16(d-1),lane)
                        B₁,B₂ = (B[1],B[2]),(B[3],B[4])
                        𝕆₁[:,2d-1] = muladd(P₁,B₁,Fragment{(4,)}(𝕆₁[:,2d-1]))
                        𝕆₁[:,2d] = muladd(P₁,B₂,Fragment{(4,)}(𝕆₁[:,2d]))
                        if $subtiles == 2
                            𝕆₂[:,2d-1] = muladd(P₂,B₁,Fragment{(4,)}(𝕆₂[:,2d-1]))
                            𝕆₂[:,2d] = muladd(P₂,B₂,Fragment{(4,)}(𝕆₂[:,2d]))
                        end
                    end
                end
            end
            # The next iteration's wait/barrier retires all PV readers.
        end
        @fragment for u in 1:$subtiles
            𝕝[1,u] = reduce_lanes(+,𝕝[1,u],Val(4))
            𝕝[2,u] = reduce_lanes(+,𝕝[2,u],Val(4))
            # Reuse one Float32 reciprocal per completed row; save the global ℓ.
            ℓ⁻¹ = (inv(𝕝[1,u]),inv(𝕝[2,u]))
            𝕆 = u == 1 ? 𝕆₁ : 𝕆₂
            query = blockᵢ+warp*16i32+$(Int32(16W))*(u-1)+group+1i32
            for d in 1:$(D÷8),r in 1:4
                channel = 8(d-1)+2part+mod(r-1,2)+1
                row = query+8*((r-1)÷2)
                if row <= T
                    @inbounds O[channel,head,row,document] = 𝕆[r,d]*ℓ⁻¹[(r-1)÷2+1]
                end
            end
            if part == 0
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

# Both paired sources have the same (D,H,T,B) shape, and both destinations
# have the same padded pitch. Share the validity test and linear offsets.
@inline function copy_pair!(::typeof(copy8!),output1,output2,input1,input2,valid)
    zero = ntuple(_ -> VecElement(Int32(0)),Val(4))
    a = valid ? CUDA.unsafe_cached_load(input1,1,Val(16)) : zero
    unsafe_store!(output1,a,1,Val(16))
    b = valid ? CUDA.unsafe_cached_load(input2,1,Val(16)) : zero
    unsafe_store!(output2,b,1,Val(16))
    nothing
end
@inline function copy_pair!(::typeof(copy8_async!),output1,output2,input1,input2,valid)
    if valid
        CUDA.CG.pipeline_memcpy_async(output1,input1)
        CUDA.CG.pipeline_memcpy_async(output2,input2)
    else
        zero = ntuple(_ -> VecElement(Int32(0)),Val(4))
        unsafe_store!(output1,zero,1,Val(16))
        unsafe_store!(output2,zero,1,Val(16))
    end
    nothing
end
@inline function copy_pair_offsets!(copy!::F,destination1,destination2,source1,source2,
    shared_index,global_index,valid,
) where F
    @inbounds begin
        output1 = reinterpret(Core.LLVMPtr{Half8,CUDA.AS.Shared},pointer(destination1,shared_index))
        output2 = reinterpret(Core.LLVMPtr{Half8,CUDA.AS.Shared},pointer(destination2,shared_index))
        input1 = reinterpret(Core.LLVMPtr{Half8,CUDA.AS.Global},pointer(source1,global_index))
        input2 = reinterpret(Core.LLVMPtr{Half8,CUDA.AS.Global},pointer(source2,global_index))
        copy_pair!(copy!,output1,output2,input1,input2,valid)
    end
    nothing
end

# Only matrix-copy loops use this helper. Global indices stay Int64; signed
# sequence coordinates and all matrix/statistics/barrier code remain unchanged.
@inline function copy_paired_rows!(copy!::F,destination1,destination2,source1,source2,
    head,first_token,document,::Val{D},::Val{Rows},::Val{W},
) where {F,D,Rows,W}
    thread = threadIdx().x
    T = size(source1,3)%Int32
    global_pitch = D*size(source1,2)
    # One runtime shape-derived base for both sources, even for query tails.
    indices = LinearIndices((D,Base.tail(size(source1))...))
    base = @inbounds indices[1,head,first_token+1i32,document]
    if (32W) % (D÷8) == 0
        d = 8i32*((thread-1i32)%Int32(D÷8))+1i32
        first_row = (thread-1i32)÷Int32(D÷8)+1i32
        row_stride = Int32(32W÷(D÷8))
        shared_first = d+(first_row-1i32)*Int32(D+8)
        global_first = base+(d-1i32)+(first_row-1i32)*global_pitch
        global_step = row_stride*global_pitch
        @loopinfo unroll=true for step in 0i32:Int32(cld(D÷8*Rows,32W)-1)
            row = first_row+step*row_stride
            if D÷8*Rows % (32W) == 0 || row <= Int32(Rows)
                shared_index = shared_first+step*Int32((32W÷(D÷8))*(D+8))
                global_index = global_first+step*global_step
                copy_pair_offsets!(copy!,destination1,destination2,source1,source2,
                    shared_index,global_index,first_token+row<=T)
            end
        end
    else
        # General widths keep the original per-thread linear-word order.
        @loopinfo unroll=true for step in 0i32:Int32(cld(D÷8*Rows,32W)-1)
            index = thread+step*Int32(32W)
            if D÷8*Rows % (32W) == 0 || index <= Int32(D÷8*Rows)
                d = 8i32*((index-1i32)%Int32(D÷8))+1i32
                row = (index-1i32)÷Int32(D÷8)+1i32
                shared_index = d+(row-1i32)*Int32(D+8)
                global_index = base+(d-1i32)+(row-1i32)*global_pitch
                copy_pair_offsets!(copy!,destination1,destination2,source1,source2,
                    shared_index,global_index,first_token+row<=T)
            end
        end
    end
    nothing
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

@inline function store_half8!(destination,source,input_index,output_index)
    @inbounds begin
        input = reinterpret(Core.LLVMPtr{Half8,CUDA.AS.Shared},pointer(source,input_index))
        output = reinterpret(Core.LLVMPtr{Half8,CUDA.AS.Global},pointer(destination,output_index))
        values = unsafe_load(input,1,Val(16))
        unsafe_store!(output,values,1,Val(16))
    end
    nothing
end

# Both storage contracts share row statistics and padded scratch clearing.
# Nothing means no conversion buffer and no scale; those branches disappear.
function prepare_upstream!(dO₁₆,δQ,Δ,L,dO,O,ℓ,m,s,::Val{D}) where D
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
            dO₁₆ === nothing || (dO₁₆[d,head,query,document] = Float16(δ))
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

@inline function load8(A::CuDeviceArray{Float16,4},channel,head,query,document)
    index=@inbounds LinearIndices(A)[channel,head,query,document]
    pointer8=reinterpret(Core.LLVMPtr{Half8,CUDA.AS.Global},pointer(A,index))
    CUDA.unsafe_cached_load(pointer8,1,Val(16))
end

# Every half product is exact in Float32. Only the per-lane accumulation and
# subgroup reduction round; there is no half-precision arithmetic here.
@inline function dot8(a::Half8,b::Half8,value::Float32=0f0)
    @fragment for word in 1:4
        x,y=a[word].value,b[word].value
        value=muladd(Float32(reinterpret(Float16,x%UInt16)),
            Float32(reinterpret(Float16,y%UInt16)),value)
        value=muladd(Float32(reinterpret(Float16,(x>>>16)%UInt16)),
            Float32(reinterpret(Float16,(y>>>16)%UInt16)),value)
    end
    value
end

@inline function clear4!(scratch::CuDeviceArray{Float32,1},index)
    destination=reinterpret(Core.LLVMPtr{Float32ᵛ{4},CUDA.AS.Global},pointer(scratch,index))
    unsafe_store!(destination,ntuple(_->VecElement(0f0),Val(4)),1,Val(16))
    nothing
end

@generated function prepare_vector!(δQ,Δ,L,dO,output,ℓ,m,::Val{D},::Val{Values},::Val{Rows}) where {D,Values,Rows}
    D in (64,128) || return :(@inline prepare_upstream!(nothing,δQ,Δ,L,dO,output,ℓ,m,nothing,Val($D)))
    G=D÷Values
    quote
        thread=threadIdx().x-1i32
        lane,group=thread%$(Int32(G)),thread÷$(Int32(G))
        i,head,document=blockIdx()
        groups=blockDim().x÷$(Int32(G))
        T,H=size(output,3)%Int32,size(output,2)%Int32
        padded=16i32*cld(T,16i32)
        @fragment for row in 0:$(Rows-1)
            query=((i-1i32)*$(Int32(Rows))+row)*groups+group+1i32
            if query<=padded
                offset=$(Int32(D))*(query-1i32+padded*(head-1i32+H*(document-1i32)))
                # Each instruction writes one contiguous row segment across the group.
                @fragment for segment in 0:$(Values÷4-1)
                    clear4!(δQ,offset+4i32*lane+$(Int32(4G))*segment+1i32)
                end
            end
            value=0f0
            if query<=T
                @fragment for chunk in 0:$(Values÷8-1)
                    channel=8i32*lane+$(Int32(8G))*chunk+1i32
                    a=load8(dO,channel,head,query,document)
                    b=load8(output,channel,head,query,document)
                    value=dot8(a,b,value)
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


function prepare!(δQ,Δ,L,dO::CuArray{Float16,4},O,ℓ,m,::Val{D},::Val{W}) where {D,W}
    _,H,T,B = size(dO)
    if D in (64,128)
        # Eight lanes per row, two rows per subgroup:64 queries/256-thread CTA.
        @cuda threads=256 blocks=(cld(16cld(T,16),64),H,B) prepare_vector!(
            δQ,Δ,L,dO,O,ℓ,m,Val(D),Val(D÷8),Val(2))
    else
        @cuda threads=32W blocks=(cld(16cld(T,16),W),H,B) prepare_upstream!(
            nothing,δQ,Δ,L,dO,O,ℓ,m,nothing,Val(D))
    end
    dO,nothing
end

function prepare!(δQ,Δ,L,dO::CuArray{Float32,4},O,ℓ,m,::Val{D},::Val{W}) where {D,W}
    _,H,T,B = size(dO)
    s = mapreduce(abs,max,vec(dO);dims=1)
    @. s = ifelse(iszero(s),1f0,exp2(clamp(floor(log2(32f0/s)),-120f0,120f0)))
    dO₁₆ = similar(dO,Float16)
    @cuda threads=32W blocks=(cld(16cld(T,16),W),H,B) prepare_upstream!(
        dO₁₆,δQ,Δ,L,dO,O,ℓ,m,s,Val(D))
    dO₁₆,s
end

# Preserve the final rounding boundary while transposing the MMA scratch.
@inline stage_pair!(S::CuDeviceArray{Float16},d,q,a,b) = store_pair!(S,d,q,pack(a,b))
@inline function stage_pair!(S::CuDeviceArray{Float32},d,q,a,b)
    @inbounds S[d,q],S[d+1,q] = a,b
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
        @fragment for pair in 0:1
            q = row+8i32*pair
            query = 16i32*tile+q
            x,y = pair==0 ? (unscale*a,unscale*b) : (unscale*c,unscale*e)
            if query<=T
                @inbounds index = LinearIndices(dQ)[channel,head,query,document]
                x = gradient_value(dQ,index,x,policy)
                y = gradient_value(dQ,index+1,y,policy)
            end
            stage_pair!(S,channel,q,x,y)
        end
    end
    sync_threads()
    width = Int32(16÷sizeof(F))
    for index in thread:128i32:Int32(16D÷(16÷sizeof(F)))
        channel = width*((index-1i32)%Int32(D÷(16÷sizeof(F))))+1i32
        row = (index-1i32)÷Int32(D÷(16÷sizeof(F)))+1i32
        query = 16i32*tile+row
        if query<=T
            @inbounds begin
                input = reinterpret(Core.LLVMPtr{Half8,CUDA.AS.Shared},pointer(S,channel+(row-1i32)*Int32(D+8)))
                output = reinterpret(Core.LLVMPtr{Half8,CUDA.AS.Global},pointer(dQ,LinearIndices(dQ)[channel,head,query,document]))
                unsafe_store!(output,unsafe_load(input,1,Val(16)),1,Val(16))
            end
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


# Structural backward launch policy; no timing-based tuning.
# Four compact warps is a software choice, not a GPU architectural constant.
# A compact tile is square: R=C=16W. Large tiles retain the existing R=2C rule.
function initial_geometry(D,H,Hkv,T,B,budget,sms,compact_warps)
    W,C = backward_geometry(D,budget)
    w = typeof(W).parameters[1]
    while w>1 && H*B*cld(T,16w)<sms
        w -= min(4,w÷2)
    end
    c = 16max(1,w÷2)
    if w<=compact_warps && backward_layout(D,16w,16w).bytes<=budget
        c = 16w
    end
    span = H>Hkv && Hkv*B*cld(T,16w)>=sms ? H÷Hkv : 1
    (;warps=w,queries=c,span)
end

prefer_compact(large_blocks,large_residency,compact_blocks,compact_residency,sms) =
    large_residency>0 && compact_residency>large_residency &&
    cld(compact_blocks,sms*compact_residency)<=cld(large_blocks,sms*large_residency)

# This barrier specializes on ordinary tuple/Val types. It never executes GPU
# work, copies an array, initializes scratch, or constructs a heterogeneous Dict.
Base.@noinline function compile_candidate(common::Tuple,::Val{W},::Val{C},
    groups::Val,span::Val,policy::Val,
) where {W,C}
    args = (common...,Val(W),Val(C),groups,span,policy)
    kernel = @cuda launch=false maxthreads=32W attention_gradient!(args...)
    D = size(common[4],1)
    shmem = backward_layout(D,16W,C).bytes
    attributes = CUDA.attributes(kernel.fun)
    if shmem>attributes[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES]
        attributes[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem
    end
    residency = CUDA.active_blocks(kernel.fun,32W;shmem)
    (;kernel,args,shmem,residency)
end

Base.@noinline function launch_candidate!(candidate,::Val{W},H,T,B,::Val{Span}) where {W,Span}
    candidate.kernel(candidate.args...;threads=32W,blocks=(H÷Span,B,cld(T,16W)),shmem=candidate.shmem)
    nothing
end

# Keep branches separate through launch: differently specialized HostKernels
# are never merged into a resource dictionary or a large dynamic result union.
Base.@noinline function select_and_launch!(common::Tuple,::Val{W},::Val{C},
    groups::Val,span::Val{Span},policy::Val,::Val{Compact},sms,shared_per_sm,reserved_shared,
) where {W,C,Span,Compact}
    D,H,T,B = size(common[4])
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
            return (;warps=Compact,queries=16Compact,head_span=Span,
                    active_blocks=small.residency,compact_selected=true)
        end
    end
    launch_candidate!(large,Val(W),H,T,B,span)
    (;warps=W,queries=C,head_span=Span,active_blocks=large.residency,compact_selected=false)
end


"""
    Δattention!(::TensorCoreInstruction,dQ,dK,dV,dO,Q,K,V,O,ℓ,m,window;
                accumulate=true,warps=nothing,key_tile=nothing,head_span=nothing)

One tiled Tensor Core backward. Q/K/V are Float16; O/dO/dQ/dK/dV share
Float16 or Float32 storage. Statistics, atomics, and partial KV sums remain
Float32. Float32 upstream retains GPU power-of-two scaling; Half upstream is
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
    dQ::CuArray{F,4},dK::CuArray{F,4},dV::CuArray{F,4},dO::CuArray{F,4},
    Q::CuArray{Float16,4},K::CuArray{Float16,4},V::CuArray{Float16,4},
    O::CuArray{F,4},ℓ::CuArray{Float32,4},m::CuArray{Float32,4},window::Tuple{Int,Int};
    accumulate::Bool=true,warps=nothing,key_tile=nothing,head_span=nothing,
) where {F<:Union{Float16,Float32}}
    D,H,T,B = size(Q)
    dev = device(Q)
    budget = attribute(dev,CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    sms = attribute(dev,CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    shared_per_sm = attribute(dev,CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
    reserved_shared = attribute(dev,CUDA.DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK)
    geometry = initial_geometry(D,H,size(K,2),T,B,budget,sms,4)
    W = something(warps,Val(geometry.warps))
    C = something(key_tile,Val(geometry.queries))
    Span = something(head_span,Val(geometry.span))
    automatic = isnothing(warps) && isnothing(key_tile) && isnothing(head_span)
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
    dO₁₆,s = prepare!(δQ,Δ,L,dO,O,ℓ,m,Val(D),Val(W))
    common = (δQ,δK,δV,Q,K,V,dO₁₆,L,Δ,s,Val(window),Val(D))
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
        group,part = lane÷4i32,lane%4i32
        T = size(Q,3) % Int32
        blockⱼ = (j-1i32)*$(Int32(R))
        key = blockⱼ+16i32*warp+group+1i32
        left,right = $(Int32.(Window))
        τ = $(Float32(inv(sqrt(D))))
        τ₂ = $(Float32(log2(exp(1.0))/sqrt(D)))
        unscale = s === nothing ? 1f0 : (@inbounds inv(s[1]))

        # The same multiply, transposed: rows are keys, columns are queries.
        @inbounds begin
            Kⱼ = CuDynamicSharedArray(Float16,($D+8,$R),$(offsets.K))
            Vⱼ = CuDynamicSharedArray(Float16,($D+8,$R),$(offsets.V))
            Qᵢ = CuDynamicSharedArray(Float16,($D+8,$C),$(offsets.Q))
            dOᵢ = CuDynamicSharedArray(Float16,($D+8,$C),$(offsets.dO))
            dSᵢⱼ = CuDynamicSharedArray(Float16,($C+8,$R),$(offsets.dS))
            Lᵢ = CuDynamicSharedArray(Float32,$C,$(offsets.L))
            Δᵢ = CuDynamicSharedArray(Float32,$C,$(offsets.Δ))
        end
        copy_paired_rows!(copy8!,Kⱼ,Vⱼ,K,V,kv_head,blockⱼ,document,Val($D),Val($R),Val($W))
        sync_threads()
        dA = Fragment{(4,$(D÷16))}(UInt32(0))
        @fragment for d in 1:$(D÷16)
            dA[:,d] = query_matrix(Vⱼ,16warp,16(d-1),lane)
        end
        sync_threads()
        # V is no longer live; Q/dO/dS/statistics can now overwrite its storage.
        𝔾ₖ = Fragment{(4,$(D÷8))}(0f0)
        𝔾ᵥ = Fragment{(4,$(D÷8))}(0f0)
        first_query = max(0i32,blockⱼ-($Window == (-1,0) ? 0i32 : right))÷$(Int32(C))*$(Int32(C))
        last_query = $Window == (-1,0) ? T : min(T,blockⱼ+$(Int32(R))+left)

        # One CTA owns Span consecutive query heads from the same KV group.
        # Keep both gradients live across heads; prime each head's Q/dO tile.
        @loopinfo unroll=false for head in ((shard-1i32)*$(Int32(Span))+1i32):(shard*$(Int32(Span)))
        # Prime the first query tile after V's register capture has completed.
        copy_paired_rows!(copy8_async!,Qᵢ,dOᵢ,Q,dO,head,first_query,document,Val($D),Val($C),Val($W))
        copy8_commit!()
        for row in thread:$(32W):$C
            query = first_query+row
            @inbounds Lᵢ[row] = query <= T ? L[1,query,head,document] : 0f0
            @inbounds Δᵢ[row] = query <= T ? Δ[1,query,head,document] : 0f0
        end
        copy8_wait!()

        for i in first_query÷$(Int32(C)):cld(last_query,$(Int32(C)))-1i32
            blockᵢ = i*$(Int32(C))
            interior = blockⱼ+$(Int32(R)) <= T && blockᵢ+$(Int32(C)) <= T &&
                ($Window == (-1,0) ? blockⱼ+$(Int32(R)) <= blockᵢ+1i32 :
                 blockᵢ+$(Int32(C))-left <= blockⱼ+1i32 &&
                 blockⱼ+$(Int32(R)) <= blockᵢ+1i32+right)
            @loopinfo unroll=false for column in 0i32:16i32:$(Int32(C-16))
                if $Window == (-1,0) && !interior &&
                    (blockⱼ+16i32*warp+1i32 > min(T,blockᵢ+column+16i32) || blockᵢ+column >= T)
                    @fragment for n in 1:2
                        row=16warp+group+1
                        query=column+8(n-1)+2part+1
                        store_pair!(dSᵢⱼ,query,row,UInt32(0))
                        store_pair!(dSᵢⱼ,query,row+8,UInt32(0))
                    end
                else
                P = Fragment{(4,2)}(0f0)
                dS = Fragment{(4,2)}(0f0)
                @fragment for d in 1:$(D÷16)
                    A = query_matrix(Kⱼ,16warp,16(d-1),lane)
                    for n in 1:2
                        B = key_matrix(Qᵢ,column+8(n-1),16(d-1),lane)
                        P[:,n] = muladd(A,B,Fragment{(4,)}(P[:,n]))
                        B = key_matrix(dOᵢ,column+8(n-1),16(d-1),lane)
                        dS[:,n] = muladd(dA[:,d],B,Fragment{(4,)}(dS[:,n]))
                    end
                end
                if interior
                    @fragment for n in 1:2, r in 1:4
                        col = column+8(n-1)+2part+mod(r-1,2)+1
                        query = blockᵢ+col
                        @inbounds p = exp₂(muladd(P[r,n],τ₂,-Lᵢ[col]))
                        @inbounds dS[r,n] = p*(dS[r,n]-Δᵢ[col])
                        P[r,n] = p
                    end
                else
                    @fragment for n in 1:2, r in 1:4
                        col = column+8(n-1)+2part+mod(r-1,2)+1
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
                p = probability_matrix(P,1)
                ds = probability_matrix(dS,1)
                # The same rounded words feed dK and the transposed dQ tile.
                @fragment for n in 1:2
                    row = 16warp+group+1
                    query = column+8(n-1)+2part+1
                    store_pair!(dSᵢⱼ,query,row,ds[2n-1])
                    store_pair!(dSᵢⱼ,query,row+8,ds[2n])
                end
                @fragment for d in 1:$(D÷8)
                    B = value_matrix(Qᵢ,column,8(d-1),lane)
                    𝔾ₖ[:,d] = muladd(ds,B,Fragment{(4,)}(𝔾ₖ[:,d]))
                    B = value_matrix(dOᵢ,column,8(d-1),lane)
                    𝔾ᵥ[:,d] = muladd(p,B,Fragment{(4,)}(𝔾ᵥ[:,d]))
                end
                end
            end
            sync_threads()

            # All current Q/dO reads are finished. Their next tile can arrive
            # while dQ uses only resident K and the completed dS tile.
            if blockᵢ+$(Int32(C)) < last_query
                copy_paired_rows!(copy8_async!,Qᵢ,dOᵢ,Q,dO,head,blockᵢ+$(Int32(C)),document,Val($D),Val($C),Val($W))
                copy8_commit!()
                for row in thread:$(32W):$C
                    query = blockᵢ+$(Int32(C))+row
                    @inbounds Lᵢ[row] = query <= T ? L[1,query,head,document] : 0f0
                    @inbounds Δᵢ[row] = query <= T ? Δ[1,query,head,document] : 0f0
                end
            end

            # K stays resident in shared storage throughout the tiled query loop.
            # When C has fewer query blocks than warps, use the remaining
            # warp dimension for disjoint output-channel chunks.
            if warp < $(Int32(query_warps*channel_warps))
                for q₀ in (16i32*(warp%$(Int32(query_warps)))):$(Int32(16query_warps)):$(Int32(C-1))
                    @loopinfo unroll=false for d₀ in (1i32+warp÷$(Int32(query_warps))):$(Int32(2channel_warps)):$(Int32(D÷8))
                        # Two independent outputs reuse each dS matrix load.
                        𝔾q = Fragment{(4,2)}(0f0)
                        second = $(D÷8 % (2channel_warps) == 0) || d₀+$(Int32(channel_warps)) <= $(Int32(D÷8))
                        ds = transposed_query_matrix(dSᵢⱼ,q₀,0i32,lane)
                        B₁ = value_matrix(Kⱼ,0i32,8i32*(d₀-1i32),lane)
                        B₂ = second ? value_matrix(Kⱼ,0i32,8i32*(d₀+$(Int32(channel_warps))-1i32),lane) : (0x00000000,0x00000000)
                        # Load the next fragments while the current multiply runs.
                        # Peel the last multiply to avoid guarded lookahead loads.
                        for k₀ in 0i32:16i32:$(Int32(R-32))
                            dsⁿᵉʷ = transposed_query_matrix(dSᵢⱼ,q₀,k₀+16i32,lane)
                            @fragment 𝔾q[:,1] = muladd(ds,B₁,Fragment{(4,)}(𝔾q[:,1]))
                            B₁ⁿᵉʷ = value_matrix(Kⱼ,k₀+16i32,8i32*(d₀-1i32),lane)
                            if second
                                @fragment 𝔾q[:,2] = muladd(ds,B₂,Fragment{(4,)}(𝔾q[:,2]))
                            end
                            B₂ⁿᵉʷ = second ? value_matrix(Kⱼ,k₀+16i32,8i32*(d₀+$(Int32(channel_warps))-1i32),lane) : B₂
                            ds,B₁,B₂ = dsⁿᵉʷ,B₁ⁿᵉʷ,B₂ⁿᵉʷ
                        end
                        @fragment 𝔾q[:,1] = muladd(ds,B₁,Fragment{(4,)}(𝔾q[:,1]))
                        if second
                            @fragment 𝔾q[:,2] = muladd(ds,B₂,Fragment{(4,)}(𝔾q[:,2]))
                        end
                        @fragment for n in 1:2, r in 1:4
                            d = d₀+(n-1)*$(Int32(channel_warps))
                            query = blockᵢ+q₀+group+8*((r-1)÷2)+1
                            if query <= T && ($(D÷8 % (2channel_warps) == 0) || d <= $(Int32(D÷8)))
                                index = fragment_index(lane,r,d,(blockᵢ+q₀)÷16i32,
                                    head,document,cld(T,16i32),size(Q,2)%Int32,Val($D))
                                CUDA.atomic_add!(pointer(dQ,index),𝔾q[r,n])
                            end
                        end
                    end
                end
            end
            # Finish next-tile copies and all dS readers before reusing dS.
            copy8_wait!()
        end
        end # owned query heads
        if $Span == $Groups && eltype(dK) == Float16
            # Every input/dS reader retired at the final wait. Stage final
            # half values only after optionally adding the old value in FP32.
            @inbounds begin
                gradientK = CuDynamicSharedArray(Float16,($D,$R),0)
                gradientV = CuDynamicSharedArray(Float16,($D,$R),$(2D*R))
            end
            @fragment for d in 1:$(D÷8), pair in 1:2
                channel = 8(d-1)+2part+1
                local_row = 16warp+group+8(pair-1)+1
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
                store_pair!(gradientK,channel,local_row,pack(k₁,k₂))
                store_pair!(gradientV,channel,local_row,pack(v₁,v₂))
            end
            sync_threads()
            @fragment for vector in 0:$(D÷16-1)
                index = thread+$(Int32(32W))*vector
                channel = 8i32*((index-1i32)%$(Int32(D÷8)))+1i32
                row = (index-1i32)÷$(Int32(D÷8))+1i32
                if blockⱼ+row <= T
                    @inbounds output = LinearIndices(dK)[channel,kv_head,blockⱼ+row,document]
                    store_half8!(dK,gradientK,8index-7i32,output)
                    store_half8!(dV,gradientV,8index-7i32,output)
                end
            end
        else
            @fragment for d in 1:$(D÷8), r in 1:4
                channel = 8(d-1)+2part+mod(r-1,2)+1
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
