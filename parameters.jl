module Parameters

using Core: BFloat16
using Base: @kwdef

export Params,
       GPTConfig,
       Embedding,
       Linear,
       MLP,
       CausalSelfAttention,
       Block,
       Transformer,
       🤖,
       ParamSpec,
       parameter_layout,
       paramview

const ParamFloat = Union{Float32,BFloat16}

"""
Params implements a struct to hold the parameter vector and gradient vector for a neural
    network. 

Example:
    p = Params{Float32}(randn(10), zeros(10))
    p.Θ .= randn(10)
    p.δ .= randn(10)
"""
struct Params{
    T<:ParamFloat,
    P<:AbstractVector{T},
    G<:AbstractVector{Float32}
}
    Θ::P
    δ::G

    function Params(Θ::P, δ::G) where {
        T<:ParamFloat,
        P<:AbstractVector{T},
        G<:AbstractVector{Float32},
    }
        length(Θ) == length(δ) ||
            throw(DimensionMismatch("Θ and δ must have equal length"))

        new{T,P,G}(Θ, δ)
    end
end


"""
Create parameter storage from an existing parameter vector.

Gradient storage is allocated using `similar`, preserving the storage
backend (e.g. CPU or CUDA) while changing the element type to Float32.
"""
function Params(Θ::AbstractVector{T}) where {T<:ParamFloat}
    δ = similar(Θ, Float32)
    fill!(δ, 0f0)
    Params(Θ, δ)
end

"""
Configuration parameters for GPT model. These are used to determine how
    many parameters are in the model, and to map them to the parameter vector.
"""
@kwdef struct GPTConfig
    sequence_len::Int = 2^11
    vocab_size::Int = 2^15
    n_layer::Int = 12
    n_head::Int = 6
    n_kv_head::Int = 6
    n_embed::Int = 768
    window_pattern::String = "SSL"
end

struct Linear{
    T<:ParamFloat,
    A<:AbstractMatrix{T},
}
    𝕎::A
end

struct Embedding{
    T<:ParamFloat,
    A<:AbstractMatrix{T},
}
    𝔼::A
end

struct MLP{L<:Linear}
    𝔽::L
    ℙ::L
end

struct CausalSelfAttention{L<:Linear}
    layer_idx::Int
    𝕎::L
    𝕂::L
    𝕍::L
    ℙ::L
    𝕧𝕖::Union{Nothing,L}
    window::Tuple{Int,Int}
    head_dim::Int
    n_head::Int
    n_kv_head::Int
end 

"""
Attention block
Attention is 👀 since it looks at everything, and 🧠 is a callback to old Perceptron days.
"""
struct Block 
    👀::CausalSelfAttention
    🧠::MLP
    🍰::Union{Nothing,Embedding}
    λᵦ::AbstractVector{<:ParamFloat}
    λx₀::AbstractVector{<:ParamFloat}
end

struct Transformer{
    E<:Embedding,
    B<:AbstractVector{<:Block},
}
    embed::E
    blocks::B
end

struct 🤖{
    T<:Transformer,
    L<:Linear,
    W<:AbstractVector{Tuple{Int,Int}},
    S<:Linear,
    SL<:AbstractVector{<:ParamFloat},
    BL<:AbstractVector{<:ParamFloat},
    R<:Tuple{AbstractMatrix{<:AbstractFloat}, AbstractMatrix{<:AbstractFloat}},
}
    config::GPTConfig

    window_sizes::W
    padded_vocab_size::Int

    transformer::T
    lm_head::L

    smear_gate::S
    λₛ::SL # smear lambda
    λᵧ::BL # Backout lambda

    rotary_seq_len::Int
    rope_sin_cos::R
end

# -----------------------------------------------------------------------------
# Parameter layout
# -----------------------------------------------------------------------------

"""
Description of one tensor inside the flat parameter vector.

`range` identifies its elements in Θ/δ.
`shape` gives the tensor shape obtained when that range is reshaped.
"""
struct ParamSpec{N}
    range::UnitRange{Int}
    shape::NTuple{N,Int}
end

Base.length(p::ParamSpec) = length(p.range)


"""
Return a shaped zero-copy view of parameter `p` from flat buffer `buf`.
"""
paramview(buf::AbstractVector, p::ParamSpec) =
    reshape(@view(buf[p.range]), p.shape)


"""
Whether a transformer layer uses a value embedding.

`layer_idx` uses Julia's 1-based indexing.
"""
has_ve(layer_idx::Int, n_layer::Int) =
    (layer_idx - 1) % 2 == (n_layer - 1) % 2

function window_sizes(config::GPTConfig)
    pattern = collect(config.window_pattern)
    L = config.sequence_len
    # Short is 4x smaller than long, and rounded up to a multiple of 128
    # ... so long as it's not longer than the long sequence length.
    S = min(L, cld(L, 4*128) * 128)
    windows = [pattern[mod1(i, length(pattern))] == 'S' ? (S, 0) : (L, 0) for i in 1:config.n_layer]
    # Force last layer to be full-length
    windows[end]  = (L, 0)
    windows
end


"""
Generate the complete mapping from GPT parameters into one flat vector.

All parameters belonging to an individual transformer block are
contiguous in memory.
"""
function parameter_layout(
    config::GPTConfig;
    pad_vocab_size_to::Int = 64,
)
    @assert config.n_embed % config.n_head == 0
    @assert config.n_kv_head <= config.n_head
    @assert config.n_head % config.n_kv_head == 0

    # -------------------------------------------------------------------------
    # Flat-vector allocator
    # -------------------------------------------------------------------------

    i = 1

    function take(shape::Int...)
        n = prod(shape)
        r = i:(i + n - 1)
        i += n
        ParamSpec(r, shape)
    end


    # -------------------------------------------------------------------------
    # Derived dimensions
    # -------------------------------------------------------------------------

    d = config.n_embed

    head_dim = d ÷ config.n_head
    kv_dim = config.n_kv_head * head_dim

    padded_vocab_size =
        cld(config.vocab_size, pad_vocab_size_to) * pad_vocab_size_to


    # -------------------------------------------------------------------------
    # Transformer
    # -------------------------------------------------------------------------

    embedding = take(d, padded_vocab_size)

    blocks = [
        (
            👀 = (
                # Query: d → d
                𝕎 = take(d, d),

                # Key/value: d → kv_dim
                𝕂 = take(kv_dim, d),
                𝕍 = take(kv_dim, d),

                # Attention output: d → d
                ℙ = take(d, d),

                # Value-embedding gate: 12 → n_kv_head
                𝕧𝕖 = has_ve(layer_idx, config.n_layer) ?
                      take(config.n_kv_head, 12) :
                      nothing,
            ),

            🧠 = (
                # MLP expansion: d → 4d
                𝔽 = take(4d, d),

                # MLP projection: 4d → d
                ℙ = take(d, 4d),
            ),

            🍰 = (
                𝔼 = has_ve(layer_idx, config.n_layer) ? take(kv_dim, padded_vocab_size) : nothing,
            ),

            λᵦ = take(1),
            λx₀ = take(1),
        )
        for layer_idx in 1:config.n_layer
    ]

    transformer = (
        embedding = embedding,
        blocks = blocks,
    )


    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    lm_head = take(padded_vocab_size, d)


    # -------------------------------------------------------------------------
    # Smear / backout
    # -------------------------------------------------------------------------

    # Smear gate: first 24 embedding channels → one scalar gate
    smear_gate = take(1, 24)

    λₛ = take(1)
    λᵧ = take(1)


    # -------------------------------------------------------------------------
    # Complete layout
    # -------------------------------------------------------------------------

    return (
        transformer = transformer,

        lm_head = lm_head,

        smear_gate = smear_gate,
        λₛ = λₛ,
        λᵧ = λᵧ,

        padded_vocab_size = padded_vocab_size,
        nparams = i - 1,
    )
end

Linear(Θ::AbstractVector, spec::ParamSpec{2}) = Linear(paramview(Θ, spec))
Embedding(Θ::AbstractVector, spec::ParamSpec{2}) = Embedding(paramview(Θ, spec))
function Block(
    Θ::AbstractVector,
    layout,
    layer_idx::Int,
    window::Tuple{Int,Int},
    head_dim::Int,
    n_head::Int,
    n_kv_head::Int,
)
    👀 = layout.👀

    𝕧𝕖 = isnothing(👀.𝕧𝕖) ?
        nothing :
        Linear(Θ, 👀.𝕧𝕖)

    attention = CausalSelfAttention(
        layer_idx,
        Linear(Θ, 👀.𝕎),
        Linear(Θ, 👀.𝕂),
        Linear(Θ, 👀.𝕍),
        Linear(Θ, 👀.ℙ),
        𝕧𝕖,
        window,
        head_dim,
        n_head,
        n_kv_head
    )

    🧠 = layout.🧠

    mlp = MLP(
        Linear(Θ, 🧠.𝔽),
        Linear(Θ, 🧠.ℙ),
    )

    🍰 = isnothing(layout.🍰.𝔼) ?
        nothing :
        Embedding(Θ, layout.🍰.𝔼)

    λᵦ = paramview(Θ, layout.λᵦ)
    λx₀ = paramview(Θ, layout.λx₀)

    Block(attention, mlp, 🍰, λᵦ, λx₀)
end

function Transformer(
    Θ::AbstractVector,
    layout,
    config::GPTConfig,
)
    embedding = Embedding(
        Θ,
        layout.embedding,
    )

    windows = window_sizes(config)

    blocks = [
        Block(Θ, block_layout, layer_idx, windows[layer_idx], config.n_embed ÷ config.n_head, config.n_head, config.n_kv_head)
        for (layer_idx, block_layout) in enumerate(layout.blocks)
    ]

    Transformer(
        embedding,
        blocks,
    )
end

function rotary_embeddings(
    ::Type{A},
    seq_len::Int,
    head_dim::Int,
    base::AbstractFloat = 100_000f0,
) where {A<:AbstractArray}

    channel_range = A(undef, head_dim ÷ 2)
    t = A(undef, seq_len)

    # Mathematically these start at zero
    # We'll set up basic ranges and subtract 1 from them to get the correct values
    channel_range .= (1:2:head_dim) .- 1
    t .= (1:seq_len) .- 1

    inv_freq = @. 1f0 / base^(channel_range / head_dim)
    freqs = inv_freq * t'

    return sin.(freqs), cos.(freqs)
end

"""
Construct a GPT model whose trainable tensors are zero-copy views into
`params.Θ`.

`layout` must have been generated from the same GPTConfig.
"""
function 🤖(
    params::Params,
    config::GPTConfig,
    layout,
)
    length(params.Θ) == layout.nparams ||
        throw(DimensionMismatch(
            "parameter vector has $(length(params.Θ)) elements, " *
            "but model layout requires $(layout.nparams)"
        ))

    Θ = params.Θ

    transformer = Transformer(
        Θ,
        layout.transformer,
        config
    )

    lm_head = Linear(
        Θ,
        layout.lm_head,
    )

    smear_gate = Linear(
        Θ,
        layout.smear_gate,
    )

    λₛ = paramview(
        Θ,
        layout.λₛ,
    )

    λᵧ = paramview(
        Θ,
        layout.λᵧ,
    )

    rope_sin_cos = rotary_embeddings(
        typeof(Θ),
        10 * config.sequence_len,
        config.n_embed ÷ config.n_head,
    )

    🤖(
        config,
        window_sizes(config),
        layout.padded_vocab_size,

        transformer,
        lm_head,

        smear_gate,
        λₛ,
        λᵧ,

        10config.sequence_len,
        rope_sin_cos
    )
end

function 🤖🤝🤖(
    params::Params,
    config::GPTConfig,
    layout,
)
    # TODO: Implement Mixture of Experts
end

# -----------------------------------------------------------------------------
# Display
# -----------------------------------------------------------------------------

function _bytestring(n::Integer)
    n < 2^10 && return "$(n) B"
    n < 2^20 && return "$(round(n / 2^10; digits=1)) KiB"
    n < 2^30 && return "$(round(n / 2^20; digits=1)) MiB"
    return "$(round(n / 2^30; digits=2)) GiB"
end


# -----------------------------------------------------------------------------
# Params
# -----------------------------------------------------------------------------

function Base.show(io::IO, p::Params)
    print(
        io,
        "Params(",
        length(p.Θ),
        " parameters, ",
        eltype(p.Θ),
        " → ",
        eltype(p.δ),
        ")",
    )
end


function Base.show(io::IO, ::MIME"text/plain", p::Params)
    n = length(p.Θ)

    θbytes = n * sizeof(eltype(p.Θ))
    δbytes = n * sizeof(eltype(p.δ))

    println(io, "Params")
    println(io, "  parameters:  ", n)
    println(io, "  Θ:           ", typeof(p.Θ))
    println(io, "  δ:           ", typeof(p.δ))
    println(io, "  Θ storage:   ", _bytestring(θbytes))
    println(io, "  δ storage:   ", _bytestring(δbytes))
    print(  io, "  total:       ", _bytestring(θbytes + δbytes))
end


# -----------------------------------------------------------------------------
# GPT
# -----------------------------------------------------------------------------

function Base.show(io::IO, m::🤖)
    c = m.config

    print(
        io,
        "🤖(",
        c.n_layer, "L, ",
        c.n_embed, "d, ",
        c.n_head, "H, ",
        c.n_kv_head, "KV, ",
        "seq=", c.sequence_len,
        ")",
    )
end


function Base.show(io::IO, ::MIME"text/plain", m::🤖)
    c = m.config

    println(io, "🤖 GPT")
    println(io, "  layers:          ", c.n_layer)
    println(io, "  embedding:       ", c.n_embed)
    println(io, "  attention heads: ", c.n_head)
    println(io, "  KV heads:        ", c.n_kv_head)
    println(io, "  sequence length: ", c.sequence_len)
    println(io, "  vocabulary:      ", c.vocab_size,
                " (", m.padded_vocab_size, " padded)")
    println(io, "  window pattern:  ", c.window_pattern)
    println(io, "  window sizes:    ", m.window_sizes)
    println(io, "  rotary length:   ", m.rotary_seq_len)
    print(  io, "  parameter type:  ", eltype(m.transformer.embed.𝔼))
end

end
