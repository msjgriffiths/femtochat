
module GPT
using ..Parameters

function sigmoid(x)
    @. 1 / (1 + exp(-x))
end

function norm(x)
    """TODO: Build RMSnorm"""
end

function (ℓ::Linear)(x::AbstractArray)
    Din = size(x, 1)

    X = reshape(x, Din, :)
    Y = ℓ.𝕎 * X

    reshape(Y, size(ℓ.𝕎, 1), Base.tail(size(x))...)
end

(𝔼::Embedding)(token_ids::AbstractVector{<:Integer}) = 𝔼.𝔼[:, token_ids]
function (e::Embedding)(tokens::AbstractMatrix{<:Integer})
    D = size(e.𝔼, 1)
    T, B = size(tokens)

    reshape(e.𝔼[:, vec(tokens)], D, T, B)
end

function (m::MLP)(x::AbstractMatrix)
    x = m.𝔽(x)
    x = max.(x, zero(eltype(x))) .^ 2
    m.ℙ(x)
end

function apply_rotary_embedding(X, cos, sin)
    head_dim, n_head, T, B = size(X)

    # Expose adjacent channel pairs
    X₂ = reshape(X, 2, head_dim ÷ 2, n_head, T, B)

    x₁ = @view X₂[1, :, :, :, :]
    x₂ = @view X₂[2, :, :, :, :]

    # Output buffer
    Y₂ = similar(X₂)

    y₁ = @view Y₂[1, :, :, :, :]
    y₂ = @view Y₂[2, :, :, :, :]

    # RoPE lookup for positions in this sequence.
    # Shape: (head_dim/2) × 1 × T × 1
    c = reshape(@view(cos[:, 1:T]), head_dim ÷ 2, 1, T, 1)
    s = reshape(@view(sin[:, 1:T]), head_dim ÷ 2, 1, T, 1)

    # Rotate each adjacent channel pair
    @. y₁ = x₁ * c - x₂ * s
    @. y₂ = x₁ * s + x₂ * c

    reshape(Y₂, size(X)...)
end


function(👀::CausalSelfAttention)(x::AbstractArray{<:Any,3}, sin_cos, ve::Union{Nothing,AbstractMatrix} = nothing)
    # TODO: Implement causal self-attention with value embedding and rotary embedding and kb cache

    (; 𝕎, 𝕂, 𝕍, ℙ, 𝕧𝕖, head_dim, n_head, n_kv_head, window) = 👀
    C, T, B = size(x)
    
    Q = 𝕎(x) |> q -> reshape(q, head_dim, n_head, T, B)
    K = 𝕂(x) |> k -> reshape(k, head_dim, n_kv_head, T, B)
    V = 𝕍(x) |> v -> reshape(v, head_dim, n_kv_head, T, B)
    
    # Value Residual Learning (ResFormer) from https://arxiv.org/abs/2410.17897
    # Not to be confused with ResFormer from ViTs
    if !isnothing(𝕧𝕖)
        VE = reshape(ve, head_dim, n_kv_head, T, B) # Match shape of V above 
        gate = 3sigmoid(𝕧𝕖(x[1:12, :, :])) # Range (0, 3)
        V .+= reshape(gate, 1, n_kv_head, T, B) .* VE # Residual connection
    end

    # Rotary Embedding (RoPE) from https://arxiv.org/abs/2104.09864
    rope_sin, rope_cos = sin_cos



    y = attention(Q, K, V, window)
    ℙ(y)
end

function (𝔹::Block)(x::AbstractMatrix, ve::Union{Nothing,AbstractMatrix} = nothing, sin_cos::Union{Nothing,Tuple{AbstractMatrix,AbstractMatrix}} = nothing)
    x .+= 𝔹.👀(norm(x), sin_cos, ve) # Residual highway
    x .+= 𝔹.🧠(norm(x)) # Residual highway
end

function (ω::🤖)(tokens::Union{AbstractVector,AbstractMatrix})
    x = ω.transformer.embed(tokens)
    sin_cos = ω.rope_sin_cos

    for (i, block) in enumerate(ω.transformer.blocks)
        x = block(x, ve, sin_cos)
    end

    ω.lm_head(x)
end

end