
module GPT
using ..Parameters

function norm(x)
    """TODO: Build RMSnorm"""
end

(□::Linear)(x::AbstractMatrix) = □.𝕎 * x
(𝔼::Embedding)(token_ids::AbstractVector{Int}) = 𝔼.𝔼[:, token_ids]
function (m::MLP)(x::AbstractMatrix)
    x = m.𝔽(x)
    x = max.(x, zero(eltype(x))) .^ 2
    m.ℙ(x)
end

function(👀::CausalSelfAttention)(x::AbstractMatrix)
    # TODO: Implement causal self-attention with value embedding and rotary embedding and kb cache
    Q = 👀.𝕎(x)
    K = 👀.𝕂(x)
    V = 👀.𝕍(x)
    y = attention(Q, K, V, window)
    👀.ℙ(y)
end

function attention(Q::Matrix, K::Matrix, V::Matrix, window)
    # TODO: reference implementationx
    x
end

function attention(Q::CuMatrix, K::CuMatrix, V::CuMatrix, window)
    # TODO: FA2 or FA3
end

function (𝔹::Block)(x::AbstractMatrix)
    x = x + 𝔹.👀(norm(x))
    x = x + 𝔹.🧠(norm(x))
end


end