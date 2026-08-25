
module GPT
using ..Parameters

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