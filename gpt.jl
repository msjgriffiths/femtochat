
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

function(👀::CausalSelfAttention)(x::AbstractArray{<:Any,3}, ve::Union{Nothing,AbstractMatrix} = nothing)
    # TODO: Implement causal self-attention with value embedding and rotary embedding and kb cache

    (; 𝕎, 𝕂, 𝕍, ℙ, 𝕧𝕖, head_dim, n_head, n_kv_head, window) = 👀
    C, T, B = size(x)
    
    Q = 𝕎(x) |> q -> reshape(q, head_dim, n_head, T, B)
    K = 𝕂(x) |> k -> reshape(k, head_dim, n_kv_head, T, B)
    V = 𝕍(x) |> v -> reshape(v, head_dim, n_kv_head, T, B)
    if !isnothing(𝕧𝕖)
        VE = reshape(ve, head_dim, n_kv_head, T, B) # Match shape of V above 
        gate = 3sigmoid(𝕧𝕖(x[1:12, :, :]))
        V .+= reshape(gate, 1, n_kv_head, T, B) .* VE # Residual connection
    end
    y = attention(Q, K, V, window)
    ℙ(y)
end

function (𝔹::Block)(x::AbstractMatrix, ve::Union{Nothing,AbstractMatrix} = nothing)
    x .+= 𝔹.👀(norm(x), ve) # Residual highway
    x .+= 𝔹.🧠(norm(x)) # Residual highway
end

function (ω::🤖)(tokens::Union{AbstractVector,AbstractMatrix})
    x = ω.transformer.embed(tokens)

    for (i, block) in enumerate(ω.transformer.blocks)
        x = block(x, ω.window_sizes[i])
    end

    ω.lm_head(x)
end

end