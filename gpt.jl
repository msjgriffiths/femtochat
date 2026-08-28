
module GPT
using Random
using ..Parameters
using ..Kernels

function sigmoid(x)
    @. 1 / (1 + exp(-x))
end
σ = sigmoid

Σ = sum

function norm(x, ϵ=eps(Float32))
    D = size(x, 1) # Since we're column orientated, first dimension is token embedding size
    x ./ sqrt.(Σ(abs2, x; dims=1) ./ D .+ ϵ )
end
 ∥(x) = norm(x)

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

function (m::MLP)(x::AbstractArray)
    x = m.𝔽(x)
    x = max.(x, zero(eltype(x))) .^ 2
    m.ℙ(x)
end

function apply_rotary_embedding(X, cos, sin)
    head_dim, n_head, T, B = size(X)
    d = head_dim ÷ 2

    x₁ = @view X[1:d, :, :, :]
    x₂ = @view X[d+1:end, :, :, :]

    c = reshape(@view(cos[:, 1:T]), d, 1, T, 1)
    s = reshape(@view(sin[:, 1:T]), d, 1, T, 1)

    Y = similar(X)

    y₁ = @view Y[1:d, :, :, :]
    y₂ = @view Y[d+1:end, :, :, :]

    @. y₁ = x₁ * c + x₂ * s
    @. y₂ = x₁ * -s + x₂ * c

    Y
end


function(👀::CausalSelfAttention)(x::AbstractArray{<:Any,3}, sin_cos, ve::Union{Nothing,AbstractArray} = nothing)
    # TODO: Implement causal self-attention with value embedding and rotary embedding and kb cache

    (; 𝕎, 𝕂, 𝕍, ℙ, 𝕧𝕖, head_dim, n_head, n_kv_head, window) = 👀
    C, T, B = size(x)
    
    Q = 𝕎(x) |> q -> reshape(q, head_dim, n_head,    T, B)
    K = 𝕂(x) |> k -> reshape(k, head_dim, n_kv_head, T, B)
    V = 𝕍(x) |> v -> reshape(v, head_dim, n_kv_head, T, B)
    
    # Value Residual Learning (ResFormer) from https://arxiv.org/abs/2410.17897
    # Not to be confused with ResFormer from ViTs
    # This is really nothing like ResFormer, because it adds another token embedding - 
    # ... where each layer gets its own embedding matrix.
    # ... as opposed to a skip connection from an earlier (first) layer (ResFormer)
    if !isnothing(𝕧𝕖)
        VE = reshape(ve, head_dim, n_kv_head, T, B) # Match shape of V above 
        gate = 3sigmoid(𝕧𝕖(x[1:12, :, :])) # Range (0, 3)
        V .+= reshape(gate, 1, n_kv_head, T, B) .* VE # Residual connection
    end

    # Rotary Embedding (RoPE) from https://arxiv.org/abs/2104.09864
    rope_sin, rope_cos = sin_cos
    Q = apply_rotary_embedding(Q, rope_cos, rope_sin)
    K = apply_rotary_embedding(K, rope_cos, rope_sin)
    Q = norm(Q) .* 1.2f0
    K = norm(K) .* 1.2f0

    y = attention(Q, K, V, window)
    y = reshape(y, C, T, B)
    ℙ(y)
end

function (𝔹::Block)(x::AbstractArray, ve::Union{Nothing,AbstractArray} = nothing, sin_cos::Union{Nothing,Tuple{AbstractMatrix,AbstractMatrix}} = nothing)
    x .+= 𝔹.👀(norm(x), sin_cos, ve) # Residual highway
    x .+= 𝔹.🧠(norm(x)) # Residual highway
end

function (ω::🤖)(tokens::Union{AbstractVector,AbstractMatrix})
    (; λᵧ, λₛ) = ω
    (; n_layer, vocab_size) = ω.config
    x = ω.transformer.embed(tokens)
    x = norm(x)

    # Smear token embeddings together for cheap bigram information
    gate = λₛ .* σ(ω.smear_gate(@view x[1:24, 2:end, :]))
    xₛ = copy(@view x[:, 1:end-1, :]) # Create a copy to read from to avoid race condition updating values
    @views x[:, 2:end, :] .+=  gate .* xₛ

    x₀ = x

    x_backout = nothing
    backout_layer = n_layer ÷ 2 - 1
    for (i, block) in enumerate(ω.transformer.blocks)
        (; 🍰, λᵦ, λx₀) = block
        x = @. λᵦ * x + λx₀ * x₀ # X is linear interpolation between the original x and the current x
        # Get value embedding matrix from this block given tokens
        # We pull out embedding matrix here because we have tokens here, instead of 
        # passing token indexes down. 
        ve = isnothing(🍰) ? nothing : 🍰(tokens)
        x = block(x, ve, ω.rope_sin_cos)
        if i == backout_layer
            x_backout = x
        end
    end

    if !isnothing(x_backout)
        # Subtract mid-layer residual to remove low-level features before logit projection
        x = @. x - λᵧ * x_backout
    end

    x = norm(x)

    softcap = 15f0
    logits = ω.lm_head(x)[:1:vocab_size, :, :]
    @. logits = softcap * tanh(logits / softcap)
    logits
end

function (ω::🤖)(tokens::Union{AbstractVector,AbstractMatrix}, targets::AbstractVector)
    logits = ω(tokens)
    # TODO: Apply cross entropy here
end

function uniform!(ℛ, Θ::AbstractVector, spec::ParamSpec, low, high)
    paramview(Θ, spec) .=
        low .+ (high - low) .* rand(ℛ, Float32, spec.shape)

    return nothing
end

function initialize!(params::Params, layout, ℛ = Random.default_rng())
    (; Θ) = params
    (; blocks, embedding) = layout.transformer
    (; lm_head, smear_gate, λₛ, λᵧ) = layout
    
    D = embedding.shape[1]
    n_layer = length(blocks)
    s = √(3f0 / D) # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal

    paramview(Θ, embedding) .= .8f0 .* randn(ℛ, Float32, embedding.shape)
    paramview(Θ, lm_head) .= 0.001f0 .* randn(ℛ, Float32, layout.lm_head.shape)
    for (i, block) in enumerate(blocks)
        (; 👀, 🍰, λᵦ, λx₀) = block
        (; 𝕎, 𝕂, 𝕍, ℙ, 𝕧𝕖) = 👀
        uniform!(ℛ, Θ, 𝕎, -s, s)
        uniform!(ℛ, Θ, 𝕂, -s, s)
        uniform!(ℛ, Θ, 𝕍, -s, s)
        paramview(Θ, ℙ) .= 0f0 # Projection starts at zero
        
        # Value embedding and its gate exist on alternating layers.
        if !isnothing(🍰.𝔼)
            uniform!(ℛ, Θ, 🍰.𝔼, -s, s)
            uniform!(ℛ, Θ, 𝕧𝕖, 0f0, 0.02f0)
        end

        (; 𝔽, ℙ) = block.🧠
        uniform!(ℛ, Θ, 𝔽, -.4s, .4s)
        paramview(Θ, ℙ) .= 0f0 # Projection starts at zero

        depth = Float32(i - 1) / max(n_layer - 1, 1)
        paramview(Θ, λᵦ) .= 1.15f0 - 0.10f0depth
        paramview(Θ, λx₀) .=  0.20f0 - 0.15f0depth
    end
    # Backout lambda + smear lamda/gate
    paramview(Θ, λᵧ) .= 0.2f0
    paramview(Θ, λₛ) .= 0f0
    uniform!(ℛ, Θ, smear_gate, 0f0, 0.02f0)
end

end
