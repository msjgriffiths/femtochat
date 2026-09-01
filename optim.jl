module Optimizer

using LinearAlgebra: norm, mul!

mutable struct AdamW{F<:AbstractFloat,V<:AbstractVector}
    α::F
    β₁::F
    β₂::F
    𝓂ₜ::V
    𝓋ₜ::V
    t::Int
    ϵ::F
    λ::F
end

function AdamW(θ::AbstractVector; α=0.001, β₁=0.9, β₂=0.999, ϵ=1e-8, λ=0.01)
    𝓂ₜ = similar(θ, Float32)
    𝓋ₜ = similar(θ, Float32)
    fill!(𝓂ₜ, 0f0)
    fill!(𝓋ₜ, 0f0)

    AdamW(α, β₁, β₂, 𝓂ₜ, 𝓋ₜ, 0, ϵ, λ)
end

"""
Source: https://arxiv.org/abs/1412.6980
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
𝐀𝐥𝐠𝐨𝐫𝐢𝐭𝐡𝐦 𝟏: 𝘈𝘥𝘢𝘮, our proposed algorithm for stochastic optimization. See section 2 for details,
and for a slightly more efficient (but less clear) order of computation. 𝑔ₜ² indicates the elementwise
square 𝑔ₜ ⊙ 𝑔ₜ. Good default settings for the tested machine learning problems are α = 0.001,
β₁ = 0.9, β₂ = 0.999 and ε = 10⁻⁸. All operations on vectors are element-wise. With β₁ᵗ and β₂ᵗ
we denote β₁ and β₂ to the power t.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
𝐑𝐞𝐪𝐮𝐢𝐫𝐞:  α: Stepsize
𝐑𝐞𝐪𝐮𝐢𝐫𝐞:  β₁, β₂ ∈ [0, 1): Exponential decay rates for the moment estimates
𝐑𝐞𝐪𝐮𝐢𝐫𝐞:  𝑓(θ): Stochastic objective function with parameters θ
𝐑𝐞𝐪𝐮𝐢𝐫𝐞:  θ₀: Initial parameter vector

    𝑚₀ ← 0                         (Initialize 1ˢᵗ moment vector)
    𝑣₀ ← 0                         (Initialize 2ⁿᵈ moment vector)
    𝑡 ← 0                          (Initialize timestep)

    𝐰𝐡𝐢𝐥𝐞 θₜ not converged 𝐝𝐨
        𝑡 ← 𝑡 + 1
        𝑔ₜ ← ∇θ 𝑓ₜ(θₜ₋₁)                    (Get gradients w.r.t. stochastic objective at timestep t)
        𝑚ₜ ← β₁ · 𝑚ₜ₋₁ + (1 − β₁) · 𝑔ₜ     (Update biased first moment estimate)
        𝑣ₜ ← β₂ · 𝑣ₜ₋₁ + (1 − β₂) · 𝑔ₜ²     (Update biased second raw moment estimate)
        𝑚̂ₜ ← 𝑚ₜ / (1 − β₁ᵗ)                (Compute bias-corrected first moment estimate)
        𝑣̂ₜ ← 𝑣ₜ / (1 − β₂ᵗ)                 (Compute bias-corrected second raw moment estimate)
        θₜ ← θₜ₋₁ − α · 𝑚̂ₜ / (√𝑣̂ₜ + ε)       (Update parameters)
    𝐞𝐧𝐝 𝐰𝐡𝐢𝐥𝐞

    𝐫𝐞𝐭𝐮𝐫𝐧 θₜ                               (Resulting parameters)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Weight decay: https://arxiv.org/abs/1711.05101

"""
function (ω::AdamW)(θ, gₜ)
    (; α, β₁, β₂, 𝓂ₜ, 𝓋ₜ, ϵ, λ) = ω
    ω.t += 1
    @. 𝓂ₜ = β₁ * 𝓂ₜ + (1f0 - β₁) * gₜ
    @. 𝓋ₜ = β₂ * 𝓋ₜ + (1f0 - β₂) * gₜ^2
    
    # 𝓂̂ₜ = 𝓂ₜ ./ (1f0 - β₁^ω.t) # Avoid allocating
    # 𝓋̂ₜ = 𝓋ₜ ./ (1f0 - β₂^ω.t)  #  intermediate vectors

    θ .*= 1f0 - α * λ # Weight decay

    @. θ -= α * (𝓂ₜ / (1f0 - β₁^ω.t)) /         # Fused operation
               (√(𝓋ₜ / (1f0 - β₂^ω.t)) + ϵ)
    return nothing
end

struct Muon
end

# From nanochat code, we get (a, b, c) for each step
const ρₜ = (
    (8.156554524902461f0,  -22.48329292557795f0,  15.878769915207462f0),
    (4.042929935166739f0,   -2.808917465908714f0,  0.5000178451051316f0),
    (3.8916678022926607f0,  -2.772484153217685f0,  0.5060648178503393f0),
    (3.285753657755655f0,   -2.3681294933425376f0,  0.46449024233003106f0),
    (2.3465413258596377f0,  -1.7097828382687081f0,  0.42323551169305323f0),
)

"""
Apply ρₜ(σ) = aₜσ + bₜσ³ + cₜσ⁵ to the singular values of `G`,
using the smaller Gram matrix for efficiency.
"""
function polar_express(G::AbstractMatrix; steps::Int=5)
    @assert 1 ≤ steps ≤ length(ρₜ)

    transposed = size(G, 1) > size(G, 2)
    scale = 1.01f0 * norm(G) + 1f-6
    X = transposed ? G' ./ scale : G ./ scale

    A = similar(X, size(X, 1), size(X, 1))
    B, Y = similar(A), similar(X)

    for t in 1:steps
        a, b, c = ρₜ[t]

        mul!(A, X, X')       # A = XX'
        mul!(B, A, A)        # B = A²
        @. B = b * A + c * B
        mul!(Y, B, X)
        @. X = a * X + Y
    end

    return transposed ? X' : X
end


end