module Optimizer

using LinearAlgebra: norm, mul!
using ..Parameters: Params, ParamSpec, paramview

export AdamW, Muon, MuonAdamW, polar_express

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

# From nanochat code, we get (a, b, c) for each step
const ρₜ = (
    (8.156554524902461f0,  -22.48329292557795f0,  15.878769915207462f0),
    (4.042929935166739f0,   -2.808917465908714f0,  0.5000178451051316f0),
    (3.8916678022926607f0,  -2.772484153217685f0,  0.5060648178503393f0),
    (3.285753657755655f0,   -2.3681294933425376f0,  0.46449024233003106f0),
    (2.3465413258596377f0,  -1.7097828382687081f0,  0.42323551169305323f0),
)

"""
Source: https://arxiv.org/abs/2505.16932
Apply ρₜ(σ) = aₜσ + bₜσ³ + cₜσ⁵ to the singular values of `G`,
using the smaller Gram matrix for efficiency.
def PolarExpress(G:torch.Tensor,steps:int)->torch.Tensor:
    X = G.bfloat16() #forspeed
    i fG.size(-2) > G.size(-1): X=X.mT # thisreducesFLOPs
    X = X / (X.norm(dim=(-2,-1),keepdim=True) * 1.01+1e-7)
    hs = coeffs_list[:steps] +list(
         repeat(coeffs_list[-1],steps-len(coeffs_list)))
    for a,b,c in hs:
        A = X @ X.mT
        B = b * A+c * A @ A
        X = a * X+B @ X # X<-aX+bXˆ3+cXˆ5
    ifG.size(-2)>G.size(-1):X=X.mT
    return X
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

mutable struct Muon{F<:AbstractFloat,M<:AbstractMatrix}
    α::F
    μ::F
    β₂::F
    λ::F
    steps::Int
    𝓂ₜ::M
    𝓋ₜ::M
end

function Muon(
    θ::AbstractMatrix;
    α=0.02f0,
    μ=0.95f0,
    β₂=0.9f0,
    λ=0.01f0,
    steps=5,
)
    m, n = size(θ)

    𝓂ₜ = similar(θ)
    𝓋ₜ = similar(θ, m ≥ n ? (m, 1) : (1, n))
    fill!(𝓂ₜ, 0f0)
    fill!(𝓋ₜ, 0f0)

    return Muon(α, μ, β₂, λ, steps, 𝓂ₜ, 𝓋ₜ)
end

function (ω::Muon)(θ, gₜ)
    (; α, μ, β₂, λ, steps, 𝓂ₜ, 𝓋ₜ) = ω
    m, n = size(θ)

    # Nesterov momentum
    @. 𝓂ₜ = μ * 𝓂ₜ + (1f0 - μ) * gₜ
    X = @. (1f0 - μ) * gₜ + μ * 𝓂ₜ

    # MuonEq row equilibration
    target = norm(X) / √Float32(m)
    row_norm = sqrt.(sum(abs2, X; dims=2))
    @. X *= target / max(row_norm, 1f-6)

    # Polar Express
    X = polar_express(X; steps)

    # Muon+ renormalization
    X .*= √Float32(min(m, n)) / max(norm(X), 1f-6)

    # NorMuon variance reduction
    dimension = m ≥ n ? 2 : 1
    dimension_size = size(X, dimension)
    v_mean = sum(abs2, X; dims=dimension) ./ dimension_size

    @. 𝓋ₜ = β₂ * 𝓋ₜ + (1f0 - β₂) * v_mean
    step_size = @. inv(√max(𝓋ₜ, 1f-10))

    old_norm = norm(X)
    new_norm = √sum(dimension_size .* v_mean .* step_size.^2)
    @. X *= step_size * old_norm / max(new_norm, 1f-10)

    # Shape-adjusted learning rate and cautious decay
    η = α * √max(1f0, Float32(m) / Float32(n))
    @. θ -= η * X + η * λ * θ * ((X * θ) ≥ 0)

    return nothing
end

struct ParameterUpdate{O,P,G}
    ω::O
    θ::P
    δ::G
end

(update::ParameterUpdate)() = update.ω(update.θ, update.δ)

function adamw_update(params::Params, spec; kwargs...)
    θ = vec(paramview(params.Θ, spec))
    δ = vec(paramview(params.δ, spec))
    ParameterUpdate(AdamW(θ; kwargs...), θ, δ)
end

function muon_update(params::Params, spec::ParamSpec{2}; kwargs...)
    θ = paramview(params.Θ, spec)
    δ = paramview(params.δ, spec)
    ParameterUpdate(Muon(δ; kwargs...), θ, δ)
end

struct MuonAdamW{A,M}
    adamw::A
    muon::M
end

"""
Build nanochat's optimizer groups as zero-copy views into `params`.

Transformer-block matrices use Muon. Embeddings, the language-model head,
residual scalars, and smear parameters use their corresponding AdamW settings.
"""
function MuonAdamW(
    params::Params,
    layout;
    unembedding_lr=0.004f0,
    embedding_lr=0.2f0,
    matrix_lr=0.02f0,
    weight_decay=0f0,
    scalar_lr=0.5f0,
)
    D = layout.transformer.embedding.shape[1]
    scale = √(768f0 / D)

    adamw = [adamw_update(
        params,
        layout.lm_head;
        α=Float32(unembedding_lr) * scale,
        β₁=0.8f0,
        β₂=0.96f0,
        ϵ=1f-10,
        λ=0.01f0,
    )]

    push!(adamw, adamw_update(
        params,
        layout.transformer.embedding;
        α=Float32(embedding_lr) * scale,
        β₁=0.8f0,
        β₂=0.995f0,
        ϵ=1f-10,
        λ=0.001f0,
    ))

    for block in layout.transformer.blocks
        if !isnothing(block.🍰.𝔼)
            push!(adamw, adamw_update(
                params,
                block.🍰.𝔼;
                α=0.5f0 * Float32(embedding_lr) * scale,
                β₁=0.8f0,
                β₂=0.995f0,
                ϵ=1f-10,
                λ=0.01f0,
            ))
        end

        push!(adamw, adamw_update(
            params,
            block.λᵦ;
            α=0.01f0 * Float32(scalar_lr),
            β₁=0.8f0,
            β₂=0.95f0,
            ϵ=1f-10,
            λ=0.05f0,
        ))

        push!(adamw, adamw_update(
            params,
            block.λx₀;
            α=Float32(scalar_lr),
            β₁=0.96f0,
            β₂=0.95f0,
            ϵ=1f-10,
            λ=0f0,
        ))
    end

    for spec in (layout.smear_gate, layout.λₛ, layout.λᵧ)
        push!(adamw, adamw_update(
            params,
            spec;
            α=0.2f0,
            β₁=0.8f0,
            β₂=0.95f0,
            ϵ=1f-10,
            λ=0f0,
        ))
    end

    muon_specs = ParamSpec{2}[]
    for block in layout.transformer.blocks
        (; 👀, 🧠) = block
        append!(muon_specs, (👀.𝕎, 👀.𝕂, 👀.𝕍, 👀.ℙ, 🧠.𝔽, 🧠.ℙ))
        isnothing(👀.𝕧𝕖) || push!(muon_specs, 👀.𝕧𝕖)
    end

    muon = [
        muon_update(
            params,
            spec;
            α=Float32(matrix_lr),
            μ=0.95f0,
            β₂=0.9f0,
            λ=Float32(weight_decay),
            steps=5,
        )
        for spec in muon_specs
    ]

    return MuonAdamW(adamw, muon)
end

function (ω::MuonAdamW)()
    for update in ω.adamw
        update()
    end
    for update in ω.muon
        update()
    end
    return nothing
end


end
