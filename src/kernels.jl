module Kernels

using LinearAlgebra: mul!

export attention, softmax!

function attention_mask!(mask, keys, queries, window)
    left, right = window
    key = reshape(keys, :, 1)
    query = reshape(queries, 1, :)

    if window == (-1, 0)
        @. mask = key <= query
    else
        @. mask = (query - left <= key) & (key <= query + right)
    end

    return mask
end

function attention_mask(X, keys, queries, window)
    mask = similar(X, Bool, length(keys), length(queries))
    attention_mask!(mask, keys, queries, window)
end

attention_mask(X, T::Int, window) =
    attention_mask(X, 1:T, 1:T, window)

function softmax!(X; dims=1)
    X .-= maximum(X; dims)
    X .= exp.(X)
    X ./= sum(X; dims)
    X
end

function attention(Q::AbstractArray{F,4}, K::AbstractArray{F,4}, V::AbstractArray{F,4}, window) where F
    D, H, T, B = size(Q)
    mask = attention_mask(Q, T, window)
    n_kv_head = size(K, 2)
    heads_per_kv = H ÷ n_kv_head
    
    # Output to store result in
    # key × query × head × batch
    S = similar(Q, T, T, H, B)
    for document in 1:B, head in 1:H
        kv_head = cld(head, heads_per_kv)
        # For each document in document
        # For each head in H
        @views mul!(
            S[:, :, head, document], # Place the result in S, one for each head and document
            K[:, kv_head, :, document]', # Transpose K for the dot product
            Q[:, head, :, document] # Q for the dot product
        )
    end
    S ./= sqrt(F(D)) # Scale by sqrt of dimension

    S .= ifelse.(mask, S, typemin(F)) # Apply mask
    softmax!(S; dims=1) # Softmax along the key dimension

    Y = similar(Q) # Create output matrix same shape as Q
    for document in 1:B, head in 1:H
        kv_head = cld(head, heads_per_kv)
        # For each document in document
        # For each head in H
        @views mul!(
            Y[:, head, :, document], # Place the result in Y, one for each head and document
            V[:, kv_head, :, document], # Value matrix is D x head x token x document
            S[:, :, head, document] # S is token x token x head x document
        )
    end

    Y
end

"""
Source: https://arxiv.org/abs/2205.14135
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
𝐀𝐥𝐠𝐨𝐫𝐢𝐭𝐡𝐦 𝟏: 𝐅𝐥𝐚𝐬𝐡𝐀𝐭𝐭𝐞𝐧𝐭𝐢𝐨𝐧
# 𝐑𝐞𝐪𝐮𝐢𝐫𝐞: Matrices 𝐐, 𝐊, 𝐕 ∈ ℝᴺˣᵈ in HBM, on-chip SRAM of size M.
1: Set block sizes Bᶜ = ⌈M / 4d⌉, Bᵣ = min(⌈M / 4d⌉, d).
2: Initialize 𝐎 = (0)ᴺˣᵈ ∈ ℝᴺˣᵈ, ℓ = (0)ᴺ ∈ ℝᴺ,
   m = (−∞)ᴺ ∈ ℝᴺ in HBM.
3: 𝐃𝐢𝐯𝐢𝐝𝐞 𝐐 into Tᵣ = ⌈N / Bᵣ⌉ blocks 𝐐₁, …, 𝐐_Tᵣ of size Bᵣ × d each,
   and 𝐝𝐢𝐯𝐢𝐝𝐞 𝐊, 𝐕 into Tᶜ = ⌈N / Bᶜ⌉ blocks
   𝐊₁, …, 𝐊_Tᶜ and 𝐕₁, …, 𝐕_Tᶜ, of size Bᶜ × d each.
4: 𝐃𝐢𝐯𝐢𝐝𝐞 𝐎 into Tᵣ blocks 𝐎ᵢ, …, 𝐎_Tᵣ of size Bᵣ × d each,
   𝐝𝐢𝐯𝐢𝐝𝐞 ℓ into Tᵣ blocks ℓᵢ, …, ℓ_Tᵣ of size Bᵣ each,
   𝐝𝐢𝐯𝐢𝐝𝐞 m into Tᵣ blocks mᵢ, …, m_Tᵣ of size Bᵣ each.
5: 𝐟𝐨𝐫 1 ≤ j ≤ Tᶜ 𝐝𝐨
6:     Load 𝐊ⱼ, 𝐕ⱼ from HBM to on-chip SRAM.
7:     𝐟𝐨𝐫 1 ≤ i ≤ Tᵣ 𝐝𝐨
8:         Load 𝐐ᵢ, 𝐎ᵢ, ℓᵢ, mᵢ from HBM to on-chip SRAM.
9:         On chip, compute 𝐒ᵢⱼ = 𝐐ᵢ𝐊ⱼᵀ ∈ ℝᴮʳˣᴮᶜ.
10:        On chip, compute m̃ᵢⱼ = rowmax(𝐒ᵢⱼ) ∈ ℝᴮʳ,
            𝐏̃ᵢⱼ = exp(𝐒ᵢⱼ − m̃ᵢⱼ) ∈ ℝᴮʳˣᴮᶜ pointwise,
            ℓ̃ᵢⱼ = rowsum(𝐏̃ᵢⱼ) ∈ ℝᴮʳ.
11:        On chip, compute mᵢⁿᵉʷ = max(mᵢ, m̃ᵢⱼ) ∈ ℝᴮʳ,
            ℓᵢⁿᵉʷ = eᵐⁱ⁻ᵐⁱⁿᵉʷℓᵢ + eᵐ̃ⁱⱼ⁻ᵐⁱⁿᵉʷℓ̃ᵢⱼ ∈ ℝᴮʳ.
12:        Write 𝐎ᵢ ← diag(ℓᵢⁿᵉʷ)⁻¹
            (diag(ℓᵢ)eᵐⁱ⁻ᵐⁱⁿᵉʷ𝐎ᵢ
            + eᵐ̃ⁱⱼ⁻ᵐⁱⁿᵉʷ𝐏̃ᵢⱼ𝐕ⱼ) to HBM.
13:        Write ℓᵢ ← ℓᵢⁿᵉʷ, mᵢ ← mᵢⁿᵉʷ to HBM.
14:    𝐞𝐧𝐝 𝐟𝐨𝐫
15: 𝐞𝐧𝐝 𝐟𝐨𝐫
16: Return 𝐎.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
function flash_attention₁(Q::AbstractArray{F,4}, K::AbstractArray{F,4}, V::AbstractArray{F,4}, window) where F
    D, H, T, B = size(Q)
    Bᶜ, Bᵣ = min(T, 512), min(T, 512)

    n_kv_head = size(K, 2)
    heads_per_kv = H ÷ n_kv_head

    𝕆 = similar(Q); ℓ = similar(Q, 1, T, H, B); m = similar(Q, 1, T, H, B)
    fill!(𝕆, 0f0); fill!(ℓ, 0f0); fill!(m, typemin(F))

    Tᵣ, Tᶜ = cld(T, Bᵣ), cld(T, Bᶜ)
    scale = inv(sqrt(F(D)))
    masked = -floatmax(F)

    𝕆 = fill!(similar(Q), zero(F))
    ℓ = fill!(similar(Q, 1, T, H, B), zero(F))
    m = fill!(similar(Q, 1, T, H, B), typemin(F))

    # Scratch space for intermediate computations
    # These are used to avoid allocating new arrays in the inner loop
    S = similar(Q, Bᶜ, Bᵣ)
    M = similar(Q, Bool, Bᶜ, Bᵣ)
    𝕆̃ = similar(Q, D, Bᵣ)
    m̃, ℓ̃, mⁿᵉʷ, eᵐ, eᵐ̃, ℓⁿᵉʷ =
        ntuple(_ -> similar(Q, 1, Bᵣ), 6)

    for document in 1:B, head in 1:H
        kv_head = cld(head, heads_per_kv)

        for j in 1:Tᶜ
            blockⱼ = (j - 1) * Bᶜ + 1:min(j * Bᶜ, T)
            Kⱼ = @view K[:, kv_head, blockⱼ, document]
            Vⱼ = @view V[:, kv_head, blockⱼ, document]

            for i in 1:Tᵣ
                blockᵢ = (i - 1) * Bᵣ + 1:min(i * Bᵣ, T)
                nᵢ, nⱼ = length(blockᵢ), length(blockⱼ)
                # We reference the views of the matrices in the inner loop to
                # avoid allocating new arrays for each block. 
                # This is important for performance
                @views begin
                    Qᵢ = Q[:, head, blockᵢ, document]
                    𝕆ᵢ = 𝕆[:, head, blockᵢ, document]
                    mᵢ = m[:, blockᵢ, head, document]
                    ℓᵢ = ℓ[:, blockᵢ, head, document]

                    Sᵢⱼ = S[1:nⱼ, 1:nᵢ]
                    Mᵢⱼ = M[1:nⱼ, 1:nᵢ]
                    𝕆̃ᵢⱼ = 𝕆̃[:, 1:nᵢ]
                    m̃ᵢⱼ = m̃[:, 1:nᵢ]
                    ℓ̃ᵢⱼ = ℓ̃[:, 1:nᵢ]
                    mᵢⁿᵉʷ = mⁿᵉʷ[:, 1:nᵢ]
                    eᵐᵢⱼ = eᵐ[:, 1:nᵢ]
                    eᵐ̃ᵢⱼ = eᵐ̃[:, 1:nᵢ]
                    ℓᵢⁿᵉʷ = ℓⁿᵉʷ[:, 1:nᵢ]
                end

                # One downside of our implementation is that each
                # mul! is a single CUDA kernel launch, which can be slower
                # than a single fused kernel
                # This runs on CPU and GPU though
                mul!(Sᵢⱼ, Kⱼ', Qᵢ)
                @. Sᵢⱼ *= scale

                attention_mask!(Mᵢⱼ, blockⱼ, blockᵢ, window)
                @. Sᵢⱼ = ifelse(Mᵢⱼ, Sᵢⱼ, masked)

                maximum!(m̃ᵢⱼ, Sᵢⱼ)
                @. Sᵢⱼ = ifelse(Mᵢⱼ, exp(Sᵢⱼ - m̃ᵢⱼ), zero(F))
                P̃ᵢⱼ = Sᵢⱼ
                sum!(ℓ̃ᵢⱼ, P̃ᵢⱼ)

                @. mᵢⁿᵉʷ = max(mᵢ, m̃ᵢⱼ)
                @. eᵐᵢⱼ = exp(mᵢ - mᵢⁿᵉʷ)
                @. eᵐ̃ᵢⱼ = exp(m̃ᵢⱼ - mᵢⁿᵉʷ)
                @. ℓᵢⁿᵉʷ = eᵐᵢⱼ * ℓᵢ + eᵐ̃ᵢⱼ * ℓ̃ᵢⱼ

                mul!(𝕆̃ᵢⱼ, Vⱼ, P̃ᵢⱼ)
                @. 𝕆ᵢ = (
                    𝕆ᵢ * (ℓᵢ * eᵐᵢⱼ) +
                    𝕆̃ᵢⱼ * eᵐ̃ᵢⱼ
                ) / max(ℓᵢⁿᵉʷ, eps(F))

                mᵢ .= mᵢⁿᵉʷ
                ℓᵢ .= ℓᵢⁿᵉʷ
            end
        end
    end

    return 𝕆
end

function Δflash_attention₁()

end

# function attention(Q::CuMatrix, K::CuMatrix, V::CuMatrix, window)
#     # TODO: FA2 or FA3
# end

end
