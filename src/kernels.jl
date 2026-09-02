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

function naive_attention(Q::AbstractArray{F,4}, K::AbstractArray{F,4}, V::AbstractArray{F,4}, window) where F
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
function flash_attention₁!(
    𝕆::AbstractArray{F,4},
    ℓ::AbstractArray{F,4},
    m::AbstractArray{F,4},
    Q::AbstractArray{F,4},
    K::AbstractArray{F,4},
    V::AbstractArray{F,4},
    window,
) where F
    D, H, T, B = size(Q)
    Bᶜ, Bᵣ = min(T, 512), min(T, 512)

    n_kv_head = size(K, 2)
    heads_per_kv = H ÷ n_kv_head

    fill!(𝕆, 0f0); fill!(ℓ, 0f0); fill!(m, typemin(F))

    Tᵣ, Tᶜ = cld(T, Bᵣ), cld(T, Bᶜ)
    scale = inv(sqrt(F(D)))
    masked = -floatmax(F)

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
                # Views to avoid allocations
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

                # One downside of our implementation is that each mul! is a single CUDA kernel launch, so slower than fused kernel
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

function flash_attention₁(
    Q::AbstractArray{F,4},
    K::AbstractArray{F,4},
    V::AbstractArray{F,4},
    window,
) where F
    _, H, T, B = size(Q)
    𝕆 = similar(Q)
    ℓ = similar(Q, 1, T, H, B)
    m = similar(Q, 1, T, H, B)

    return flash_attention₁!(𝕆, ℓ, m, Q, K, V, window)
end

"""
Source: https://arxiv.org/abs/2205.14135
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
𝐀𝐥𝐠𝐨𝐫𝐢𝐭𝐡𝐦 𝟒: FʟᴀsʜAᴛᴛᴇɴᴛɪᴏɴ Backward Pass
𝐑𝐞𝐪𝐮𝐢𝐫𝐞: Matrices 𝐐, 𝐊, 𝐕, 𝐎, d𝐎 ∈ ℝᴺˣᵈ in HBM,
          vectors ℓ, m ∈ ℝᴺ in HBM, on-chip SRAM of size M,
          softmax scaling constant τ ∈ ℝ, masking function MASK,
          dropout probability p_drop, pseudo-random number generator
          state ℛ from the forward pass.
1: Set the pseudo-random number generator state to ℛ.
2: Set block sizes B_c = ⌈M / 4d⌉, Bᵣ = min(⌈M / 4d⌉, d).
3: Divide 𝐐 into Tᵣ = ⌈N / Bᵣ⌉ blocks 𝐐₁, …, 𝐐_Tᵣ of size Bᵣ × d each,
   and divide 𝐊, 𝐕 into T_c = ⌈N / B_c⌉ blocks
   𝐊₁, …, 𝐊_T_c and 𝐕₁, …, 𝐕_T_c, of size B_c × d each.
4: Divide 𝐎 into Tᵣ blocks 𝐎ᵢ, …, 𝐎_Tᵣ of size Bᵣ × d each,
   divide d𝐎 into Tᵣ blocks d𝐎ᵢ, …, d𝐎_Tᵣ of size Bᵣ × d each,
   divide ℓ into Tᵣ blocks ℓᵢ, …, ℓ_Tᵣ of size Bᵣ each,
   divide m into Tᵣ blocks mᵢ, …, m_Tᵣ of size Bᵣ each.
5: Initialize d𝐐 = (0)ᴺˣᵈ in HBM and divide it into Tᵣ blocks
   d𝐐₁, …, d𝐐_Tᵣ of size Bᵣ × d each.
   Initialize d𝐊 = (0)ᴺˣᵈ, d𝐕 = (0)ᴺˣᵈ in HBM and divide d𝐊, d𝐕
   into T_c blocks d𝐊₁, …, d𝐊_T_c and d𝐕₁, …, d𝐕_T_c,
   of size B_c × d each.
6: 𝐟𝐨𝐫 1 ≤ j ≤ T_c 𝐝𝐨
7:     Load 𝐊ⱼ, 𝐕ⱼ from HBM to on-chip SRAM.
8:     Initialize d𝐊̃ⱼ = (0)ᴮᶜˣᵈ, d𝐕̃ⱼ = (0)ᴮᶜˣᵈ on SRAM.
9:     𝐟𝐨𝐫 1 ≤ i ≤ Tᵣ 𝐝𝐨
10:        Load 𝐐ᵢ, 𝐎ᵢ, d𝐎ᵢ, d𝐐ᵢ, ℓᵢ, mᵢ from HBM to on-chip SRAM.
11:        On chip, compute 𝐒ᵢⱼ = τ𝐐ᵢ𝐊ⱼᵀ ∈ ℝᴮʳˣᴮᶜ.
12:        On chip, compute 𝐒ᵢⱼᵐᵃˢᵏᵉᵈ = MASK(𝐒ᵢⱼ).
13:        On chip, compute
           𝐏ᵢⱼ = diag(ℓᵢ)⁻¹ exp(𝐒ᵢⱼᵐᵃˢᵏᵉᵈ − mᵢ) ∈ ℝᴮʳˣᴮᶜ.
14:        On chip, compute dropout mask 𝐙ᵢⱼ ∈ ℝᴮʳˣᴮᶜ,
           where each entry has value 1 / (1 − p_drop) with probability
           1 − p_drop and value 0 with probability p_drop.
15:        On chip, compute 𝐏ᵢⱼᵈʳᵒᵖᵖᵉᵈ = 𝐏ᵢⱼ ∘ 𝐙ᵢⱼ
           (pointwise multiply).
16:        On chip, compute d𝐕̃ⱼ ← d𝐕̃ⱼ + (𝐏ᵢⱼᵈʳᵒᵖᵖᵉᵈ)ᵀd𝐎ᵢ ∈ ℝᴮᶜˣᵈ.
17:        On chip, compute d𝐏ᵢⱼᵈʳᵒᵖᵖᵉᵈ = d𝐎ᵢ𝐕ⱼᵀ ∈ ℝᴮʳˣᴮᶜ.
18:        On chip, compute d𝐏ᵢⱼ = d𝐏ᵢⱼᵈʳᵒᵖᵖᵉᵈ ∘ 𝐙ᵢⱼ
           (pointwise multiply).
19:        On chip, compute 𝐃ᵢ = rowsum(d𝐎ᵢ ∘ 𝐎ᵢ) ∈ ℝᴮʳ.
20:        On chip, compute d𝐒ᵢⱼ = 𝐏ᵢⱼ ∘ (d𝐏ᵢⱼ − 𝐃ᵢ) ∈ ℝᴮʳˣᴮᶜ.
21:        Write d𝐐ᵢ ← d𝐐ᵢ + τd𝐒ᵢⱼ𝐊ⱼ ∈ ℝᴮʳˣᵈ to HBM.
22:        On chip, compute d𝐊̃ⱼ ← d𝐊̃ⱼ + τd𝐒ᵢⱼᵀ𝐐ᵢ ∈ ℝᴮᶜˣᵈ.
23:    𝐞𝐧𝐝 𝐟𝐨𝐫
24:    Write d𝐊ⱼ ← d𝐊̃ⱼ, d𝐕ⱼ ← d𝐕̃ⱼ to HBM.
25: 𝐞𝐧𝐝 𝐟𝐨𝐫
26: Return d𝐐, d𝐊, d𝐕.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
function Δflash_attention₁!(
    dQ::AbstractArray{F,4},
    dK::AbstractArray{F,4},
    dV::AbstractArray{F,4},
    dO::AbstractArray{F,4},
    Q::AbstractArray{F,4},
    K::AbstractArray{F,4},
    V::AbstractArray{F,4},
    O::AbstractArray{F,4},
    ℓ::AbstractArray{F,4},
    m::AbstractArray{F,4},
    window::Tuple{Int,Int},
) where {F<:AbstractFloat}
    D, H, T, B = size(Q)
    Bᶜ, Bᵣ = min(T, 512), min(T, 512)

    n_kv_head = size(K, 2)
    heads_per_kv = H ÷ n_kv_head

    Tᵣ, Tᶜ = cld(T, Bᵣ), cld(T, Bᶜ)
    scale = inv(sqrt(F(D)))
    masked = -floatmax(F)

    S = similar(Q, Bᶜ, Bᵣ)
    M = similar(Q, Bool, Bᶜ, Bᵣ)
    dP = similar(Q, Bᶜ, Bᵣ)
    dS = similar(Q, Bᶜ, Bᵣ)
    Dᵢ = similar(Q, 1, Bᵣ)
    dOO = similar(Q, D, Bᵣ)

    for document in 1:B, head in 1:H
        kv_head = cld(head, heads_per_kv)

        for j in 1:Tᶜ
            blockⱼ = (j - 1) * Bᶜ + 1:min(j * Bᶜ, T)
            Kⱼ = @view K[:, kv_head, blockⱼ, document]
            Vⱼ = @view V[:, kv_head, blockⱼ, document]
            dKⱼ = @view dK[:, kv_head, blockⱼ, document]
            dVⱼ = @view dV[:, kv_head, blockⱼ, document]

            for i in 1:Tᵣ
                blockᵢ = (i - 1) * Bᵣ + 1:min(i * Bᵣ, T)
                nᵢ, nⱼ = length(blockᵢ), length(blockⱼ)
                @views begin
                    Qᵢ = Q[:, head, blockᵢ, document]
                    dQᵢ = dQ[:, head, blockᵢ, document]
                    Oᵢ = O[:, head, blockᵢ, document]
                    dOᵢ = dO[:, head, blockᵢ, document]
                    mᵢ = m[:, blockᵢ, head, document]
                    ℓᵢ = ℓ[:, blockᵢ, head, document]
                    Sᵢⱼ = S[1:nⱼ, 1:nᵢ]
                    Mᵢⱼ = M[1:nⱼ, 1:nᵢ]
                    dPᵢⱼ = dP[1:nⱼ, 1:nᵢ]
                    dSᵢⱼ = dS[1:nⱼ, 1:nᵢ]
                    Dᵢⱼ = Dᵢ[:, 1:nᵢ]
                    dOOᵢ = dOO[:, 1:nᵢ]
                end
                mul!(Sᵢⱼ, Kⱼ', Qᵢ)
                @. Sᵢⱼ *= scale

                attention_mask!(Mᵢⱼ, blockⱼ, blockᵢ, window)
                @. Sᵢⱼ = ifelse(Mᵢⱼ, Sᵢⱼ, masked)
                @. Sᵢⱼ = ifelse(
                    Mᵢⱼ,
                    exp(Sᵢⱼ - mᵢ) / max(ℓᵢ, eps(F)),
                    zero(F),
                )

                Pᵢⱼ = Sᵢⱼ

                # 16: d𝐕̃ⱼ ← d𝐕̃ⱼ + 𝐏ᵢⱼᵀd𝐎ᵢ
                mul!(dVⱼ, dOᵢ, Pᵢⱼ', one(F), one(F))

                # 17: d𝐏ᵢⱼ = d𝐎ᵢ𝐕ⱼᵀ
                mul!(dPᵢⱼ, Vⱼ', dOᵢ)

                # 19: 𝐃ᵢ = rowsum(d𝐎ᵢ ∘ 𝐎ᵢ)
                @. dOOᵢ = dOᵢ * Oᵢ
                sum!(Dᵢⱼ, dOOᵢ)

                # 20: d𝐒ᵢⱼ = 𝐏ᵢⱼ ∘ (d𝐏ᵢⱼ − 𝐃ᵢ)
                @. dSᵢⱼ = Pᵢⱼ * (dPᵢⱼ - Dᵢⱼ)

                # 21: d𝐐ᵢ ← d𝐐ᵢ + τ𝐊ⱼd𝐒ᵢⱼ
                mul!(dQᵢ, Kⱼ, dSᵢⱼ, scale, one(F))

                # 22: d𝐊̃ⱼ ← d𝐊̃ⱼ + τ𝐐ᵢd𝐒ᵢⱼᵀ
                mul!(dKⱼ, Qᵢ, dSᵢⱼ', scale, one(F))
            end
        end
    end
    return nothing
end


"""
Source: https://arxiv.org/abs/2307.08691
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
𝐀𝐥𝐠𝐨𝐫𝐢𝐭𝐡𝐦 𝟏: FʟᴀsʜAᴛᴛᴇɴᴛɪᴏɴ-𝟐 forward pass
𝐑𝐞𝐪𝐮𝐢𝐫𝐞: Matrices 𝐐, 𝐊, 𝐕 ∈ ℝᴺˣᵈ in HBM, block sizes B_c, Bᵣ.
1: Divide 𝐐 into Tᵣ = ⌈N / Bᵣ⌉ blocks 𝐐₁, …, 𝐐_Tᵣ of size Bᵣ × d each,
   and divide 𝐊, 𝐕 into T_c = ⌈N / B_c⌉ blocks
   𝐊₁, …, 𝐊_T_c and 𝐕₁, …, 𝐕_T_c, of size B_c × d each.
2: Divide the output 𝐎 ∈ ℝᴺˣᵈ into Tᵣ blocks 𝐎ᵢ, …, 𝐎_Tᵣ of size Bᵣ × d each,
   and divide the logsumexp 𝐋 into Tᵣ blocks Lᵢ, …, L_Tᵣ of size Bᵣ each.
3: 𝐟𝐨𝐫 1 ≤ i ≤ Tᵣ 𝐝𝐨
4:     Load 𝐐ᵢ from HBM to on-chip SRAM.
5:     On chip, initialize 𝐎ᵢ⁽⁰⁾ = (0)ᴮʳˣᵈ ∈ ℝᴮʳˣᵈ,
       ℓᵢ⁽⁰⁾ = (0)ᴮʳ ∈ ℝᴮʳ,
       mᵢ⁽⁰⁾ = (−∞)ᴮʳ ∈ ℝᴮʳ.
6:     𝐟𝐨𝐫 1 ≤ j ≤ T_c 𝐝𝐨
7:         Load 𝐊ⱼ, 𝐕ⱼ from HBM to on-chip SRAM.
8:         On chip, compute 𝐒ᵢ⁽ʲ⁾ = 𝐐ᵢ𝐊ⱼᵀ ∈ ℝᴮʳˣᴮᶜ.
9:         On chip, compute
           mᵢ⁽ʲ⁾ = max(mᵢ⁽ʲ⁻¹⁾, rowmax(𝐒ᵢ⁽ʲ⁾)) ∈ ℝᴮʳ,
           𝐏̃ᵢ⁽ʲ⁾ = exp(𝐒ᵢ⁽ʲ⁾ − mᵢ⁽ʲ⁾) ∈ ℝᴮʳˣᴮᶜ pointwise,
           ℓᵢ⁽ʲ⁾ = eᵐⁱ⁽ʲ⁻¹⁾⁻ᵐⁱ⁽ʲ⁾ ℓᵢ⁽ʲ⁻¹⁾ + rowsum(𝐏̃ᵢ⁽ʲ⁾) ∈ ℝᴮʳ.
10:        On chip, compute
           𝐎ᵢ⁽ʲ⁾ = diag(eᵐⁱ⁽ʲ⁻¹⁾⁻ᵐⁱ⁽ʲ⁾)⁻¹ 𝐎ᵢ⁽ʲ⁻¹⁾ + 𝐏̃ᵢ⁽ʲ⁾𝐕ⱼ.
11:    𝐞𝐧𝐝 𝐟𝐨𝐫
12:    On chip, compute 𝐎ᵢ = diag(ℓᵢ⁽ᵀᶜ⁾)⁻¹𝐎ᵢ⁽ᵀᶜ⁾.
13:    On chip, compute Lᵢ = mᵢ⁽ᵀᶜ⁾ + log(ℓᵢ⁽ᵀᶜ⁾).
14:    Write 𝐎ᵢ to HBM as the i-th block of 𝐎.
15:    Write Lᵢ to HBM as the i-th block of 𝐋.
16: 𝐞𝐧𝐝 𝐟𝐨𝐫
17: Return the output 𝐎 and the logsumexp 𝐋.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
function flash_attention₂(Q::AbstractArray{F,4}, K::AbstractArray{F,4}, V::AbstractArray{F,4}, window) where F

end

attention = flash_attention₁

# function attention(Q::CuMatrix, K::CuMatrix, V::CuMatrix, window)
#     # TODO: FA2 or FA3
# end

end
