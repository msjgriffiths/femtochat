module FemtoChatCUDAExt

using CUDA
using FemtoChat

import FemtoChat.Kernels: attention, flash_attention₁!

@inline function shared_array(::Type{F}, dims, offset) where F
    array = CuDynamicSharedArray(F, dims, offset)
    return array, offset + sizeof(F) * length(array)
end

function attention(
    Q::CuArray{F,4},
    K::CuArray{F,4},
    V::CuArray{F,4},
    window,
) where F
    _, H, T, B = size(Q)

    𝕆 = similar(Q)
    ℓ = similar(Q, 1, T, H, B)
    m = similar(Q, 1, T, H, B)

    flash_attention₁!(𝕆, ℓ, m, Q, K, V, window)
    return 𝕆
end

function flash_attention₁!(
    𝕆::CuArray{F,4},
    ℓ::CuArray{F,4},
    m::CuArray{F,4},
    Q::CuArray{F,4},
    K::CuArray{F,4},
    V::CuArray{F,4},
    window,
    ::Val{Bᵣ}=Val(16),
    ::Val{Bᶜ}=Val(16),
) where {F,Bᵣ,Bᶜ}
    D, H, T, B = size(Q)

    threads = Bᵣ * Bᶜ
    blocks = (cld(T, Bᵣ), H, B)

    # Qᵢ, Kⱼ, Vⱼ, 𝕆ᵢ, ℓᵢ, mᵢ, αᵢ, and Sᵢⱼ
    elements = 2D * (Bᵣ + Bᶜ) + 3Bᵣ + Bᶜ * Bᵣ

    bytes = sizeof(F) * elements

    @cuda threads=threads blocks=blocks shmem=bytes flash_attention₁_kernel!(
        𝕆, ℓ, m, Q, K, V, window,
        Val(D), Val(Bᵣ), Val(Bᶜ),
    )

    return nothing
end

"""
Fused FlashAttention 1 device kernel.
"""
function flash_attention₁_kernel!(
    𝕆,
    ℓ,
    m,
    Q,
    K,
    V,
    window,
    ::Val{D},
    ::Val{Bᵣ},
    ::Val{Bᶜ},
) where {D,Bᵣ,Bᶜ}
    F = eltype(Q)
    𝟎 = zero(F) # Zero in the type of the passed array

    offset = 0
    Qᵢ, offset = shared_array(F, (D, Bᵣ), offset)
    Kⱼ, offset = shared_array(F, (D, Bᶜ), offset)
    Vⱼ, offset = shared_array(F, (D, Bᶜ), offset)
    𝕆ᵢ, offset = shared_array(F, (D, Bᵣ), offset)
    ℓᵢ, offset = shared_array(F, Bᵣ, offset)
    mᵢ, offset = shared_array(F, Bᵣ, offset)
    αᵢ, offset = shared_array(F, Bᵣ, offset)
    Sᵢⱼ, _ = shared_array(F, (Bᶜ, Bᵣ), offset)

    τᵢ = threadIdx().x
    τₙ = blockDim().x

    i, head, document = blockIdx()

    T = size(Q, 3)
    Tᶜ = cld(T, Bᶜ)
    blockᵢ = (i - 1) * Bᵣ

    n_kv_head = size(K, 2)
    heads_per_kv = size(Q, 2) ÷ n_kv_head
    kv_head = cld(head, heads_per_kv)

    scale = inv(sqrt(F(D)))
    ∅ = typemin(F) # Mask by setting to "negative inf"

    # Load Qᵢ from HBM and initialize 𝕆ᵢ, ℓᵢ, and mᵢ on chip.
    # Each thread jumps by number of threads from current index to 
    # end of the dimension.
    @inbounds for index = τᵢ:τₙ:(D * Bᵣ)
        q, d = fldmod1(index, D)
        query = blockᵢ + q

        Qᵢ[d, q] = query <= T ? Q[d, head, query, document] : 𝟎
        𝕆ᵢ[d, q] = 𝟎
    end

    @inbounds for q = τᵢ:τₙ:Bᵣ
        ℓᵢ[q] = 𝟎
        mᵢ[q] = ∅
    end

    sync_threads()

    for j = 1:Tᶜ
        blockⱼ = (j - 1) * Bᶜ

        # Load Kⱼ and Vⱼ from HBM to on-chip shared memory.
        @inbounds for index = τᵢ:τₙ:(D * Bᶜ)
            k, d = fldmod1(index, D)
            key = blockⱼ + k

            if key <= T
                Kⱼ[d, k] = K[d, kv_head, key, document]
                Vⱼ[d, k] = V[d, kv_head, key, document]
            else
                Kⱼ[d, k] = 𝟎
                Vⱼ[d, k] = 𝟎
            end
        end

        sync_threads()

        # Sᵢⱼ = Kⱼ'Qᵢ. Each thread computes one or more scores.
        @inbounds for index = τᵢ:τₙ:(Bᶜ * Bᵣ)
            q, k = fldmod1(index, Bᶜ)
            key = blockⱼ + k
            query = blockᵢ + q

            valid = key <= T && query <= T
            if valid
                left, right = window
                valid = window == (-1, 0) ?
                    key <= query :
                    query - left <= key && key <= query + right
            end

            if valid
                score = 𝟎
                for d = 1:D
                    score = muladd(Qᵢ[d, q], Kⱼ[d, k], score)
                end
                Sᵢⱼ[k, q] = score * scale
            else
                Sᵢⱼ[k, q] = ∅
            end
        end

        sync_threads()

        # Update the running maximum for each query.
        if τᵢ <= Bᵣ
            q = τᵢ
            query = blockᵢ + q

            if query <= T
                m̃ᵢⱼ = ∅
                @inbounds for k = 1:Bᶜ
                    m̃ᵢⱼ = max(m̃ᵢⱼ, Sᵢⱼ[k, q])
                end

                if m̃ᵢⱼ == ∅
                    αᵢ[q] = one(F)
                else
                    mᵢⁿᵉʷ = max(mᵢ[q], m̃ᵢⱼ)
                    αᵢ[q] = mᵢ[q] == ∅ ?
                        𝟎 : exp(mᵢ[q] - mᵢⁿᵉʷ)
                    mᵢ[q] = mᵢⁿᵉʷ
                end
            else
                αᵢ[q] = 𝟎
            end
        end

        sync_threads()

        # Rescale the previous numerator and form Pᵢⱼ in place of Sᵢⱼ.
        @inbounds for index = τᵢ:τₙ:(D * Bᵣ)
            q = cld(index, D)
            𝕆ᵢ[index] *= αᵢ[q]
        end

        @inbounds for index = τᵢ:τₙ:(Bᶜ * Bᵣ)
            q = cld(index, Bᶜ)
            score = Sᵢⱼ[index]
            Sᵢⱼ[index] = score == ∅ ?
                𝟎 : exp(score - mᵢ[q])
        end

        sync_threads()

        # ℓᵢ ← αᵢℓᵢ + rowsum(Pᵢⱼ).
        if τᵢ <= Bᵣ
            q = τᵢ
            ℓ̃ᵢⱼ = 𝟎
            @inbounds for k = 1:Bᶜ
                ℓ̃ᵢⱼ += Sᵢⱼ[k, q]
            end
            ℓᵢ[q] = αᵢ[q] * ℓᵢ[q] + ℓ̃ᵢⱼ
        end

        # 𝕆ᵢ ← αᵢ𝕆ᵢ + VⱼPᵢⱼ.
        @inbounds for index = τᵢ:τₙ:(D * Bᵣ)
            q, d = fldmod1(index, D)
            𝕆̃ᵢⱼ = 𝟎

            for k = 1:Bᶜ
                𝕆̃ᵢⱼ = muladd(Vⱼ[d, k], Sᵢⱼ[k, q], 𝕆̃ᵢⱼ)
            end

            𝕆ᵢ[d, q] += 𝕆̃ᵢⱼ
        end

        sync_threads()
    end

    # Normalize the accumulated numerator and write the result to HBM.
    @inbounds for index = τᵢ:τₙ:(D * Bᵣ)
        q, d = fldmod1(index, D)
        query = blockᵢ + q

        if query <= T
            𝕆[d, head, query, document] = 𝕆ᵢ[d, q] / ℓᵢ[q]
        end
    end

    if τᵢ <= Bᵣ
        query = blockᵢ + τᵢ

        if query <= T
            ℓ[1, query, head, document] = ℓᵢ[τᵢ]
            m[1, query, head, document] = mᵢ[τᵢ]
        end
    end

    return nothing
end

end
