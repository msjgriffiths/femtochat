module Kernels

using LinearAlgebra: mul!

export attention, softmax!

function attention_mask(X, T, window)
    left, right = window
    mask = similar(X, Bool, T, T)
    key = reshape(1:T, T, 1)
    query = reshape(1:T, 1, T)

    if window == (-1, 0)
        @. mask = key <= query
    else
        @. mask = (query - left <= key) & (key <= query + right)
    end

    return mask
end

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

# function attention(Q::CuMatrix, K::CuMatrix, V::CuMatrix, window)
#     # TODO: FA2 or FA3
# end

end
