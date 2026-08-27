module Kernels

using LinearAlgebra: mul!, UpperTriangular
using CUDA: CuMatrix

function attention_mask(T, window::Tuple{Int,Int})
    left, right = window

    if window == (-1, 0)
        UpperTriangular(trues(T, T))
    else
        [q - left <= k <= q + right for k in 1:T, q in 1:T]
    end
end

function softmax!(X; dims=1)
    X .-= maximum(X; dims)
    X .= exp.(X)
    X ./= sum(X; dims)
    X
end

function attention(Q::AbstractArray{F,4}, K::AbstractArray{F,4}, V::AbstractArray{F,4}, window) where F
    D, H, T, B = size(Q)
    mask = attention_mask(T, window)
    
    # Output to store result in
    # key × query × head × batch
    S = Array{F}(undef, T, T, H, B)
    for document in 1:B, head in 1:H
        # For each document in document
        # For each head in H
        @views mul!(
            S[:, :, head, document], # Place the result in S, one for each head and document
            K[:, head, :, document]', # Transpose K for the dot product
            Q[:, head, :, document] # Q for the dot product
        )
    end
    S = S ./ sqrt(D) # Scale by sqrt of dimension

    S .= ifelse.(mask, S, -Inf) # Apply mask
    softmax!(S; dims=1) # Softmax along the key dimension

    Y = similar(Q) # Create output matrix same shape as Q
    for document in 1:B, head in 1:H
        # For each document in document
        # For each head in H
        @views mul!(
            Y[:, head, :, document], # Place the result in Y, one for each head and document
            V[:, head, :, document], # Value matrix is D x head x token x document
            S[:, :, head, document] # S is token x token x head x document
        )
    end

    Y
end

function attention(Q::CuMatrix, K::CuMatrix, V::CuMatrix, window)
    # TODO: FA2 or FA3
end

end