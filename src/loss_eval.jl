module Loss

using ..GPT: cross_entropy
using Iterators: take

export evaluate_bpb

function evaluate_bpb(
    model,
    batches,
    steps::Integer,
    token_bytes::AbstractVector{<:Integer},
)
    total_nats = nothing
    total_bytes = nothing

    for (tokens, targets) in take(batches, steps)
        losses = cross_entropy(model(tokens), targets; reduction=:none)

        valid = targets .!= -1
        safe_targets = ifelse.(valid, targets, one(eltype(targets)))
        bytes = ifelse.(
            valid,
            token_bytes[safe_targets],
            zero(eltype(token_bytes)),
        )

        nats = sum(
            ifelse.(bytes .> 0, losses, zero(eltype(losses)));
            dims=(1, 2),
        )
        nbytes = sum(bytes; dims=(1, 2))

        if isnothing(total_nats)
            total_nats = nats
            total_bytes = nbytes
        else
            total_nats .+= nats
            total_bytes .+= nbytes
        end
    end

    total_nats ./ (log(2f0) .* total_bytes)
end

end