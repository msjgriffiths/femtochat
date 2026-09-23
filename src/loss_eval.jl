module Loss

using ..GPT: cross_entropy
using Base.Iterators: take

export evaluate_bpb

function evaluate_bpb(
    model,
    batches,
    steps::Integer,
    token_bytes::AbstractVector{<:Integer},
)
    total_nats = nothing
    total_bytes = nothing

    for batch in take(batches, steps)
        tokens, targets = batch
        logits = hasproperty(batch, :positions) ? model(tokens; positions=batch.positions) : model(tokens)
        losses = cross_entropy(logits, targets; reduction=:none)

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
