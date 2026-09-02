module Generation

using Random: AbstractRNG, default_rng, rand

export generate

function sample_token(rng, logits, temperature, top_k)
    iszero(temperature) && return argmax(logits)

    k = min(top_k, length(logits))
    candidates = partialsortperm(logits, 1:k; rev=true)
    scores = logits[candidates] ./ temperature
    weights = exp.(scores .- maximum(scores))
    threshold = rand(rng) * sum(weights)

    cumulative = zero(eltype(weights))
    for (token, weight) in zip(candidates, weights)
        cumulative += weight
        cumulative >= threshold && return token
    end

    return last(candidates)
end

"""Generate tokens autoregressively, using at most the model's context window."""
function generate(
    model,
    prompt::AbstractVector{<:Integer};
    max_new_tokens::Int=20,
    temperature::Real=0.8f0,
    top_k::Int=40,
    rng::AbstractRNG=default_rng(),
)
    isempty(prompt) && throw(ArgumentError("prompt must contain at least one token"))
    max_new_tokens >= 0 || throw(ArgumentError("max_new_tokens must be nonnegative"))
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    top_k > 0 || throw(ArgumentError("top_k must be positive"))

    tokens = collect(prompt)
    context_length = model.config.sequence_len

    for _ in 1:max_new_tokens
        first_token = max(1, length(tokens) - context_length + 1)
        context = @view tokens[first_token:end]
        input = similar(model.λₛ, Int32, length(context), 1)
        copyto!(input, reshape(Int32.(context), :, 1))

        logits = Array(model(input))[:, end, 1]
        token = sample_token(rng, logits, temperature, top_k)
        push!(tokens, convert(eltype(tokens), token))
    end

    return tokens
end

end
