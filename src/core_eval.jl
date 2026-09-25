module CoreEval

using Random: MersenneTwister, randperm, shuffle
using DuckDB, DBInterface, Tables, YAML
using ..Parameters: 🤖
using ..Tokenizer: bos_token_id
using ..GPT: cross_entropy

export MultipleChoice, Schema, LanguageModeling, CoreTask,
       score_continuations, evaluate_task, evaluate_core, load_core_tasks

struct MultipleChoice end   # One context, several answers.
struct Schema end           # Several contexts, one answer.
struct LanguageModeling end # Every answer token must match.

"""A CORE task with examples or a JSONL path. In-memory `gold` indices are one-based."""
Base.@kwdef struct CoreTask{F,D}
    format::F
    data::D
    label::String
    num_fewshot::Int = 0
    delimiter::String = " "
    baseline::Float64 = 0.0 # Chance accuracy in [0, 1].
end

# Loading

function read_rows(path, reader)
    con = DBInterface.connect(DuckDB.DB, ":memory:")
    try
        Tables.rowtable(DBInterface.execute(con, "SELECT * FROM $reader(?)", [path]))
    finally
        DBInterface.close!(con)
    end
end

# Convert zero-based JSON answer indices once, when reading the examples.
example(::MultipleChoice, x) = (; query=x.query, choices=x.choices, gold=Int(x.gold)+1)
example(::Schema, x) = (; context_options=x.context_options, continuation=x.continuation, gold=Int(x.gold)+1)
example(::LanguageModeling, x) = (; context=x.context, continuation=x.continuation)

examples(format, path::AbstractString) = example.(Ref(format), read_rows(path, "read_json_auto"))
examples(format, data::AbstractVector) = data

"""Read nanochat's task definitions and chance baselines from an extracted evaluation bundle."""
function load_core_tasks(directory::AbstractString)
    metadata = read_rows(joinpath(directory, "eval_meta_data.csv"), "read_csv_auto")
    baselines = Dict(r[Symbol("Eval Task")] => Float64(r[Symbol("Random baseline")])/100 for r ∈ metadata)
    formats = Dict("multiple_choice" => MultipleChoice(), "schema" => Schema(),
                   "language_modeling" => LanguageModeling())
    map(YAML.load_file(joinpath(directory, "core.yaml"))["icl_tasks"]) do task
        CoreTask(format=formats[task["icl_task_type"]], label=task["label"],
            data=joinpath(directory, "eval_data", task["dataset_uri"]),
            num_fewshot=first(task["num_fewshot"]),
            delimiter=get(task, "continuation_delimiter", " "), baseline=baselines[task["label"]])
    end
end

# Prompts: answered examples, followed by the question being scored.

prompt_parts(::MultipleChoice, x) = (x.query, x.choices)
prompt_parts(::Schema, x) = (x.context_options, x.continuation)
prompt_parts(::LanguageModeling, x) = (strip(x.context), ["", x.continuation])
gold_index(::Union{MultipleChoice,Schema}, x) = x.gold
gold_index(::LanguageModeling, x) = 2

trim_context(format, prompts) = prompts
trim_context(::LanguageModeling, prompts) = [strip(first(prompts)), last(prompts)]

function render_prompts(format, item, delimiter, examples=())
    function render(x)
        context, answers = prompt_parts(format, x)
        string.(context, delimiter, answers)
    end
    prefix = join(render(x)[gold_index(format, x)] * "\n\n" for x ∈ examples)
    trim_context(format, prefix .* render(item))
end

# Answer tokens: after the shared prefix, or within the shared suffix.

common_length(tokens) = count(_ -> true, Iterators.takewhile(allequal, zip(tokens...)))

function continuations(::MultipleChoice, tokens)
    i₀ = common_length(tokens) + 1
    tokens, [i₀:length(t) for t ∈ tokens]
end

function continuations(::Schema, tokens)
    n = common_length(Iterators.reverse.(tokens))
    tokens, [length(t)-n+1:length(t) for t ∈ tokens]
end

function continuations(::LanguageModeling, tokens)
    prefix, complete = tokens
    length(prefix) < length(complete) && common_length(tokens) == length(prefix) ||
        throw(ArgumentError("language-modeling context must be a strict token prefix of its completion"))
    [complete], [length(prefix)+1:length(complete)]
end

function evaluation_batch(format, 𝒯, prompts; max_tokens=typemax(Int))
    # Tokenize whole prompts: BPE merges can cross the text/answer boundary.
    sequences = [[Int32(bos_token_id(𝒯)); Int32.(𝒯(p))] for p ∈ prompts]
    sequences, answers = continuations(format, sequences)
    T, B = min(maximum(length, sequences), max_tokens), length(sequences)
    tokens = fill(Int32(bos_token_id(𝒯)), T, B)
    targets = fill(Int32(-1), size(tokens))
    for (b, (sequence, answer)) ∈ enumerate(zip(sequences, answers))
        # Discard left context, but retain the answer and its preceding token.
        dropped = max(0, length(sequence) - T)
        !isempty(answer) && first(answer) > dropped + 1 ||
            throw(ArgumentError("each answer needs at least one scored token and a preceding context token"))
        @views tokens[1:length(sequence)-dropped, b] .= sequence[dropped+1:end]
        # Logits at t predict token t+1; -1 leaves context and padding unscored.
        @views targets[answer .- (dropped+1), b] .= sequence[answer]
    end
    (; tokens, targets)
end

# Scoring: mean answer-token loss for choices, exact match for language modeling.

context_length(ℳ::🤖) = ℳ.config.sequence_len
context_length(ℳ) = typemax(Int)
evaluation_input(ℳ::🤖, x) = copyto!(similar(ℳ.Θ, eltype(x), size(x)), x)
evaluation_input(ℳ, x) = x

function scores(::Union{MultipleChoice,Schema}, logits, targets)
    ℒ = cross_entropy(Float32.(logits), targets; reduction=:none)
    Nₜ = sum(targets .!= -1; dims=1)
    vec(Array(sum(ℒ; dims=1) ./ Nₜ))
end

function scores(::LanguageModeling, logits, targets)
    ŷ = reshape(getindex.(argmax(logits; dims=1), 1), size(targets))
    vec(Array(all((ŷ .== targets) .| (targets .== -1); dims=1)))
end

correct(::Union{MultipleChoice,Schema}, s, item) = argmin(s) == item.gold
correct(::LanguageModeling, s, item) = only(s)

function score_example(format, ℳ, 𝒯, item, examples=();
                       delimiter=" ", max_tokens=context_length(ℳ))
    prompts = render_prompts(format, item, delimiter, examples)
    batch = evaluation_batch(format, 𝒯, prompts; max_tokens)
    tokens, targets = evaluation_input.((ℳ,), (batch.tokens, batch.targets))
    scores(format, ℳ(tokens), targets)
end

"""
    score_continuations(ℳ, 𝒯, prompt, answers; delimiter=" ")

Mean answer-token losses; lower is better. Like nanochat, score only tokens after
the candidates' common prefix. Supply at least two answers, each with a nonempty
suffix. Context and padding do not contribute to the loss.
"""
score_continuations(ℳ, 𝒯, prompt::AbstractString, answers; kwargs...) =
    score_example(MultipleChoice(), ℳ, 𝒯, (; query=prompt, choices=answers); kwargs...)

# Evaluation: score questions, then average chance-adjusted task accuracies.

"""
Evaluate one task. `max_examples` limits scored questions, not the few-shot pool.
Julia's seeded RNG gives reproducible draws, but not the same draws as Python.
Use `fewshot_indices(i, n, k)` to supply the same examples for paired comparisons.
"""
function evaluate_task(ℳ, 𝒯, task::CoreTask; max_examples=-1,
                       max_tokens=context_length(ℳ), seed=1234,
                       shuffle_seed=nothing, fewshot_indices=nothing)
    𝒟 = examples(task.format, task.data)
    isnothing(shuffle_seed) || (𝒟 = shuffle(MersenneTwister(shuffle_seed), 𝒟))
    n, k = length(𝒟), task.num_fewshot
    n > k || throw(ArgumentError("task needs more examples than its few-shot count"))
    total = max_examples > 0 ? min(max_examples, n) : n
    ncorrect = 0
    for i ∈ 1:total
        indices = if k == 0
            Int[]
        elseif isnothing(fewshot_indices)
            ℛ = MersenneTwister(seed+i-1)
            selected = randperm(ℛ, n-1)[1:k]
            selected .+ (selected .>= i) # Sample without replacement, excluding i.
        else
            fewshot_indices(i, n, k)
        end
        s = score_example(task.format, ℳ, 𝒯, 𝒟[i], 𝒟[indices];
                          delimiter=task.delimiter, max_tokens)
        ncorrect += correct(task.format, s, 𝒟[i])
    end
    (; accuracy=ncorrect/total, correct=ncorrect, total)
end

"""Return per-task accuracies and their chance-adjusted mean: 0 is chance, 1 is perfect."""
function evaluate_core(ℳ, 𝒯, tasks::AbstractVector; max_per_task=-1,
                       shuffle_seed=1337, logger=identity, kwargs...)
    rows = map(tasks) do task
        t₀ = time()
        result = evaluate_task(ℳ, 𝒯, task; max_examples=max_per_task, shuffle_seed, kwargs...)
        α, α₀ = result.accuracy, task.baseline
        centered = (α - α₀) / (1 - α₀)
        row = (; task.label, result..., centered, seconds=time()-t₀)
        logger(row)
        row
    end
    (; core_metric=sum(r -> r.centered, rows)/length(rows), tasks=rows)
end

evaluate_core(ℳ, 𝒯, directory::AbstractString; kwargs...) =
    evaluate_core(ℳ, 𝒯, load_core_tasks(directory); kwargs...)

end
