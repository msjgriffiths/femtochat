module Checkpoints

using Serialization
using Printf

export save_checkpoint, load_checkpoint, optimizer_state, restore_optimizer!

# Save persistent state, not device handles, workspaces, or views into Θ/δ.
optimizer_state(ω) = (
    adamw=map(u -> (; 𝓂ₜ=Array(u.ω.𝓂ₜ), 𝓋ₜ=Array(u.ω.𝓋ₜ), t=Int(u.ω.t)), ω.adamw),
    muon=map(g -> (; 𝓂ₜ=Array(g.update.ω.𝓂ₜ), 𝓋ₜ=Array(g.update.ω.𝓋ₜ)), ω.muon),
)

function restore_optimizer!(ω, saved)
    for (u, state) in zip(ω.adamw, saved.adamw)
        copyto!(u.ω.𝓂ₜ, state.𝓂ₜ)
        copyto!(u.ω.𝓋ₜ, state.𝓋ₜ)
        u.ω.t = convert(typeof(u.ω.t), state.t)
    end
    for (g, state) in zip(ω.muon, saved.muon)
        copyto!(g.update.ω.𝓂ₜ, state.𝓂ₜ)
        copyto!(g.update.ω.𝓋ₜ, state.𝓋ₜ)
    end
    nothing
end

function save_checkpoint(path, step, parameters, optimizer, metadata, rank=0)
    rank == 0 || return
    mkpath(path)
    file = joinpath(path, @sprintf("checkpoint_%06d.jls", step))
    serialize(file * ".part", (; version=1, step, parameters=Array(parameters),
                                optimizer=optimizer_state(optimizer), metadata))
    # Publish only a complete checkpoint, leaving the previous one intact on failure.
    mv(file * ".part", file; force=true)
    file
end

load_checkpoint(path, step) = load_checkpoint(joinpath(path, @sprintf("checkpoint_%06d.jls", step)))
load_checkpoint(file) = deserialize(file)

end
