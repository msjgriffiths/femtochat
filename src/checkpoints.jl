module Checkpoints

using Serialization
using Printf

function save_checkpoint(path, step, parameters, optimizer, metadata, rank=0)
    if rank != 0 && return end  
    serialize(
        joinpath(path, @sprintf("checkpoint_%06d.jls", step)), 
        (; step, parameters, optimizer, metadata)
    )
end

function load_checkpoint(path, step)
    checkpoint_file = joinpath(path, @sprintf("checkpoint_%06d.jls", step))
    return deserialize(checkpoint_file)
end

end 