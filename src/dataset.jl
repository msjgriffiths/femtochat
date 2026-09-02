module Dataset

using Downloads: download
using Printf: @sprintf

export download_dataset!

const BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
const MAX_SHARD = 6542

function download_shard(index, directory)
    filename = @sprintf("shard_%05d.parquet", index)
    path = joinpath(directory, filename)

    isfile(path) || download("$BASE_URL/$filename", path)
end

download_dataset!(directory::AbstractString, n::Int = MAX_SHARD) = begin
    for index in 0:n
        download_shard(index, directory)
    end
    download_shard(MAX_SHARD, directory)
end


end