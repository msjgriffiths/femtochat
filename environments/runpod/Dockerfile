# syntax=docker/dockerfile:1
# Build from the root of the FemtoChat checkout.
ARG BASE_IMAGE=julia:1.12.7-bookworm
FROM ${BASE_IMAGE}
ARG CUDA_VERSION=12.9
ENV JULIA_PROJECT=/opt/femtochat \
    JULIA_DEPOT_PATH=/opt/julia-depot: \
    JULIA_CPU_TARGET=generic \
    JULIA_NUM_PRECOMPILE_TASKS=1 \
    JULIA_PKG_PRECOMPILE_AUTO=0 \
    OPENBLAS_NUM_THREADS=1 \
    REACTANT_GPU=cuda \
    REACTANT_GPU_VERSION=${CUDA_VERSION}
WORKDIR /opt/femtochat
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl git ninja-build python3 \
    && rm -rf /var/lib/apt/lists/*
COPY Project.toml Manifest.toml ./
COPY environments/reactant/Project.toml environments/reactant/Project.toml
COPY src/ src/
COPY ext/ ext/
COPY scripts/ scripts/
COPY deps/reactant/build.sh deps/reactant/xla_ffi.cpp deps/reactant/attention_rule.cpp deps/reactant/

# These preferences must be visible to Julia before it selects artifacts or builds caches.
RUN julia --startup-file=no -e 'using TOML; \
    project = TOML.parsefile("Project.toml"); \
    project["extras"]["CUDA_Runtime_jll"] = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"; \
    open(io -> TOML.print(io, project), "Project.toml", "w"); \
    prefs = Dict("FemtoChat" => Dict("precompile_training_gradient" => true), \
                 "CUDA_Runtime_jll" => Dict("version" => ENV["REACTANT_GPU_VERSION"])); \
    open(io -> TOML.print(io, prefs), "LocalPreferences.toml", "w")'
RUN julia --startup-file=no -e 'using Pkg; Pkg.instantiate(; allow_autoprecomp=false); Pkg.precompile(); \
    using FemtoChat, Enzyme, CUDA, Reactant; \
    @assert FemtoChat._precompile_training_gradient; \
    @assert isdefined(Base.get_extension(FemtoChat, :FemtoChatReactantExt), :compile_training_gradient); \
    jll = Reactant.Reactant_jll; \
    @assert jll.host_platform.tags["gpu"] == "cuda"; \
    @assert jll.host_platform.tags["gpu_version"] == ENV["REACTANT_GPU_VERSION"]; \
    run(`bash deps/reactant/build.sh $(jll.libReactantExtra_path)`)' \
    && rm -rf deps/reactant/build
CMD ["julia","--startup-file=no","--project=.","--threads=6","scripts/train.jl"]
