# Optional XLA/CUDA bridge. Numerical kernels remain in FemtoChatCUDAExt.
# Typed XLA FFI passes its own stream and buffers to our existing Julia kernels.
module ReactantCUDACall
using Reactant, CUDA
const IR = Reactant.MLIR.IR
const stablehlo = Reactant.MLIR.Dialects.stablehlo
const StreamHandle = fieldtype(CUDA.CuStream,:handle)
const libffi = joinpath(@__DIR__,"..","..","deps","reactant","libreactant_xla_ffi.so")
const librule = joinpath(@__DIR__,"..","..","deps","reactant","libreactant_attention_rule.so")
const registry = Dict{Any,Any}()
const errors = Any[]

# CUDA.jl has no public constructor for a borrowed stream in the tested version.
# XLA owns this stream: deliberately do not attach a destruction finalizer.
@generated borrowed_stream(handle::StreamHandle, ctx::CUDA.CuContext) =
    Expr(:new, CUDA.CuStream, :handle, true, :ctx)

function launch_buffers(f,::Val{Types},::Val{Shapes},::Val{N},buffers) where {Types,Shapes,N}
    arrays = ntuple(Val(length(Shapes))) do i
        T,shape = fieldtype(Types,i),Shapes[i]
        ptr = reinterpret(CUDA.CuPtr{T},UInt(unsafe_load(buffers,i)))
        unsafe_wrap(CUDA.CuArray,ptr,shape;own=false)
    end
    f(arrays[N+1:end]...,arrays[1:N]...)
end

function register(f, outputs, inputs)
    get!(registry, (f,outputs,inputs)) do
        # The plug-in uses the compiler ABI, not a stable external C API.
        artifact = joinpath(dirname(libffi),"artifact.txt")
        isfile(artifact) && strip(read(artifact,String)) == basename(Reactant.Reactant_jll.artifact_dir) ||
            error("Rebuild the Reactant adapters with deps/reactant/build.sh")
        ctx = CUDA.context()
        specs = (inputs...,outputs...)
        # Keep buffer arguments a typed tuple rather than a dynamically typed vector.
        types,shapes,n = Val(Tuple{first.(specs)...}),Val(last.(specs)),Val(length(inputs))
        function launch(stream, buffers)
            try
                CUDA.context!(ctx) do
                    CUDA.stream!(borrowed_stream(stream,ctx)) do
                        launch_buffers(f,types,shapes,n,buffers)
                    end
                end
                return true
            catch err
                # Never unwind a Julia exception through XLA's C ABI.
                push!(errors,(err,catch_backtrace()))
                return false
            end
        end
        callback = @cfunction($launch,Bool,(StreamHandle,Ptr{Ptr{Cvoid}}))
        name = "femtochat_$(length(registry)+1)_$(nameof(typeof(f)))"
        error = ccall((:register_femtochat_ffi,libffi),Ptr{Cvoid},(Cstring,),name)
        @assert error == C_NULL "XLA FFI registration failed"
        (;name,callback,launch) # Root the closure and C function for executable lifetime.
    end
end

function call(f,outputs,args...;vjp=nothing,alias_input=nothing)
    inputs = map(x -> (Reactant.unwrapped_eltype(typeof(x)),size(x)),args)
    # Callback creation belongs to state preparation, never the tracing context.
    entry = registry[(f,outputs,inputs)]
    op = stablehlo.custom_call(IR.Value[x.mlir_data for x in args];
        result_0=IR.Type[IR.TensorType(collect(Int,shape),IR.Type(T)) for (T,shape) in outputs],
        call_target_name=entry.name,api_version=Int32(4),
        has_side_effect=IR.Attribute(false),
        backend_config=Dict("callback_ptr" => IR.Attribute(Int64(UInt(Base.unsafe_convert(Ptr{Cvoid},entry.callback))))),
        operand_layouts=Reactant.Ops._col_major_layout(inputs),
        result_layouts=Reactant.Ops._col_major_layout(outputs))
    if alias_input !== nothing
        alias = parse(IR.Attribute,"#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = $alias_input, operand_tuple_indices = []>")
        IR.setattr!(op,"output_operand_aliases",IR.Attribute([alias]))
    end
    if vjp !== nothing
        # Attach a reverse recipe to this operation, not to the whole loss.
        reverse_inputs = (outputs[1],inputs...,outputs...)
        reverse_outputs = Tuple(spec for spec in inputs if first(spec) <: AbstractFloat)
        reverse = registry[(vjp,reverse_outputs,reverse_inputs)]
        IR.setattr!(op,"femtochat.backward_target",IR.Attribute(reverse.name))
        IR.setattr!(op,"femtochat.backward_config",IR.Attribute(Dict(
            "callback_ptr" => IR.Attribute(Int64(UInt(Base.unsafe_convert(Ptr{Cvoid},reverse.callback)))))))
        IR.setattr!(op,"femtochat.backward_operand_layouts",IR.Attribute(Reactant.Ops._col_major_layout(reverse_inputs)))
        IR.setattr!(op,"femtochat.backward_result_layouts",IR.Attribute(Reactant.Ops._col_major_layout(reverse_outputs)))
        ok = ccall((:register_femtochat_typed_reverse_rule,librule),Bool,
                   (Reactant.MLIR.API.MlirContext,),IR.context(op))
        @assert ok "Could not attach the Enzyme-MLIR reverse interface"
    end
    map(enumerate(outputs)) do (i,(T,shape))
        Reactant.TracedRArray{T,length(shape)}((),IR.result(op,i),shape)
    end |> Tuple
end
Reactant.@skip_rewrite_func call

function check()
    isempty(errors) || throw(first(errors)[1])
    nothing
end

function __init__()
    Reactant.@skip_rewrite_func call
end
end
