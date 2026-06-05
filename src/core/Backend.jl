abstract type AbstractBackend end

struct CPUBackend <: AbstractBackend end
struct CUDABackend <: AbstractBackend end

backend(::Val{:cpu}) = CPUBackend()
backend(::Val{:cuda}) = CUDABackend()
backend(name::Symbol=:cpu) = backend(Val(name))
backend(::AbstractString) = error("backend expects :cpu or :cuda")

function zeros_backend(::CPUBackend, ::Type{T}, dims::Integer...) where {T}
    return zeros(T, Int.(dims)...)
end

function zeros_backend(::CUDABackend, ::Type{T}, dims::Integer...) where {T}
    error("CUDA backend requires CUDA.jl. Load the CUDA extension or use backend(:cpu).")
end

adapt_backend(::CPUBackend, x) = x
adapt_backend(::CUDABackend, x) = error("CUDA adaptation requires the CUDA/Adapt extensions.")

is_gpu_backend(::AbstractBackend) = false
is_gpu_backend(::CUDABackend) = true
