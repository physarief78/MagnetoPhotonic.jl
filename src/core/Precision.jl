default_compute_type(::CPUBackend) = Float64
default_compute_type(::CUDABackend) = Float32

function resolve_compute_type(spec, b::AbstractBackend)
    spec === nothing && return default_compute_type(b)
    spec === :auto && return default_compute_type(b)
    spec === :default && return default_compute_type(b)
    spec === :f32 && return Float32
    spec === :float32 && return Float32
    spec === Float32 && return Float32
    spec === :f64 && return Float64
    spec === :float64 && return Float64
    spec === Float64 && return Float64
    throw(ArgumentError("compute_precision must be :auto, :f32, :f64, Float32, or Float64"))
end

compute_type(b::AbstractBackend, spec=:auto) = resolve_compute_type(spec, b)
compute_type(cfg::BackendConfig) = compute_type(backend(cfg), cfg.compute_precision)
compute_type(cfg::SimConfig) = compute_type(cfg.backend)

storage_type(spec) = spec isa Type ? spec : throw(ArgumentError("storage precision must be a concrete floating-point type"))
