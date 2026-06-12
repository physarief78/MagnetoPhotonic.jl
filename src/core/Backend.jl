abstract type AbstractBackend end

struct CPUBackend <: AbstractBackend end
struct CUDABackend <: AbstractBackend end

backend(::Val{:cpu}) = CPUBackend()
backend(::Val{:cuda}) = CUDABackend()
backend(name::Symbol=:cpu) = backend(Val(name))
backend(name::AbstractString) = backend(Symbol(name))
backend(cfg::BackendConfig) = backend(cfg.device)
backend(cfg::SimConfig) = backend(cfg.backend)

# Backend hooks. The CPU methods are concrete; the non-CPU methods are GENERIC
# `AbstractBackend` fallbacks so a GPU extension can ADD a more-specific
# `::CUDABackend` method without *overwriting* a core method (method overwriting
# during extension precompilation is illegal — it silently breaks the GPU path).
const _BACKEND_EXT_MSG = "requires its package extension to be loaded (e.g. `using CUDA`); or use backend(:cpu)."

ka_device(::CPUBackend) = KernelAbstractions.CPU()
ka_device(b::AbstractBackend) = error("ka_device($b) " * _BACKEND_EXT_MSG)

synchronize(b::AbstractBackend) = KernelAbstractions.synchronize(ka_device(b))

array_type(::CPUBackend) = Array
array_type(b::AbstractBackend) = error("array_type($b) " * _BACKEND_EXT_MSG)

# A GPU extension sets this Ref in its `__init__` (avoids overwriting `has_gpu`).
const _GPU_FUNCTIONAL = Ref(false)
has_gpu() = _GPU_FUNCTIONAL[]

zeros_backend(::CPUBackend, ::Type{T}, dims::Integer...) where {T} = zeros(T, Int.(dims)...)
zeros_backend(b::AbstractBackend, ::Type{T}, dims::Integer...) where {T} =
    error("zeros_backend($b) " * _BACKEND_EXT_MSG)

function fill_backend(b::AbstractBackend, value, dims::Integer...)
    A = zeros_backend(b, typeof(value), dims...)
    fill!(A, value)
    return A
end

adapt_backend(::CPUBackend, x) = x
adapt_backend(b::AbstractBackend, x) = error("adapt_backend($b) " * _BACKEND_EXT_MSG)

to_host(x) = x
to_host(x::AbstractArray) = Array(x)
to_host(x::Tuple) = map(to_host, x)
to_host(x::NamedTuple) = NamedTuple{keys(x)}(map(to_host, values(x)))

reduce_sum(x) = isempty(x) ? 0.0 : Float64(sum(Float64, x))
reduce_mean(x) = isempty(x) ? NaN : reduce_sum(x) / length(x)
reduce_count_negative(x) = isempty(x) ? 0 : Int(sum(v -> ifelse(v < zero(v), Int32(1), Int32(0)), x))

function reduce_norm_mean(x, y, z)
    isempty(x) && return NaN
    norms = sqrt.(Float64.(x).^2 .+ Float64.(y).^2 .+ Float64.(z).^2)
    return Float64(sum(norms)) / length(x)
end

any_nonfinite(x) = isempty(x) ? false : !Bool(mapreduce(isfinite, &, x))

function scalar_to_host(x::AbstractArray, inds::Integer...)
    ranges = ntuple(d -> Int(inds[d]):Int(inds[d]), length(inds))
    return first(Array(view(x, ranges...)))
end

function plane_to_host(field::AbstractArray, axis::Symbol, idx::Integer)
    i = Int(idx)
    nd = ndims(field)
    if nd == 1
        axis === :x || throw(ArgumentError("1-D field plane axis must be :x"))
        return Array(@view field[i:i])
    elseif nd == 2
        axis === :x && return Array(@view field[i, :])
        axis === :y && return Array(@view field[:, i])
        throw(ArgumentError("2-D field plane axis must be :x or :y"))
    elseif nd == 3
        axis === :x && return Array(@view field[i, :, :])
        axis === :y && return Array(@view field[:, i, :])
        axis === :z && return Array(@view field[:, :, i])
        throw(ArgumentError("3-D field plane axis must be :x, :y, or :z"))
    end
    throw(ArgumentError("field planes require a 1-D, 2-D, or 3-D array"))
end

is_gpu_backend(::AbstractBackend) = false
is_gpu_backend(::CUDABackend) = true

# Windows-WDDM keep-alive cadence for the hot stepping loops. On consumer GeForce
# cards under the WDDM driver model, free-running kernel submission gets batched by
# the OS and the GPU starves between batch flushes. A bare synchronize() every N steps
# flushes the queue at ~µs cost against a ~100 ms step. Returns the stride in steps
# (0 = never). Default: every step on Windows+CUDA, off elsewhere (the bubble is
# WDDM-specific; on Linux/TCC a per-step sync only adds overhead). Override with the
# MP_SYNC_EVERY environment variable (integer; 0 disables, e.g. for A/B timing).
#
# !!! KNOWN ISSUE (open as of 2026-06-12): a one-off A/B on 2026-06-11 measured
# 140 → 90 s per 1000 steps from this keep-alive, but subsequent production-config
# runs STILL log ~140 s per 1000 steps with the keep-alive ON. The reference script
# runs the same grid at 92.7 s/1000 steps (its H5 metadata/pump_elapsed_s). The
# 90 s/1000 figure has NOT been reproduced — do not treat this sync as the fix for
# the package-vs-reference gap; the root cause is still being hunted (see the native
# Yee port in ext/MagnetoPhotonicCUDAExt.jl for the current lead).
function wddm_sync_every(b::AbstractBackend)
    s = get(ENV, "MP_SYNC_EVERY", "")
    if !isempty(s)
        v = tryparse(Int, s)
        v !== nothing && return max(0, v)
    end
    return (Sys.iswindows() && is_gpu_backend(b)) ? 1 : 0
end

# Optional device introspection/hygiene hooks; the CUDA extension adds ::CUDABackend
# methods. Core fallbacks are inert so CPU runs and GPU-less loads are unaffected.
# device_memory_info: NamedTuple (free_GiB, total_GiB) or nothing.
# reclaim_device_memory!: return freed-but-cached pool pages to the driver — on a
# saturated WDDM card, init-time temporaries held as pool reserve otherwise push
# residency to 100% and the OS pages GPU memory mid-run.
device_memory_info(::AbstractBackend) = nothing
reclaim_device_memory!(::AbstractBackend) = nothing

# Eagerly release one device array's memory back to the allocator pool (the reference
# code's CUDA.unsafe_free! pattern). No-op for host arrays, so callers can free
# uniformly without backend checks. The CUDA extension adds the ::CuArray method.
# After the call the array contents are INVALID — only use on arrays that will never
# be read again (see free_state!).
free_device!(::Any) = nothing
