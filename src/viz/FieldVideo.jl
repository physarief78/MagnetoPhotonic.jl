# Field visualization. The pure-Julia parts (slice extraction, frame capture) need
# no plotting backend; the actual heatmap video is rendered by the Makie extension,
# which adds methods to `render_field_video`/`plot_scene` declared here.

function plot_scene end

# Extract a 2D slice of a field component from a FieldState for plotting.
function field_slice(fields::FieldState; plane::Symbol=:xy, component::Symbol=:Ez, index::Union{Nothing,Int}=nothing)
    arr = getfield(fields, component)
    Nx, Ny, Nz = size(arr)
    if plane === :xy
        k = index === nothing ? cld(Nz, 2) : index
        return Float64.(arr[:, :, k])
    elseif plane === :xz
        j = index === nothing ? cld(Ny, 2) : index
        return Float64.(arr[:, j, :])
    elseif plane === :yz
        i = index === nothing ? cld(Nx, 2) : index
        return Float64.(arr[i, :, :])
    else
        throw(ArgumentError("plane must be :xy, :xz or :yz"))
    end
end

field_slice(state::FDTDState; kwargs...) = field_slice(state.fields; kwargs...)

# Run a simulation and collect |field| slices every `every` steps (for later video).
function capture_frames(state::FDTDState, steps::Integer; every::Integer=10, plane::Symbol=:xy, component::Symbol=:Ez)
    frames = Matrix{Float64}[]
    run!(state, steps; callback=s -> (s.n % every == 0 && push!(frames, abs.(field_slice(s; plane=plane, component=component)))))
    return frames
end

# Rendering an animation to disk requires a Makie backend (load CairoMakie or GLMakie).
function render_field_video(frames::AbstractVector{<:AbstractMatrix}, filename::AbstractString; kwargs...)
    error("render_field_video requires a Makie backend — `using CairoMakie` (or GLMakie) to load the extension.")
end
