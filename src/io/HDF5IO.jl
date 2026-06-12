using Serialization

function save_state(filename::AbstractString, state)
    open(filename, "w") do io
        serialize(io, state)
    end
    return filename
end

function load_state(filename::AbstractString)
    open(filename, "r") do io
        return deserialize(io)
    end
end

save_hdf5_state(args...; kwargs...) = error("HDF5 support requires the optional HDF5 extension.")
load_hdf5_state(args...; kwargs...) = error("HDF5 support requires the optional HDF5 extension.")
save_magnetization(args...; kwargs...) = error("HDF5 support requires the optional HDF5 extension.")
load_magnetization(args...; kwargs...) = error("HDF5 support requires the optional HDF5 extension.")
load_production_magnetization(args...; kwargs...) = error("HDF5 support requires the optional HDF5 extension.")
write_production_h5(args...; kwargs...) = error("HDF5 support requires the optional HDF5 extension.")
write_goldstd_h5(args...; kwargs...) = error("HDF5 support requires the optional HDF5 extension.")
production_frame_writer(args...; kwargs...) = error("HDF5 support requires the optional HDF5 extension.")
record_production_frame!(args...; kwargs...) = error("HDF5 support requires the optional HDF5 extension.")
probe_frame_writer(args...; kwargs...) = error("HDF5 support requires the optional HDF5 extension.")
record_probe_frame!(args...; kwargs...) = error("HDF5 support requires the optional HDF5 extension.")
close_frame_writer!(args...; kwargs...) = error("HDF5 support requires the optional HDF5 extension.")
h5_schema_signature(args...; kwargs...) = error("HDF5 support requires the optional HDF5 extension.")
assert_h5_schema_compatible(args...; kwargs...) = error("HDF5 support requires the optional HDF5 extension.")

# Public, namespaced run-IO entry points (avoid clobbering Base/FileIO `save`/`load`).
save_run(filename::AbstractString, result; kwargs...) = save_hdf5_state(filename, result; kwargs...)
load_run(filename::AbstractString; kwargs...) = load_hdf5_state(filename; kwargs...)

function load_magnetization!(state, filename::AbstractString; kwargs...)
    data = load_magnetization(filename; kwargs...)
    apply_magnetization!(state, data)
    return state
end
