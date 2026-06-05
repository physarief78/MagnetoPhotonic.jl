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
