module MagnetoPhotonicHDF5Ext

using MagnetoPhotonic
import HDF5

function MagnetoPhotonic.save_hdf5_state(filename::AbstractString, state; overwrite::Bool=true)
    mode = overwrite ? "w" : "cw"
    HDF5.h5open(filename, mode) do h5
        h5["metadata/n"] = getproperty(state, :n)
        h5["metadata/dt"] = getproperty(state, :dt)
        h5["fields/Ez"] = state.fields.Ez
    end
    return filename
end

function MagnetoPhotonic.load_hdf5_state(filename::AbstractString)
    HDF5.h5open(filename, "r") do h5
        return (n=read(h5["metadata/n"]), dt=read(h5["metadata/dt"]), Ez=read(h5["fields/Ez"]))
    end
end

end
