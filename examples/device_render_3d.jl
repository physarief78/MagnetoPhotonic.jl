using MagnetoPhotonic

device = not_gate_60um()
mkpath(joinpath(@__DIR__, "outputs"))
obj = joinpath(@__DIR__, "outputs", "not_gate_60um.obj")
write_device_obj(obj, device.scene)
println("wrote ", obj)
