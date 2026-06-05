using MagnetoPhotonic

device = not_gate_60um()
mkpath(joinpath(@__DIR__, "outputs"))
svg = joinpath(@__DIR__, "outputs", "not_gate_60um_plan.svg")
write_plan_svg(svg, device.scene)
println("wrote ", svg)
