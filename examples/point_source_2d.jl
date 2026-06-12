# Point source in free space: a single cell at the center of the domain drives Ez
# with a sine wave modulated by a Gaussian envelope (`GaussianPulse`). The result
# is an expanding train of cylindrical wavefronts — a "ripple in a pond" — that is
# cleanly swallowed by the CPML boundary on all sides.
#
#   julia --project=. examples/point_source_2d.jl
using MagnetoPhotonic

p  = FDTDParams()
# tau = 20 fs ≈ 7-8 optical cycles at 700 nm, so several wavefronts radiate out.
pw = GaussianPulse(; amplitude=1.0, tau=20e-15, t0=30e-15, omega=2pi * p.c0 / 700e-9)

L, dx = 4e-6, 20e-9
sim = Simulation(; cell=((0.0, L), (0.0, L)), dx=dx, dimension=2, mode=:TM,
                 sources=[PointSource(pw, :Ez, (L / 2, L / 2))],   # source in the MIDDLE
                 boundary=PML(10), courant=0.45, params=p)

# A FieldMonitor stores |Ez| slices for an animation (every 25 steps here).
frames = FieldMonitor(:Ez; every=25)
run!(sim, 1300; monitors=[frames])

E = Float64.(sim.fields.Ez)
println("grid           = ", size(E))
println("frames captured = ", length(frames.frames))
println("peak |Ez|      = ", round(maximum(abs, E); sigdigits=3))
println("final energy   = ", field_energy(sim.fields, sim.grid, p))
