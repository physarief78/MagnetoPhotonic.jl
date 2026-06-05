using MagnetoPhotonic

p = FDTDParams()
scene = Scene()
add_shape!(scene, Box(1.0e-6, 2.0e-6, -1.0, 1.0, -1.0, 1.0), Material("n=2 slab"; index=2.0))
pulse = GaussianPulse(; amplitude=1.0, tau=8e-15, t0=32e-15, omega=2pi * p.c0 / 800e-9)
sim = Simulation(; cell=(2.0e-6,), dx=5e-9, dimension=1, geometry=scene,
                 sources=[PointSource(pulse, :Ez, 0.35e-6)], boundary=PML(20),
                 courant=0.5, params=p)
mon = PointMonitor(:Ez, 0.75e-6)
run!(sim; until=80e-15, monitors=[mon])
println("interface demo samples = ", length(mon.t))
println("trace rms = ", sqrt(sum(abs2, mon.values) / length(mon.values)))

