using MagnetoPhotonic

p = FDTDParams()
pulse = GaussianPulse(; amplitude=1.0, tau=8e-15, t0=32e-15, omega=2pi * p.c0 / 800e-9)
sim = Simulation(; cell=(0.6e-6, 0.4e-6, 0.4e-6), dx=50e-9, dimension=3,
                 sources=[PlaneSource(pulse, :Ez; axis=:x, position=0.15e-6)],
                 boundary=PEC(), courant=0.35, params=p)
run!(sim; until=20e-15)
println("3-D planewave smoke: n=", sim.n, " energy=", field_energy(sim.fields, sim.grid, p))

