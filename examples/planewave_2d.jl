using MagnetoPhotonic

p = FDTDParams()
pulse = GaussianPulse(; amplitude=1.0, tau=10e-15, t0=40e-15, omega=2pi * p.c0 / 800e-9)
sim = Simulation(; cell=(1.2e-6, 0.8e-6), dx=20e-9, dimension=2, mode=:TM,
                 sources=[PlaneSource(pulse, :Ez; axis=:x, position=0.25e-6)],
                 boundary=PML(8), courant=0.45, params=p)
frames = FieldMonitor(:Ez; every=10)
run!(sim; until=60e-15, monitors=[frames])
println("captured frames = ", length(frames.frames))
println("final energy = ", field_energy(sim.fields, sim.grid, p))

