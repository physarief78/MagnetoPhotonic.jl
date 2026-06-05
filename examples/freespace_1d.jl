using MagnetoPhotonic

p = FDTDParams()
pulse = GaussianPulse(; amplitude=1.0, tau=8e-15, t0=32e-15, omega=2pi * p.c0 / 800e-9)
sim = Simulation(; cell=(2.0e-6,), dx=5e-9, dimension=1,
                 sources=[PointSource(pulse, :Ez, 0.4e-6)], boundary=PML(20),
                 courant=0.5, params=p)
mon = PointMonitor(:Ez, 1.4e-6)
run!(sim; until=80e-15, monitors=[mon])
println("samples = ", length(mon.t))
println("peak |Ez| = ", maximum(abs, mon.values))

