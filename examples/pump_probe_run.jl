using MagnetoPhotonic

# Coupled FDTD + 4TM + LLB all-optical switching demo on a small CPU-scale grid.
# (The production device is 60 µm and GPU-scale; here we use a thin GdFeCo slab in
# a short waveguide and deposit the pump energy as a hot transient via thermal_kick.)
config = SimConfig(
    grid   = GridConfig(xlim=(0.0, 1.5e-6), ylim=(-0.4e-6, 0.4e-6), zlim=(-0.4e-6, 0.4e-6), dx=0.1e-6, courant=0.3),
    source = SourceConfig(component=:Ez, amplitude=1.0e2, tau=15e-15, t0=45e-15),
    device = DeviceConfig(wg_width=0.3e-6, wg_height=0.3e-6, film_thickness=0.2e-6),
    model  = ModelConfig(multiphysics_subcycle=4),
    steps  = 150,
    precision = Float64,
)

# Tune the model freely — e.g. MagnetoOpticModel(; T_Curie=560.0, Q_voigt_TM=0.025).
model = MagnetoOpticModel()

res = run_pump_probe_sim(config; model=model, thermal_kick=1150.0, relax_steps=8000, relax_dt=1e-15)

m0 = sum(res.m_TM_x0) / length(res.m_TM_x0)
m1 = sum(res.m_TM_x) / length(res.m_TM_x)
r1 = sum(res.m_RE_x) / length(res.m_RE_x)
println("material cells     : ", res.n_material)
println("mean m_TM_x        : ", round(m0, digits=3), "  ->  ", round(m1, digits=3))
println("mean m_RE_x        : ", round(sum(res.m_RE_x0)/length(res.m_RE_x0), digits=3), "  ->  ", round(r1, digits=3))
println("final max Te (K)   : ", round(maximum(res.Te)))
println("switched fraction  : ", round(res.switched_fraction, digits=3))
