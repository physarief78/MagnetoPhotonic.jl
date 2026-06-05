using Test
using MagnetoPhotonic

@testset "Vec2 and polygons" begin
    a = Vec2(1.0, 2.0)
    b = Vec2(3.0, -1.0)
    @test a + b == Vec2(4.0, 1.0)
    @test dot2(a, b) == 1.0
    poly = generate_waveguide_polygon([Vec2(0.0, 0.0), Vec2(1.0, 0.0)], 0.2)
    @test length(poly) == 4
    @test isapprox(abs(polygon_area(poly)), 0.2; atol=1e-12)
    @test is_inside_polygon(0.5, 0.0, poly)
    @test !is_inside_polygon(0.5, 0.2, poly)
end

@testset "Grid" begin
    ax = uniform_axis(0.0, 1.0, 0.1)
    @test length(ax.centers) == 10
    @test isapprox(ax.d_min, 0.05; atol=1e-12)
    gx = graded_axis(2.0, 0.1, 0.2, 0.0, 0.4, 1.2)
    @test first(gx.edges) < 0.0
    @test last(gx.edges) > 0.0
    grid = uniform_grid((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), 0.25)
    @test cfl_dt(grid) > 0
end

@testset "Raster scene" begin
    grid = uniform_grid((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), 0.25)
    scene = Scene()
    mat = Material("core"; epsr=4.0)
    add_shape!(scene, Box(0.25, 0.75, 0.25, 0.75, 0.25, 0.75), mat)
    geo = rasterize(scene, grid; subpixel=2)
    @test minimum(geo.epsr) >= 1.0
    @test maximum(geo.epsr) > 1.0
    @test isempty(geo.active_indices)
end

@testset "CPML and poles" begin
    p = FDTDParams()
    prof = build_cpml_profiles(20, 4, 1e-8, 1e-17, p)
    @test length(prof.kappa) == 20
    @test prof.b[1] < 1.0
    @test prof.a[10] == 0.0
    pole = create_pole(0.0, 3e14, 1e30, 1e-18, p.eps0)
    @test isfinite(pole.C1)
    @test isfinite(pole.C2)
    @test isfinite(pole.C3)
end

@testset "Magneto-optic model" begin
    model = MagnetoOpticModel(; T_Curie=550.0, Q_voigt_TM=0.025)
    @test model.params.T_Curie == 550.0
    @test optical_coupling(model, 1.0, -1.0) > 0.0
    @test abs(brillouin(1.0, 1e-8)) < 1e-6
    tm, re, Tmin, invdT, nT = build_m_eq_lut(model.params; T_max=20.0, dT=5.0)
    @test length(tm) == nT
    @test Tmin == 1.0
    @test invdT == 0.2
    m2 = cayley_rotate((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 0.1)
    @test isapprox(sqrt(sum(abs2, m2)), 1.0; atol=1e-12)
end

@testset "FDTD smoke" begin
    grid = uniform_grid((0.0, 5e-7), (0.0, 5e-7), (0.0, 5e-7), 1e-7)
    scene = Scene()
    geo = rasterize(scene, grid; subpixel=1)
    p = FDTDParams()
    state = FDTDState(grid, geo; dt=cfl_dt(grid, p; courant=0.1), params=p, T=Float64)
    state.fields.Ez[3, 3, 3] = 1.0
    step!(state)
    @test state.n == 1
    @test isfinite(field_energy(state.fields, grid, p))
end

@testset "CPML structure and absorption" begin
    grid = uniform_grid((0.0, 1.0e-6), (0.0, 1.0e-6), (0.0, 1.0e-6), 5e-8)
    p = FDTDParams()
    dt = cfl_dt(grid, p; courant=0.4)
    c = build_cpml(grid, 5, dt, p)
    mid = cld(length(grid.x.centers), 2)
    @test 0.0 < c.x.b[1] < 1.0          # absorbing at the boundary
    @test c.x.b[mid] == 0.0             # transparent in the bulk
    @test c.x.inv_kappa[mid] == 1.0
    # A short pulse leaves far less energy behind with CPML than with a reflecting wall.
    function final_energy(npml)
        scene = Scene(); geo = rasterize(scene, grid; subpixel=1)
        pulse = GaussianPulse(; amplitude=1.0, tau=3e-15, t0=12e-15, omega=2pi * p.c0 / 800e-9)
        s = FDTDState(grid, geo; dt=dt, params=p, source=(pulse, :Ez, (cld(length(grid.x.centers),2), mid, mid)), T=Float64, n_pml=npml)
        run!(s, 1200)
        field_energy(s.fields, grid, p)
    end
    e_cpml = final_energy(6)
    e_wall = final_energy(0)
    @test isfinite(e_cpml)
    @test e_cpml < 0.5 * e_wall
end

@testset "LLB all-optical switching" begin
    model = MagnetoOpticModel()
    gd = model.params
    tm_lut, re_lut, Tmin, invdT, lutN = build_m_eq_lut(gd)
    mtm = lookup_m_eq_lut(tm_lut, gd.T0, Tmin, invdT, lutN)
    mre = lookup_m_eq_lut(re_lut, gd.T0, Tmin, invdT, lutN)
    @test mtm > 0.9 && mre < -0.9        # antiparallel ferrimagnet at 300 K
    mTM = (mtm, 0.0, 1e-4); mRE = (mre, 0.0, 1e-4)
    dt = 0.5e-15
    for n in 1:6000
        T = n < 1700 ? 1200.0 : max(300.0, 1200.0 - (n - 1700) * 0.4)
        r = llb_step(mTM..., mRE..., T, T, T, gd, tm_lut, re_lut, Tmin, invdT, lutN, dt, 2)
        mTM = (r[1], r[2], r[3]); mRE = (r[4], r[5], r[6])
    end
    @test mTM[1] < -0.8                  # FeCo reversed and rebuilt
    @test mRE[1] > 0.8                   # Gd reversed and rebuilt
    @test sqrt(sum(abs2, mTM)) <= 0.992 + 1e-9
end

@testset "4TM relaxes to ambient" begin
    model = MagnetoOpticModel()
    th = ThermalState(4, model)
    fill!(th.Te, 1500.0); fill!(th.Tl, 1500.0); fill!(th.Ts_TM, 1500.0); fill!(th.Ts_RE, 1500.0)
    pabs = zeros(4)
    for _ in 1:200000
        thermal_step!(th, pabs, model, 1e-15)
    end
    @test all(t -> 290 < t < 360, th.Tl)   # lattice cools toward T0 = 300 K
    @test all(isfinite, th.Te)
end

@testset "Coupled pump-probe driver switches" begin
    cfg = SimConfig(
        grid=GridConfig(xlim=(0.0, 1.5e-6), ylim=(-0.4e-6, 0.4e-6), zlim=(-0.4e-6, 0.4e-6), dx=0.1e-6, courant=0.3),
        source=SourceConfig(component=:Ez, amplitude=1e2, tau=15e-15, t0=45e-15),
        device=DeviceConfig(wg_width=0.3e-6, wg_height=0.3e-6, film_thickness=0.2e-6),
        model=ModelConfig(multiphysics_subcycle=4),
        steps=120, precision=Float64,
    )
    res = run_pump_probe_sim(cfg; model=MagnetoOpticModel(), thermal_kick=1150.0, relax_steps=6000, relax_dt=1e-15)
    @test res.n_material > 0
    @test isfinite(last(res.energies))
    @test all(isfinite, res.m_TM_x)
    @test res.switched_fraction > 0.9
    @test maximum(res.Te) < 400.0          # cooled below Curie after relax
end

@testset "Convergence study" begin
    cv = run_convergence_study(; dxs=(50e-9, 35e-9, 22e-9), L=1.0e-6, T_max=90e-15, n_pml=5)
    @test all(isfinite, cv.cauchy_l2)
    @test cv.cauchy_l2[end] < cv.cauchy_l2[1]   # converging with refinement
    cvd = run_convergence_study_3D_dispersive(; dxs=(50e-9, 35e-9), L=1.0e-6, T_max=90e-15, n_pml=5)
    @test cvd.spectrum !== nothing
    peak_THz = cvd.spectrum.freq[argmax(cvd.spectrum.amplitude)] / 1e12
    @test 300 < peak_THz < 460             # central frequency near c/800nm = 375 THz
end

@testset "Field slice and dispersion fit" begin
    grid = uniform_grid((0.0, 4e-7), (0.0, 4e-7), (0.0, 4e-7), 1e-7)
    scene = Scene(); geo = rasterize(scene, grid; subpixel=1)
    st = FDTDState(grid, geo; dt=cfl_dt(grid; courant=0.2), T=Float64)
    sl = field_slice(st; plane=:xy, component=:Ez)
    @test size(sl) == (length(grid.x.centers), length(grid.y.centers))
    # ADE pole set reproduces the target permittivity at the fit frequency.
    gd = MagnetoOpticModel().params
    p = FDTDParams()
    dt = 1e-17
    poles = build_pump_poles(dt, p.eps0, gd)
    chi = sum(discrete_pole_chi(pl, dt, p.eps0, gd.omega_pump) for pl in poles)
    eps_fit = 1.0 + chi
    @test isapprox(real(eps_fit), gd.eps_real_pump; rtol=0.05)
    @test isapprox(imag(eps_fit), gd.eps_imag_pump; rtol=0.05)
end

@testset "General EM materials and grids" begin
    med = Medium(; index=2.1, name="test")
    @test isapprox(med.epsr, 2.1^2)
    mat = medium_material(med)
    @test mat.name == "test"
    @test isapprox(mat.epsr, med.epsr)
    g1 = uniform_grid((0.0, 1.0), 0.1)
    g2 = uniform_grid((0.0, 1.0), (0.0, 0.5), 0.1)
    @test dim(g1) == 1
    @test dim(g2) == 2
    @test cfl_dt(g1) > cfl_dt(g2)
end

@testset "General EM 1D/2D/3D smoke" begin
    p = FDTDParams()
    pulse = GaussianPulse(; amplitude=1.0, tau=2e-15, t0=8e-15, omega=0.0, phase=pi/2)
    s1 = Simulation(; cell=(6e-7,), dx=1e-8, dimension=1,
                    sources=[PointSource(pulse, :Ez, 2e-7)], boundary=PML(6),
                    courant=0.45, params=p)
    mon1 = PointMonitor(:Ez, 3e-7)
    run!(s1, 20; monitors=[mon1])
    @test s1.n == 20
    @test length(mon1.values) == 20
    @test fieldtype(typeof(s1), :grid) === typeof(s1.grid)
    @test fieldtype(typeof(s1), :fields) === typeof(s1.fields)
    @test fieldtype(typeof(s1), :cpml) === typeof(s1.cpml)
    @test isfinite(field_energy(s1.fields, s1.grid, p))

    s2 = Simulation(; cell=(5e-7, 4e-7), dx=5e-8, dimension=2, mode=:TM,
                    sources=[PlaneSource(pulse, :Ez; axis=:x, position=1e-7)],
                    boundary=PEC(), courant=0.35, params=p)
    run!(s2, 5)
    i = 3
    @test maximum(abs.(s2.fields.Ez[i, 2:end-1] .- s2.fields.Ez[i, 2])) < 1e-6
    @test isfinite(field_energy(s2.fields, s2.grid, p))

    s3 = Simulation(; cell=(4e-7, 4e-7, 4e-7), dx=1e-7, dimension=3,
                    sources=[PointSource(pulse, :Ez, (2e-7, 2e-7, 2e-7))],
                    boundary=PEC(), courant=0.2, params=p)
    run!(s3, 2)
    @test s3.n == 2
    @test isfinite(field_energy(s3.fields, s3.grid, p))
end

@testset "General EM acceptance: CPML, ADE, flux, convergence" begin
    p = FDTDParams()
    carrier = GaussianPulse(; amplitude=1.0, tau=2e-15, t0=8e-15, omega=2pi * p.c0 / 800e-9)

    function residual_ratio(sim, steps)
        peak = 0.0
        for _ in 1:steps
            step!(sim)
            peak = max(peak, field_energy(sim.fields, sim.grid, p))
        end
        return field_energy(sim.fields, sim.grid, p) / peak
    end

    s1_pml = Simulation(; cell=(6e-7,), dx=1e-8, dimension=1,
                        sources=[PointSource(carrier, :Ez, 3e-7)], boundary=PML(8),
                        courant=0.45, params=p)
    s1_pec = Simulation(; cell=(6e-7,), dx=1e-8, dimension=1,
                        sources=[PointSource(carrier, :Ez, 3e-7)], boundary=PEC(),
                        courant=0.45, params=p)
    r1_pml = residual_ratio(s1_pml, 1600)
    r1_pec = residual_ratio(s1_pec, 1600)
    @test r1_pml < 0.05
    @test r1_pml < 0.1 * r1_pec

    s2_pml = Simulation(; cell=(6e-7, 6e-7), dx=2e-8, dimension=2, mode=:TM,
                        sources=[PointSource(carrier, :Ez, (3e-7, 3e-7))],
                        boundary=PML(6), courant=0.35, params=p)
    s2_pec = Simulation(; cell=(6e-7, 6e-7), dx=2e-8, dimension=2, mode=:TM,
                        sources=[PointSource(carrier, :Ez, (3e-7, 3e-7))],
                        boundary=PEC(), courant=0.35, params=p)
    r2_pml = residual_ratio(s2_pml, 1600)
    r2_pec = residual_ratio(s2_pec, 1600)
    @test r2_pml < 1e-3
    @test r2_pml < 0.01 * r2_pec

    s3_pml = Simulation(; cell=(1.0e-6, 1.0e-6, 1.0e-6), dx=1e-7, dimension=3,
                        sources=[PointSource(carrier, :Ez, (5e-7, 5e-7, 5e-7))],
                        boundary=PML(3), courant=0.35, params=p)
    s3_pec = Simulation(; cell=(1.0e-6, 1.0e-6, 1.0e-6), dx=1e-7, dimension=3,
                        sources=[PointSource(carrier, :Ez, (5e-7, 5e-7, 5e-7))],
                        boundary=PEC(), courant=0.35, params=p)
    r3_pml = residual_ratio(s3_pml, 1000)
    r3_pec = residual_ratio(s3_pec, 1000)
    @test r3_pml < 1e-4
    @test r3_pml < 0.01 * r3_pec

    grid3 = uniform_grid((0.0, 4e-7), (0.0, 4e-7), (0.0, 4e-7), 1e-7)
    dt3 = cfl_dt(grid3, p; courant=0.2)
    pole = create_pole(0.0, 3e14, 1e30, dt3, p.eps0)
    mat = Material("disp"; epsr=1.0, poles=[pole])
    scene = Scene()
    add_shape!(scene, Box(1e-7, 3e-7, 1e-7, 3e-7, 1e-7, 3e-7), mat)
    s3_disp = Simulation(; cell=(4e-7, 4e-7, 4e-7), dx=1e-7, geometry=scene,
                         dimension=3, sources=[PointSource(carrier, :Ez, (2e-7, 2e-7, 2e-7))],
                         boundary=PEC(), courant=0.2, params=p)
    @test !isempty(s3_disp.active_indices)
    @test s3_disp.ade !== nothing
    run!(s3_disp, 5)
    @test maximum(abs.(s3_disp.ade.z.P)) > 0.0
    @test isfinite(field_energy(s3_disp.fields, s3_disp.grid, p))

    sflux = Simulation(; cell=(3.0, 2.0), dx=1.0, dimension=2, mode=:TM,
                       sources=AbstractEMSource[], boundary=PEC(), params=p)
    fill!(sflux.fields.Ez, 2.0)
    fill!(sflux.fields.Hy, -3.0)
    fmon = FluxMonitor(:x, 1.5)
    record!(fmon, sflux)
    @test isapprox(fmon.flux[end], 12.0; atol=1e-12)

    pmon = PointMonitor(:Ez, (1.0, 0.5))
    sflux.fields.Ez[1, 1] = 1.0
    sflux.fields.Ez[2, 1] = 3.0
    record!(pmon, sflux)
    @test isapprox(pmon.values[end], 2.0; atol=1e-12)

    eta = sqrt(p.mu0 / p.eps0)
    L = 4e-6
    x0 = 1.0e-6
    sigma = 0.25e-6
    Tprop = 3e-15
    function smooth_wave_error(dx)
        grid = uniform_grid((0.0, L), dx)
        fields = allocate_fields(grid)
        dt = cfl_dt(grid, p; courant=0.45)
        for i in eachindex(grid.x.centers)
            E = exp(-((grid.x.centers[i] - x0) / sigma)^2)
            fields.Ez[i] = E
            fields.Dz[i] = p.eps0 * E
            xh = i < length(grid.x.centers) ? grid.x.edges[i + 1] : grid.x.edges[end]
            fields.Hy[i] = -exp(-((xh + p.c0 * dt / 2 - x0) / sigma)^2) / eta
        end
        nsteps = round(Int, Tprop / dt)
        for _ in 1:nsteps
            update_H_1d!(fields, grid, p, dt)
            update_E_1d!(fields, grid, p, dt, ones(length(grid.x.centers)))
        end
        tfinal = nsteps * dt
        err = 0.0
        norm = 0.0
        for i in eachindex(grid.x.centers)
            exact = exp(-((grid.x.centers[i] - (x0 + p.c0 * tfinal)) / sigma)^2)
            dxcell = grid.x.edges[i + 1] - grid.x.edges[i]
            err += (fields.Ez[i] - exact)^2 * dxcell
            norm += exact^2 * dxcell
        end
        return sqrt(err / norm)
    end
    errs = smooth_wave_error.((40e-9, 20e-9, 10e-9))
    orders = (log(errs[1] / errs[2]) / log(2), log(errs[2] / errs[3]) / log(2))
    @test all(>(1.9), orders)
end

@testset "General EM convergence API" begin
    cv1 = convergence_study(; dimension=1, dxs=(60e-9, 40e-9), L=8e-7, T_max=20e-15, nsample=40)
    @test cv1.dimension == 1
    @test length(cv1.cauchy_l2) == 1
    @test all(isfinite, cv1.cauchy_l2)
    cv2 = convergence_study(; dimension=2, mode=:TE, pml=true, n_pml=3,
                            dxs=(80e-9, 50e-9), L=6e-7, T_max=15e-15, nsample=30)
    @test cv2.mode == :TE
    @test cv2.pml
    @test all(isfinite, cv2.final_energy)
    cvd = convergence_study(; dimension=1, dispersive=true, dxs=(60e-9, 40e-9),
                            L=8e-7, T_max=20e-15, nsample=40)
    @test cvd.spectrum !== nothing
end
