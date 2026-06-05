# EM-FDTD Tutorial

The general electromagnetic FDTD layer works identically in 1-D, 2-D and 3-D through the
same `Simulation` / `run!` API. All outputs below are produced on CPU with
Julia 1.12.

## 1-D: pulse propagation and CPML absorption

```julia
using MagnetoPhotonic
p = FDTDParams()
pulse = GaussianPulse(; amplitude=1.0, tau=4e-15, t0=16e-15, omega=2pi*p.c0/800e-9)

sim = Simulation(; cell=(4e-6,), dx=10e-9, dimension=1,
                 sources=[PointSource(pulse, :Ez, 2e-6)], boundary=PML(20), courant=0.5)
mon = PointMonitor(:Ez, 3e-6)
run!(sim; until=120e-15, monitors=[mon])
```

```text
steps=14390   peak |Ez| @ probe = 1.942   energy final/peak = 3.58e-22
```

`PML(20)` is a 20-cell CFS-CPML layer. After the pulse exits, the residual domain energy is
~`1e-22` of the peak — the boundary is genuinely absorbing (with `PEC()` the energy would be
retained and bounce indefinitely).

## 2-D: TM/TE modes and plane-wave sources

A `PlaneSource` injects a transverse current sheet (a line in 2-D), launching a plane
wave normal to `axis`:

```julia
pw = GaussianPulse(; amplitude=1.0, tau=10e-15, t0=40e-15, omega=2pi*p.c0/800e-9)
sim = Simulation(; cell=(2e-6, 1.5e-6), dx=20e-9, dimension=2, mode=:TM,
                 sources=[PlaneSource(pw, :Ez; axis=:x, position=0.3e-6)],
                 boundary=PML(10), courant=0.4)
frames = FieldMonitor(:Ez; every=25)
run!(sim; until=80e-15, monitors=[frames])
```

```text
grid = (100, 75)   dt = 9.43 as   frames captured = 339   final energy = 9.15e-32
```

- `mode=:TM` evolves `(Ez, Hx, Hy)`; `mode=:TE` evolves `(Hz, Ex, Ey)`.
- A `FieldMonitor` stores field slices every `every` steps, e.g. for an animation. With the
  optional `CairoMakie` extension, `render_field_video(frames.frames, "out.mp4")` writes a video.

## 3-D: full Yee with CPML

```julia
sp = GaussianPulse(; amplitude=1.0, tau=2e-15, t0=8e-15, omega=2pi*p.c0/800e-9)
sim = Simulation(; cell=(1e-6, 1e-6, 1e-6), dx=40e-9, dimension=3,
                 sources=[PointSource(sp, :Ez, (0.5e-6, 0.5e-6, 0.5e-6))],
                 boundary=PML(8), courant=0.35)
run!(sim, 4000)
```

```text
grid = (25, 25, 25)   energy final/peak = 8.9e-10
```

## Materials and geometry

Build a `Scene` of shapes, each paired with a `Material`. Materials accept a
refractive `index`, an explicit `epsr`, or Drude–Lorentz `poles` for dispersion.

```julia
scene = Scene()
add_shape!(scene, Box(1e-6, 2e-6, -1, 1, -1, 1), Material("n=2 slab"; index=2.0))

sim = Simulation(; cell=(3e-6,), dx=5e-9, dimension=1, geometry=scene,
                 sources=[PointSource(pulse, :Ez, 0.4e-6)], boundary=PML(20))
```

Shape primitives: `Box`, `PolygonShape`, `Waveguide`, `TaperedWaveguide`, `Cylinder`,
`Letter`. A device library is included — `not_gate_60um`, `passive_waveguide`,
`hm_test_pattern` — with OBJ/SVG export:

```julia
dev = not_gate_60um()
write_device_obj("device.obj", dev.scene)
write_plan_svg("device.svg", dev.scene)
```

!!! tip "Dispersive materials"
    A `Material(...; poles=…)` adds an ADE pole response on top of the static background
    `epsr`. When the fitted poles already represent the full dielectric response relative to
    vacuum, use `epsr=1.0`; otherwise set `epsr` to the intended instantaneous background.

## Boundaries

| Boundary | Behavior |
|---|---|
| `PML(n)` | `n`-cell CFS-CPML, ψ-convolution form, in 1-D/2-D/3-D |
| `PEC()` | perfect electric conductor (reflecting walls) |
| `Periodic()` | post-update edge copy (simple wraparound; not a Bloch stencil) |

## Monitors

| Monitor | Records |
|---|---|
| `PointMonitor(component, pos)` | interpolated field time-trace at a physical point |
| `FieldMonitor(component; every=n)` | field slices for animation |
| `FluxMonitor(axis, pos)` | signed, plane-integrated Poynting flux through a normal plane |
| `DFTMonitor(component, pos)` | trace + on-demand spectrum via `compute_spectrum` |

## Validation

### Second-order accuracy

Propagating a smooth Gaussian field with no source and comparing to the exact translate
recovers the expected O(Δx²) Yee convergence:

```text
dx (nm) = [40, 20, 10]
L2 err  = [0.0141, 0.00353, 0.000883]
orders  = (2.00, 2.00)
```

### Dispersion

A Drude–Lorentz slab driven at 800 nm reproduces the correct transmitted spectral peak:

```julia
cv = convergence_study(; dimension=2, mode=:TM, dispersive=true, pml=true,
                       dxs=(50e-9, 35e-9), L=1e-6, T_max=90e-15)
cv.spectrum.freq[argmax(cv.spectrum.amplitude)] / 1e12     # → 377.8  (THz; 800 nm ≈ 375 THz)
```

The `convergence_study` driver reproduces the lab's `Convergence_study` methodology
(probe-trace self-convergence + transmitted spectrum) across dimensions and boundary types;
ready-to-run scripts live in `studies/`.
