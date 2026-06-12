# Single-slit diffraction: a 2-D plane wave strikes an opaque (PEC) wall placed in
# the middle of the domain, with a sub-wavelength slit at the center. Behind the
# slit the field spreads into Huygens semicircular wavefronts — the textbook
# demonstration of diffraction. The wall is made perfectly reflecting by forcing
# Ez = 0 in the barrier cells every step (a simple PEC mask).
#
#   julia --project=. examples/slit_diffraction_2d.jl
using MagnetoPhotonic

p  = FDTDParams()
pw = GaussianPulse(; amplitude=1.0, tau=8e-15, t0=24e-15, omega=2pi * p.c0 / 700e-9)

Lx, Ly, dx = 4e-6, 3e-6, 20e-9
yc        = Ly / 2          # wall/slit centered in y
half_slit = 0.20e-6         # slit half-width (0.40 µm ≈ 0.57 λ → strong diffraction)
bx0, bx1  = 1.95e-6, 2.05e-6   # wall spans x ∈ [1.95, 2.05] µm (middle of the grid)

sim = Simulation(; cell=((0.0, Lx), (0.0, Ly)), dx=dx, dimension=2, mode=:TM,
                 sources=[PlaneSource(pw, :Ez; axis=:x, position=0.5e-6)],
                 boundary=PML(10), courant=0.45, params=p)

# Build the PEC mask: barrier cells are everywhere in the wall column EXCEPT the slit.
xs, ys = sim.grid.x.centers, sim.grid.y.centers
mask = falses(length(xs), length(ys))
for i in eachindex(xs), j in eachindex(ys)
    if bx0 <= xs[i] <= bx1 && !(yc - half_slit <= ys[j] <= yc + half_slit)
        mask[i, j] = true
    end
end

run!(sim, 2200; callback = s -> (s.fields.Ez[mask] .= 0.0))

E = Float64.(sim.fields.Ez)
xf = round(Int, 3.2e-6 / dx)              # sample line well behind the wall
jc = round(Int, yc / dx)                  # on-axis (through the slit)
println("grid              = ", size(E))
println("PEC barrier cells = ", count(mask))
println("on-axis |Ez| behind slit = ", round(maximum(abs, @view E[xf, jc-3:jc+3]); sigdigits=3))
println("all finite        = ", all(isfinite, E))
