struct Axis1D
    edges::Vector{Float64}
    centers::Vector{Float64}
    inv_d_cell::Vector{Float64}
    inv_d_dual::Vector{Float64}
    d_min::Float64
end

struct Grid1D
    x::Axis1D
end

struct Grid2D
    x::Axis1D
    y::Axis1D
end

struct Grid3D
    x::Axis1D
    y::Axis1D
    z::Axis1D
end

function _axis_from_edges(edges::Vector{Float64})
    issorted(edges) || throw(ArgumentError("axis edges must be sorted"))
    length(edges) >= 2 || throw(ArgumentError("axis requires at least two edges"))
    widths = diff(edges)
    all(>(0.0), widths) || throw(ArgumentError("axis edges must be strictly increasing"))
    N = length(edges) - 1
    centers = [(edges[i] + edges[i + 1]) / 2.0 for i in 1:N]
    inv_d_cell = 1.0 ./ widths
    inv_d_dual = zeros(Float64, N)
    inv_d_dual[1] = 1.0 / (centers[1] - edges[1])
    for i in 2:N
        inv_d_dual[i] = 1.0 / (centers[i] - centers[i - 1])
    end
    return Axis1D(edges, centers, inv_d_cell, inv_d_dual, min(minimum(widths), 1.0 / maximum(inv_d_dual)))
end

# Sort edges and drop near-coincident ones. Graded builders can leave a floating-point
# sliver cell (e.g. ~1e-22 m) at a region junction; plain `unique!` only removes EXACT
# duplicates, so a degenerate cell survives and collapses the CFL dt. Merge any edge
# within `tol` of its predecessor.
function _dedup_sorted_edges(edges::Vector{Float64}, tol::Float64)
    sort!(edges)
    out = Float64[edges[1]]
    @inbounds for i in 2:length(edges)
        edges[i] - out[end] > tol && push!(out, edges[i])
    end
    out[end] = edges[end]   # preserve the exact domain endpoint
    return out
end

function uniform_axis(xmin::Real, xmax::Real, dx::Real)
    xmax > xmin || throw(ArgumentError("xmax must exceed xmin"))
    dx > 0 || throw(ArgumentError("dx must be positive"))
    n = max(1, round(Int, (Float64(xmax) - Float64(xmin)) / Float64(dx)))
    edges = collect(range(Float64(xmin), Float64(xmax); length=n + 1))
    return _axis_from_edges(edges)
end

function _fill_linear_region!(edges::Vector{Float64}, start_val::Float64, end_val::Float64, target_d::Float64)
    if end_val <= start_val + 1e-12
        return start_val
    end
    dist = end_val - start_val
    n_cells = max(1, round(Int, dist / target_d))
    actual_d = dist / n_cells
    curr = start_val
    for _ in 1:n_cells
        curr += actual_d
        push!(edges, curr)
    end
    return curr
end

function graded_axis(L_phys::Real, d_fine::Real, fine_half_width::Real, center::Real, d_max::Real, stretch_ratio::Real)
    L = Float64(L_phys)
    c = Float64(center)
    edges_right = Float64[c]
    curr = _fill_linear_region!(edges_right, c, c + Float64(fine_half_width), Float64(d_fine))
    curr_d = Float64(d_fine)
    while curr < c + L / 2.0
        curr_d = min(curr_d * Float64(stretch_ratio), Float64(d_max))
        curr += curr_d
        push!(edges_right, curr)
    end
    edges_right[end] = c + L / 2.0

    edges_left = Float64[]
    for i in length(edges_right):-1:2
        push!(edges_left, c - (edges_right[i] - c))
    end
    return _axis_from_edges(_dedup_sorted_edges(vcat(edges_left, edges_right), 1e-3 * Float64(d_fine)))
end

function propagation_axis(L_phys::Real, d_coarse::Real, d_fine::Real, x_film_start::Real, x_film_end::Real, buffer::Real, stretch_ratio::Real)
    L = Float64(L_phys)
    fine_start = clamp(Float64(x_film_start) - Float64(buffer), 0.0, L)
    fine_end = clamp(Float64(x_film_end) + Float64(buffer), 0.0, L)
    if !(fine_start < fine_end)
        mid = clamp(0.5 * (Float64(x_film_start) + Float64(x_film_end)), 0.0, L)
        half = min(Float64(d_fine), 0.5 * L)
        fine_start = clamp(mid - half, 0.0, L)
        fine_end = clamp(mid + half, 0.0, L)
    end
    fine_start < fine_end || throw(ArgumentError("graded propagation axis needs a positive domain"))
    edges = Float64[0.0]

    grade_cells = Float64[]
    d_curr = Float64(d_coarse)
    while d_curr > Float64(d_fine) * 1.001
        d_curr = max(d_curr / Float64(stretch_ratio), Float64(d_fine))
        push!(grade_cells, d_curr)
    end
    grade_total_len = sum(grade_cells)
    x_grade_start = max(0.0, fine_start - grade_total_len)
    curr = _fill_linear_region!(edges, 0.0, x_grade_start, Float64(d_coarse))

    for gc in grade_cells
        curr += gc
        if curr < fine_start
            push!(edges, curr)
        end
    end
    push!(edges, fine_start)
    curr = fine_start

    curr = _fill_linear_region!(edges, curr, Float64(x_film_start), Float64(d_fine))
    curr = _fill_linear_region!(edges, curr, Float64(x_film_end), Float64(d_fine))
    curr = _fill_linear_region!(edges, curr, fine_end, Float64(d_fine))

    d_curr = Float64(d_fine)
    while d_curr < Float64(d_coarse) * 0.999 && curr < L
        d_curr = min(d_curr * Float64(stretch_ratio), Float64(d_coarse))
        curr += d_curr
        push!(edges, min(curr, L))
    end

    _fill_linear_region!(edges, min(curr, L), L, Float64(d_coarse))
    edges[end] = L
    return _axis_from_edges(_dedup_sorted_edges(edges, 1e-3 * Float64(d_fine)))
end

function uniform_grid(xlim, ylim, zlim, d::Real)
    return Grid3D(uniform_axis(xlim[1], xlim[2], d), uniform_axis(ylim[1], ylim[2], d), uniform_axis(zlim[1], zlim[2], d))
end

function uniform_grid(xlim, ylim, zlim, dx::Real, dy::Real, dz::Real)
    return Grid3D(uniform_axis(xlim[1], xlim[2], dx),
                  uniform_axis(ylim[1], ylim[2], dy),
                  uniform_axis(zlim[1], zlim[2], dz))
end

uniform_grid(xlim, d::Real) = Grid1D(uniform_axis(xlim[1], xlim[2], d))
uniform_grid(xlim, ylim, d::Real) = Grid2D(uniform_axis(xlim[1], xlim[2], d), uniform_axis(ylim[1], ylim[2], d))

function _shift_axis(axis::Axis1D, dx::Real)
    return _axis_from_edges(axis.edges .+ Float64(dx))
end

function _bounds_from_region(region)
    region === nothing && return nothing
    if region isa Tuple && length(region) >= 2
        return (Float64(region[1]), Float64(region[2]))
    elseif hasproperty(region, :x_film_start) && hasproperty(region, :x_film_end)
        return (Float64(getproperty(region, :x_film_start)), Float64(getproperty(region, :x_film_end)))
    elseif hasproperty(region, :x0) && hasproperty(region, :x1)
        return (Float64(getproperty(region, :x0)), Float64(getproperty(region, :x1)))
    end
    return nothing
end

function _grid_fine_bounds(cfg::GridConfig, film_region)
    explicit = _bounds_from_region(cfg.fine_region)
    bounds = explicit
    bounds === nothing && (bounds = _bounds_from_region(film_region))
    xlo, xhi = cfg.xlim
    if bounds === nothing || !all(isfinite, bounds) || !(bounds[1] < bounds[2])
        mid = 0.5 * (xlo + xhi)
        bounds = (mid - cfg.fine_dx, mid + cfg.fine_dx)
    end
    lo = clamp(bounds[1], xlo, xhi)
    hi = clamp(bounds[2], xlo, xhi)
    if !(lo < hi)
        mid = clamp(0.5 * (bounds[1] + bounds[2]), xlo, xhi)
        half = min(cfg.fine_dx, 0.5 * (xhi - xlo))
        lo = clamp(mid - half, xlo, xhi)
        hi = clamp(mid + half, xlo, xhi)
    end
    lo < hi || throw(ArgumentError("graded mesh needs a nonzero x domain"))
    return (lo, hi)
end

# ---------------------------------------------------------------------------
# Exact reference NOT-gate mesh.
#
# Verbatim port of the reference driver's grid builders
# (Main_Research/pump_probe_switching_empirical_params.jl: build_propagation_grid /
# build_graded_grid). These reproduce the validated 4099x190x100 graded mesh bit-for-bit
# (identical edges, identical per-axis d_min, hence the identical CFL dt), so the package
# output is shape-exact against the reference raw_v6_* files and the reference's frozen
# magnetization can be ingested by full-grid linear index. Kept separate from the general
# `graded_axis`/`propagation_axis` (which carry sliver-dedup tweaks that shift the smallest
# cell at the 13th digit) so the exact mesh is immune to future changes there.
function _ref_fill_linear_region!(edges::Vector{Float64}, start_val::Float64, end_val::Float64, target_d::Float64)
    if end_val <= start_val + 1e-12
        return start_val
    end
    dist = end_val - start_val
    n_cells = max(1, round(Int, dist / target_d))
    actual_d = dist / n_cells
    curr = start_val
    for _ in 1:n_cells
        curr += actual_d
        push!(edges, curr)
    end
    return curr
end

# Build an Axis1D from explicit edges with the reference's per-axis d_min convention:
# `dual_in_dmin=false` -> primary cell width only (the propagation axis); `true` ->
# min(primary cell, dual cell), as in build_graded_grid.
function _axis_from_ref_edges(edges::Vector{Float64}; dual_in_dmin::Bool)
    N = length(edges) - 1
    centers = [(edges[i] + edges[i + 1]) / 2.0 for i in 1:N]
    widths = diff(edges)
    inv_d_cell = 1.0 ./ widths
    inv_d_dual = zeros(Float64, N)
    inv_d_dual[1] = 1.0 / (centers[1] - edges[1])
    for i in 2:N
        inv_d_dual[i] = 1.0 / (centers[i] - centers[i - 1])
    end
    d_min = dual_in_dmin ? min(minimum(widths), 1.0 / maximum(inv_d_dual)) : minimum(widths)
    return Axis1D(edges, centers, inv_d_cell, inv_d_dual, d_min)
end

function ref_graded_axis(L_phys::Real, d_fine::Real, fine_half_width::Real, center::Real, d_max::Real, stretch_ratio::Real)
    L = Float64(L_phys); c = Float64(center); df = Float64(d_fine); dm = Float64(d_max); sr = Float64(stretch_ratio)
    edges_right = Float64[c]
    curr = _ref_fill_linear_region!(edges_right, c, c + Float64(fine_half_width), df)
    curr_d = df
    while curr < c + L / 2.0
        curr_d = min(curr_d * sr, dm)
        curr += curr_d
        push!(edges_right, curr)
    end
    edges_right[end] = c + L / 2.0
    edges_left = Float64[]
    for i in length(edges_right):-1:2
        push!(edges_left, c - (edges_right[i] - c))
    end
    return _axis_from_ref_edges(vcat(edges_left, edges_right); dual_in_dmin=true)
end

function ref_propagation_axis(L_phys::Real, d_coarse::Real, d_fine::Real, x_film_start::Real, x_film_end::Real, buffer::Real, stretch_ratio::Real)
    L = Float64(L_phys); dc = Float64(d_coarse); df = Float64(d_fine); sr = Float64(stretch_ratio)
    x_fine_start = Float64(x_film_start) - Float64(buffer)
    x_fine_end = Float64(x_film_end) + Float64(buffer)
    edges = Float64[0.0]
    grade_cells = Float64[]
    d_curr = dc
    while d_curr > df * 1.001
        d_curr = max(d_curr / sr, df)
        push!(grade_cells, d_curr)
    end
    x_grade_start = x_fine_start - sum(grade_cells)
    curr = _ref_fill_linear_region!(edges, 0.0, x_grade_start, dc)
    for gc in grade_cells
        curr += gc
        push!(edges, curr)
    end
    edges[end] = x_fine_start
    curr = x_fine_start
    curr = _ref_fill_linear_region!(edges, curr, Float64(x_film_start), df)
    curr = _ref_fill_linear_region!(edges, curr, Float64(x_film_end), df)
    curr = _ref_fill_linear_region!(edges, curr, x_fine_end, df)
    d_curr = df
    while d_curr < dc * 0.999
        d_curr = min(d_curr * sr, dc)
        curr += d_curr
        push!(edges, curr)
    end
    curr = _ref_fill_linear_region!(edges, curr, L, dc)
    edges[end] = L
    unique!(edges); sort!(edges)
    return _axis_from_ref_edges(edges; dual_in_dmin=false)
end

# The exact validated reference NOT-gate mesh (4099x190x100). Parameters are the reference
# driver's hardcoded values (build_simulation_geometry, ref_geo === nothing branch):
#   x: 60 um, d_coarse 15 nm, d_fine 4 nm, film [40, 40.008] um, buffer 200 nm, stretch 1.05
#   y: 6 um centred at 0,   fine half-width 1.2 um @ 20 nm, d_max 150 nm, stretch 1.05
#   z: 3 um centred at 0.2 um, fine half-width 0.5 um @ 20 nm, d_max 120 nm, stretch 1.05
function not_gate_reference_grid()
    x = ref_propagation_axis(60.0e-6, 15e-9, 4e-9, 40.0e-6, 40.008e-6, 200e-9, 1.05)
    y = ref_graded_axis(6.0e-6, 20e-9, 1.2e-6, 0.0, 150e-9, 1.05)
    z = ref_graded_axis(3.0e-6, 20e-9, 0.5e-6, 0.2e-6, 120e-9, 1.05)
    return Grid3D(x, y, z)
end

function grid_from_config(cfg::GridConfig; film_region=nothing)
    cfg.mesh === :reference && return not_gate_reference_grid()
    cfg.mesh in (:uniform, :graded) || throw(ArgumentError("GridConfig.mesh must be :uniform, :graded, or :reference"))
    if cfg.mesh === :uniform
        return uniform_grid(cfg.xlim, cfg.ylim, cfg.zlim, cfg.dx, cfg.dy, cfg.dz)
    end

    x0, x1 = _grid_fine_bounds(cfg, film_region)
    xlo, xhi = cfg.xlim
    buffer = max(cfg.fine_buffer, 2 * cfg.fine_dx)
    ax0 = propagation_axis(xhi - xlo, cfg.dx, cfg.fine_dx, x0 - xlo, x1 - xlo, buffer, cfg.stretch_ratio)
    ax = _shift_axis(ax0, xlo)

    ylo, yhi = cfg.ylim
    zlo, zhi = cfg.zlim
    yaxis = cfg.grade_y ?
            _shift_axis(graded_axis(yhi - ylo, cfg.fine_dy, min(0.5 * (yhi - ylo), cfg.fine_buffer),
                                    0.5 * (yhi - ylo), cfg.dy, cfg.stretch_ratio), ylo) :
            uniform_axis(ylo, yhi, cfg.dy)
    zaxis = cfg.grade_z ?
            _shift_axis(graded_axis(zhi - zlo, cfg.fine_dz, min(0.5 * (zhi - zlo), cfg.fine_buffer),
                                    0.5 * (zhi - zlo), cfg.dz, cfg.stretch_ratio), zlo) :
            uniform_axis(zlo, zhi, cfg.dz)
    return Grid3D(ax, yaxis, zaxis)
end

grid_from_config(cfg::SimConfig; film_region=nothing) = grid_from_config(cfg.grid; film_region=film_region)

dim(::Grid1D) = 1
dim(::Grid2D) = 2
dim(::Grid3D) = 3

Base.size(grid::Grid1D) = (length(grid.x.centers),)
Base.size(grid::Grid2D) = (length(grid.x.centers), length(grid.y.centers))
Base.size(grid::Grid3D) = (length(grid.x.centers), length(grid.y.centers), length(grid.z.centers))

min_spacing(grid::Grid1D) = grid.x.d_min
min_spacing(grid::Grid2D) = min(grid.x.d_min, grid.y.d_min)
min_spacing(grid::Grid3D) = min(grid.x.d_min, grid.y.d_min, grid.z.d_min)

function cfl_dt(grid::Grid1D, p::FDTDParams=FDTDParams(); courant::Real=0.99)
    return Float64(courant) * grid.x.d_min / p.c0
end

function cfl_dt(grid::Grid2D, p::FDTDParams=FDTDParams(); courant::Real=0.99)
    return Float64(courant) / (p.c0 * sqrt(inv(grid.x.d_min)^2 + inv(grid.y.d_min)^2))
end

function cfl_dt(grid::Grid3D, p::FDTDParams=FDTDParams(); courant::Real=0.99)
    return cfl_dt(grid.x.d_min, grid.y.d_min, grid.z.d_min, p; courant=courant)
end
