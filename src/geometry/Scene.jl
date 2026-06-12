Base.@kwdef struct Material
    name::String
    epsr::Float64 = 1.0
    poles::Vector{Any} = Any[]
    model::Any = nothing
    color::Symbol = :gray
end

function Material(name::AbstractString; epsr::Union{Nothing,Real}=nothing, index::Real=1.0,
                  poles=Any[], model=nothing, color::Symbol=:gray)
    er = epsr === nothing ? Float64(index)^2 : Float64(epsr)
    return Material(String(name), er, Any[poles...], model, color)
end

function medium_material(m::Medium; name::AbstractString=m.name, model=nothing)
    return Material(name; epsr=m.epsr, poles=m.poles, model=model, color=m.color)
end

struct SceneEntry
    shape::AbstractShape
    material::Material
end

mutable struct Scene
    entries::Vector{SceneEntry}
end

Scene() = Scene(SceneEntry[])

function add_shape!(scene::Scene, shape::AbstractShape, material::Material)
    push!(scene.entries, SceneEntry(shape, material))
    return scene
end

function _contains_xy(shape::Box, x, y; subpixel::Integer=1, xlo=x, xhi=x, ylo=y, yhi=y)
    if subpixel <= 1
        return shape.xmin <= x <= shape.xmax && shape.ymin <= y <= shape.ymax ? 1.0 : 0.0
    end
    hits = 0
    total = subpixel^2
    dx = (xhi - xlo) / subpixel
    dy = (yhi - ylo) / subpixel
    for a in 0:(subpixel - 1), b in 0:(subpixel - 1)
        xs = xlo + (a + 0.5) * dx
        ys = ylo + (b + 0.5) * dy
        hits += shape.xmin <= xs <= shape.xmax && shape.ymin <= ys <= shape.ymax ? 1 : 0
    end
    return hits / total
end

function _contains_xy(shape::Cylinder, x, y; kwargs...)
    dx = x - shape.center.x
    dy = y - shape.center.y
    return dx * dx + dy * dy <= shape.radius^2 ? 1.0 : 0.0
end

function _contains_xy(shape::Union{PolygonShape,Waveguide,TaperedWaveguide,Letter}, x, y; subpixel::Integer=1, xlo=x, xhi=x, ylo=y, yhi=y)
    poly = polygon(shape)
    if subpixel <= 1
        return is_inside_polygon(x, y, poly) ? 1.0 : 0.0
    end
    return fill_fraction(poly, xlo, xhi, ylo, yhi; subpixel=subpixel)
end

function _z_fraction(shape::AbstractShape, zlo, zhi)
    zmin, zmax = zrange(shape)
    overlap = min(zhi, zmax) - max(zlo, zmin)
    return clamp(overlap / (zhi - zlo), 0.0, 1.0)
end

# Polygon membership using a PRE-COMPUTED polygon (identical result to
# `_contains_xy(::PolygonShape/Waveguide/…)`, but without regenerating the polygon on
# every cell — the regeneration was the rasterizer's dominant cost / allocation source).
function _contains_xy_poly(poly::Vector{Vec2}, x, y; subpixel::Integer=1, xlo=x, xhi=x, ylo=y, yhi=y)
    if subpixel <= 1
        return is_inside_polygon(x, y, poly) ? 1.0 : 0.0
    end
    return fill_fraction(poly, xlo, xhi, ylo, yhi; subpixel=subpixel)
end

_shape_poly(shape::Union{PolygonShape,Waveguide,TaperedWaveguide,Letter}) = polygon(shape)
_shape_poly(::AbstractShape) = nothing

# Subpixel film fill of a cell-sized window centred at (cx, cy) — used for Yee-staggered
# sampling of a model film at an E-component's transverse position.
function _window_fill(poly::Vector{Vec2}, cx, cy, dx, dy, subpixel::Integer)
    subpixel <= 1 && return is_inside_polygon(cx, cy, poly) ? 1.0 : 0.0
    return fill_fraction(poly, cx - dx / 2, cx + dx / 2, cy - dy / 2, cy + dy / 2; subpixel=subpixel)
end

# Max model-film fill over the three staggered Yee E-field positions of cell (i,j,k):
#   Ex at (x_centre, y_centre, z_edge),  Ey at (x_edge, y_edge, z_edge),
#   Ez at (x_edge, y_centre, z_centre).
# This reproduces the reference driver's per-component active-cell test; the union of cells
# with a nonzero result is the magnetization grid (act_idx_all), which on the exact
# reference mesh is the validated 1323-cell set rather than the cell-centred 800.
function _model_stagger_fill(poly::Vector{Vec2}, zmn::Real, zmx::Real, grid, i::Integer, j::Integer, k::Integer, subpixel::Integer)
    xc = grid.x.centers[i]; dx = grid.x.edges[i + 1] - grid.x.edges[i]
    yc = grid.y.centers[j]; dy = grid.y.edges[j + 1] - grid.y.edges[j]
    zlo = grid.z.edges[k]; zhi = grid.z.edges[k + 1]; dz = zhi - zlo
    xe = xc - dx / 2; ye = yc + dy / 2
    zf_edge = clamp((min(zlo + dz / 2, zmx) - max(zlo - dz / 2, zmn)) / dz, 0.0, 1.0)
    zf_cell = clamp((min(zhi, zmx) - max(zlo, zmn)) / dz, 0.0, 1.0)
    fx = _window_fill(poly, xc, yc, dx, dy, subpixel) * zf_edge
    fy = _window_fill(poly, xe, ye, dx, dy, subpixel) * zf_edge
    fz = _window_fill(poly, xe, yc, dx, dy, subpixel) * zf_cell
    return max(fx, fy, fz)
end

function _shape_xybounds(shape, poly)
    if poly !== nothing
        b = get_bbox(poly)
        return (b.xmin, b.xmax, b.ymin, b.ymax)
    elseif shape isa Box
        return (shape.xmin, shape.xmax, shape.ymin, shape.ymax)
    elseif shape isa Cylinder
        return (shape.center.x - shape.radius, shape.center.x + shape.radius,
                shape.center.y - shape.radius, shape.center.y + shape.radius)
    end
    return (-Inf, Inf, -Inf, Inf)
end

# Inclusive cell-index range whose cells [edges[i], edges[i+1]] overlap [lo, hi], with a
# one-cell margin so sub-pixel partials at the bbox edge are not dropped. Empty range
# (i0 > i1) when the shape misses the axis entirely.
function _cell_index_range(axis, lo::Real, hi::Real)
    N = length(axis.centers)
    e = axis.edges
    (Float64(lo) > e[end] || Float64(hi) < e[1]) && return (1, 0)
    i0 = clamp(searchsortedlast(e, Float64(lo)), 1, N)
    i1 = clamp(searchsortedfirst(e, Float64(hi)) - 1, 1, N)
    return (max(1, i0 - 1), min(N, i1 + 1))
end

function rasterize(scene::Scene, grid; subpixel::Integer=4, model_stagger::Bool=false,
                   background::Material=Material("vacuum"; epsr=1.0))
    Nx, Ny, Nz = length(grid.x.centers), length(grid.y.centers), length(grid.z.centers)
    epsr = fill(background.epsr, Nx, Ny, Nz)
    material_id = zeros(Int, Nx, Ny, Nz)
    fill_fraction_grid = zeros(Float64, Nx, Ny, Nz)
    materials = [background; [entry.material for entry in scene.entries]]
    lin = LinearIndices((Nx, Ny, Nz))
    active_indices = Int[]
    # Cells carrying a magneto-optic model: these host the magnetization / 4TM
    # state and the dispersive + gyration ADE. Keyed by linear index so overlapping
    # entries collapse to one material cell (largest fill wins).
    model_cells = Dict{Int,Tuple{Int,Int,Int,Float64}}()  # lin => (i,j,k,fill)
    pole_cells = Dict{Int,Tuple{Int,Int,Int,Float64}}()
    pole_poles = Any[]

    for (entry_id, entry) in enumerate(scene.entries)
        material = entry.material
        is_model = material.model !== nothing
        # Precompute the polygon (once) and restrict the sweep to the shape's bounding box;
        # z-overlap depends only on k so it is hoisted out of the (i,j) loop.
        poly = _shape_poly(entry.shape)
        bx0, bx1, by0, by1 = _shape_xybounds(entry.shape, poly)
        zmn, zmx = zrange(entry.shape)
        i0, i1 = _cell_index_range(grid.x, bx0, bx1)
        j0, j1 = _cell_index_range(grid.y, by0, by1)
        k0, k1 = _cell_index_range(grid.z, zmn, zmx)
        # Yee-staggered active selection applies only to model (film) materials, and only
        # when requested; it can activate a cell even where the cell-centred z-overlap is
        # zero (e.g. the z=0 boundary), so the cell-z skip is bypassed in that case.
        stagger = model_stagger && is_model && poly !== nothing
        for k in k0:k1
            zf = _z_fraction(entry.shape, grid.z.edges[k], grid.z.edges[k + 1])
            (zf > 0.0 || stagger) || continue
            for j in j0:j1, i in i0:i1
                f = if stagger
                    _model_stagger_fill(poly, zmn, zmx, grid, i, j, k, subpixel)
                else
                    xf = poly === nothing ?
                        _contains_xy(entry.shape, grid.x.centers[i], grid.y.centers[j];
                                     subpixel=subpixel, xlo=grid.x.edges[i], xhi=grid.x.edges[i + 1],
                                     ylo=grid.y.edges[j], yhi=grid.y.edges[j + 1]) :
                        _contains_xy_poly(poly, grid.x.centers[i], grid.y.centers[j];
                                          subpixel=subpixel, xlo=grid.x.edges[i], xhi=grid.x.edges[i + 1],
                                          ylo=grid.y.edges[j], yhi=grid.y.edges[j + 1])
                    xf * zf
                end
                if f > 0.0
                    epsr[i, j, k] = (1.0 - f) * epsr[i, j, k] + f * material.epsr
                    material_id[i, j, k] = entry_id + 1
                    fill_fraction_grid[i, j, k] = max(fill_fraction_grid[i, j, k], f)
                    if !isempty(material.poles) || is_model
                        push!(active_indices, lin[i, j, k])
                    end
                    if is_model
                        li = lin[i, j, k]
                        prev = get(model_cells, li, (i, j, k, 0.0))
                        model_cells[li] = (i, j, k, max(prev[4], f))
                    end
                    if !isempty(material.poles)
                        li = lin[i, j, k]
                        prev = get(pole_cells, li, (i, j, k, 0.0))
                        pole_cells[li] = (i, j, k, max(prev[4], f))
                        isempty(pole_poles) && append!(pole_poles, material.poles)
                    end
                end
            end
        end
    end

    inv_eps = 1.0 ./ epsr

    # Flatten the magneto-optic region into parallel arrays indexed by material
    # cell number (1..N_mat); the magnetization, thermal, ADE and gyration states
    # all share this indexing.
    cell_keys = sort!(collect(keys(model_cells)))
    Nmat = length(cell_keys)
    material_cells = Vector{Int}(undef, Nmat)
    material_fill = Vector{Float64}(undef, Nmat)
    material_inv_eps = Vector{Float64}(undef, Nmat)
    material_ijk = Vector{NTuple{3,Int}}(undef, Nmat)
    for (n, li) in enumerate(cell_keys)
        (i, j, k, f) = model_cells[li]
        material_cells[n] = li
        material_fill[n] = f
        material_inv_eps[n] = inv_eps[i, j, k]
        material_ijk[n] = (i, j, k)
    end

    pole_keys = sort!(collect(keys(pole_cells)))
    Npole = length(pole_keys)
    pole_active_cells = Vector{Int}(undef, Npole)
    pole_active_fill = Vector{Float64}(undef, Npole)
    pole_active_inv_eps = Vector{Float64}(undef, Npole)
    pole_active_ijk = Vector{NTuple{3,Int}}(undef, Npole)
    for (n, li) in enumerate(pole_keys)
        (i, j, k, f) = pole_cells[li]
        pole_active_cells[n] = li
        pole_active_fill[n] = f
        pole_active_inv_eps[n] = inv_eps[i, j, k]
        pole_active_ijk[n] = (i, j, k)
    end

    return (;
        epsr,
        inv_eps_x=copy(inv_eps),
        inv_eps_y=copy(inv_eps),
        inv_eps_z=copy(inv_eps),
        material_id,
        fill_fraction=fill_fraction_grid,
        active_indices=unique(active_indices),
        materials,
        material_cells,
        material_fill,
        material_inv_eps,
        material_ijk,
        n_material=Nmat,
        pole_active_cells,
        pole_active_fill,
        pole_active_inv_eps,
        pole_active_ijk,
        pole_poles,
    )
end
