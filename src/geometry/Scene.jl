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

function rasterize(scene::Scene, grid; subpixel::Integer=4, background::Material=Material("vacuum"; epsr=1.0))
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
        for i in 1:Nx, j in 1:Ny, k in 1:Nz
            xf = _contains_xy(
                entry.shape,
                grid.x.centers[i],
                grid.y.centers[j];
                subpixel=subpixel,
                xlo=grid.x.edges[i],
                xhi=grid.x.edges[i + 1],
                ylo=grid.y.edges[j],
                yhi=grid.y.edges[j + 1],
            )
            zf = _z_fraction(entry.shape, grid.z.edges[k], grid.z.edges[k + 1])
            f = xf * zf
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
