function _shape_x_fraction(shape::Box, xlo, xhi)
    overlap = min(Float64(xhi), shape.xmax) - max(Float64(xlo), shape.xmin)
    return clamp(overlap / (Float64(xhi) - Float64(xlo)), 0.0, 1.0)
end

function _shape_x_fraction(shape::Union{PolygonShape,Waveguide,TaperedWaveguide,Letter}, xlo, xhi)
    box = get_bbox(polygon(shape))
    overlap = min(Float64(xhi), box.xmax) - max(Float64(xlo), box.xmin)
    return clamp(overlap / (Float64(xhi) - Float64(xlo)), 0.0, 1.0)
end

function _shape_x_fraction(shape::Cylinder, xlo, xhi)
    overlap = min(Float64(xhi), shape.center.x + shape.radius) - max(Float64(xlo), shape.center.x - shape.radius)
    return clamp(overlap / (Float64(xhi) - Float64(xlo)), 0.0, 1.0)
end

function rasterize_1d(scene::Scene, grid::Grid1D; background::Material=Material("vacuum"; epsr=1.0))
    Nx = length(grid.x.centers)
    epsr = fill(background.epsr, Nx)
    fill_fraction_grid = zeros(Float64, Nx)
    material_id = zeros(Int, Nx)
    active = Int[]
    active_fill = Float64[]
    active_inv_eps = Float64[]
    poles = Any[]
    for (entry_id, entry) in enumerate(scene.entries)
        material = entry.material
        for i in 1:Nx
            f = _shape_x_fraction(entry.shape, grid.x.edges[i], grid.x.edges[i + 1])
            if f > 0.0
                epsr[i] = (1.0 - f) * epsr[i] + f * material.epsr
                fill_fraction_grid[i] = max(fill_fraction_grid[i], f)
                material_id[i] = entry_id + 1
                if !isempty(material.poles)
                    push!(active, i)
                    push!(active_fill, f)
                    isempty(poles) && append!(poles, material.poles)
                end
            end
        end
    end
    inv_eps = 1.0 ./ epsr
    active_inv_eps = inv_eps[active]
    return (; epsr, inv_eps, material_id, fill_fraction=fill_fraction_grid,
            active_indices=active, active_fill, active_inv_eps, poles)
end

function rasterize_2d(scene::Scene, grid::Grid2D; subpixel::Integer=4, background::Material=Material("vacuum"; epsr=1.0))
    Nx, Ny = length(grid.x.centers), length(grid.y.centers)
    epsr = fill(background.epsr, Nx, Ny)
    fill_fraction_grid = zeros(Float64, Nx, Ny)
    material_id = zeros(Int, Nx, Ny)
    lin = LinearIndices((Nx, Ny))
    active = Int[]
    active_fill = Float64[]
    poles = Any[]
    for (entry_id, entry) in enumerate(scene.entries)
        material = entry.material
        for i in 1:Nx, j in 1:Ny
            f = _contains_xy(entry.shape, grid.x.centers[i], grid.y.centers[j];
                             subpixel=subpixel, xlo=grid.x.edges[i], xhi=grid.x.edges[i + 1],
                             ylo=grid.y.edges[j], yhi=grid.y.edges[j + 1])
            if f > 0.0
                epsr[i, j] = (1.0 - f) * epsr[i, j] + f * material.epsr
                fill_fraction_grid[i, j] = max(fill_fraction_grid[i, j], f)
                material_id[i, j] = entry_id + 1
                if !isempty(material.poles)
                    push!(active, lin[i, j])
                    push!(active_fill, f)
                    isempty(poles) && append!(poles, material.poles)
                end
            end
        end
    end
    inv_eps = 1.0 ./ epsr
    active_inv_eps = inv_eps[active]
    return (; epsr, inv_eps, material_id, fill_fraction=fill_fraction_grid,
            active_indices=unique(active), active_fill, active_inv_eps, poles)
end
