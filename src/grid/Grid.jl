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
    return _axis_from_edges(vcat(edges_left, edges_right))
end

function propagation_axis(L_phys::Real, d_coarse::Real, d_fine::Real, x_film_start::Real, x_film_end::Real, buffer::Real, stretch_ratio::Real)
    L = Float64(L_phys)
    fine_start = Float64(x_film_start) - Float64(buffer)
    fine_end = Float64(x_film_end) + Float64(buffer)
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
    sort!(unique!(edges))
    return _axis_from_edges(edges)
end

function uniform_grid(xlim, ylim, zlim, d::Real)
    return Grid3D(uniform_axis(xlim[1], xlim[2], d), uniform_axis(ylim[1], ylim[2], d), uniform_axis(zlim[1], zlim[2], d))
end

uniform_grid(xlim, d::Real) = Grid1D(uniform_axis(xlim[1], xlim[2], d))
uniform_grid(xlim, ylim, d::Real) = Grid2D(uniform_axis(xlim[1], xlim[2], d), uniform_axis(ylim[1], ylim[2], d))

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
