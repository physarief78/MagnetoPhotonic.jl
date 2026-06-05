abstract type AbstractShape end

struct Box <: AbstractShape
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
    zmin::Float64
    zmax::Float64
end

struct PolygonShape <: AbstractShape
    points::Vector{Vec2}
    zmin::Float64
    zmax::Float64
end

struct Waveguide <: AbstractShape
    path::Vector{Vec2}
    width::Float64
    zmin::Float64
    zmax::Float64
end

struct TaperedWaveguide <: AbstractShape
    path::Vector{Vec2}
    width_start::Float64
    width_end::Float64
    zmin::Float64
    zmax::Float64
end

struct Cylinder <: AbstractShape
    center::Vec2
    radius::Float64
    zmin::Float64
    zmax::Float64
end

struct Letter <: AbstractShape
    name::Char
    origin::Vec2
    width::Float64
    height::Float64
    thickness::Float64
    zmin::Float64
    zmax::Float64
end

function _check_path(path)
    length(path) >= 2 || throw(ArgumentError("path must contain at least two points"))
    return nothing
end

function generate_waveguide_polygon(path::Vector{Vec2}, width::Real)
    _check_path(path)
    half_w = Float64(width) / 2.0
    upper_wall = Vec2[]
    lower_wall = Vec2[]
    n_points = length(path)

    for i in 1:n_points
        t_in = i > 1 ? normalize2(path[i] - path[i - 1]) : normalize2(path[i + 1] - path[i])
        t_out = i < n_points ? normalize2(path[i + 1] - path[i]) : t_in
        n_in = Vec2(-t_in.y, t_in.x)
        n_out = Vec2(-t_out.y, t_out.x)
        miter_dir = normalize2(n_in + n_out)
        scale_factor = dot2(miter_dir, n_in)
        miter_len = abs(scale_factor) > 1e-6 ? half_w / scale_factor : half_w
        miter_vec = miter_len * miter_dir
        push!(upper_wall, path[i] + miter_vec)
        push!(lower_wall, path[i] - miter_vec)
    end

    final_poly = copy(upper_wall)
    append!(final_poly, reverse(lower_wall))
    return final_poly
end

function generate_tapered_polygon(path::Vector{Vec2}, width_start::Real, width_end::Real)
    _check_path(path)
    n_points = length(path)
    upper_wall = Vec2[]
    lower_wall = Vec2[]

    total_len = 0.0
    for i in 2:n_points
        total_len += norm2(path[i] - path[i - 1])
    end

    curr_len = 0.0
    for i in 1:n_points
        if i > 1
            curr_len += norm2(path[i] - path[i - 1])
        end
        t = total_len == 0.0 ? 0.0 : curr_len / total_len
        half_w = (Float64(width_start) + t * (Float64(width_end) - Float64(width_start))) / 2.0

        t_in = i > 1 ? normalize2(path[i] - path[i - 1]) : normalize2(path[i + 1] - path[i])
        t_out = i < n_points ? normalize2(path[i + 1] - path[i]) : t_in
        n_in = Vec2(-t_in.y, t_in.x)
        n_out = Vec2(-t_out.y, t_out.x)
        miter_dir = normalize2(n_in + n_out)
        scale_factor = dot2(miter_dir, n_in)
        miter_len = abs(scale_factor) > 1e-6 ? half_w / scale_factor : half_w
        miter_vec = miter_len * miter_dir
        push!(upper_wall, path[i] + miter_vec)
        push!(lower_wall, path[i] - miter_vec)
    end

    final_poly = copy(upper_wall)
    append!(final_poly, reverse(lower_wall))
    return final_poly
end

function generate_H_geometry(origin_x::Real, origin_y::Real; width=6.0, height=10.0, thickness=1.6)
    x0, x1 = Float64(origin_x), Float64(origin_x) + thickness
    x2, x3 = Float64(origin_x) + width - thickness, Float64(origin_x) + width
    y0, y_top = Float64(origin_y), Float64(origin_y) + height
    y_bar_start = Float64(origin_y) + (height / 2) - (thickness / 2)
    y_bar_end = Float64(origin_y) + (height / 2) + (thickness / 2)
    return Vec2[
        Vec2(x0, y0), Vec2(x1, y0), Vec2(x1, y_bar_start), Vec2(x2, y_bar_start),
        Vec2(x2, y0), Vec2(x3, y0), Vec2(x3, y_top), Vec2(x2, y_top),
        Vec2(x2, y_bar_end), Vec2(x1, y_bar_end), Vec2(x1, y_top), Vec2(x0, y_top),
    ]
end

function generate_M_geometry(origin_x::Real, origin_y::Real; width=8.0, height=10.0, thickness=1.6)
    ox, oy = Float64(origin_x), Float64(origin_y)
    x_left_outer, x_left_inner = ox, ox + thickness
    x_right_inner, x_right_outer = ox + width - thickness, ox + width
    x_mid = ox + width / 2
    y_bottom, y_top = oy, oy + height
    y_v_tip_outer = oy + (height / 2) + (thickness / 2)

    p_diag_top = Vec2(x_left_inner, y_top)
    p_diag_bot = Vec2(x_mid, y_v_tip_outer)
    vec_diag = p_diag_bot - p_diag_top
    norm_vec = Vec2(vec_diag.y, -vec_diag.x)
    p_inner_ref = p_diag_top + (thickness / norm2(norm_vec)) * norm_vec
    slope = vec_diag.y / vec_diag.x
    y_inner_shoulder = slope * (x_left_inner - p_inner_ref.x) + p_inner_ref.y
    y_inner_crotch = slope * (x_mid - p_inner_ref.x) + p_inner_ref.y

    return Vec2[
        Vec2(x_left_outer, y_bottom), Vec2(x_left_inner, y_bottom),
        Vec2(x_left_inner, y_inner_shoulder), Vec2(x_mid, y_inner_crotch),
        Vec2(x_right_inner, y_inner_shoulder), Vec2(x_right_inner, y_bottom),
        Vec2(x_right_outer, y_bottom), Vec2(x_right_outer, y_top),
        Vec2(x_right_inner, y_top), Vec2(x_mid, y_v_tip_outer),
        Vec2(x_left_inner, y_top), Vec2(x_left_outer, y_top),
    ]
end

polygon(shape::PolygonShape) = shape.points
polygon(shape::Waveguide) = generate_waveguide_polygon(shape.path, shape.width)
polygon(shape::TaperedWaveguide) = generate_tapered_polygon(shape.path, shape.width_start, shape.width_end)
polygon(shape::Letter) = shape.name == 'H' ? generate_H_geometry(shape.origin.x, shape.origin.y; width=shape.width, height=shape.height, thickness=shape.thickness) :
                          shape.name == 'M' ? generate_M_geometry(shape.origin.x, shape.origin.y; width=shape.width, height=shape.height, thickness=shape.thickness) :
                          throw(ArgumentError("supported letters are H and M"))

zrange(shape::Box) = (shape.zmin, shape.zmax)
zrange(shape::PolygonShape) = (shape.zmin, shape.zmax)
zrange(shape::Waveguide) = (shape.zmin, shape.zmax)
zrange(shape::TaperedWaveguide) = (shape.zmin, shape.zmax)
zrange(shape::Cylinder) = (shape.zmin, shape.zmax)
zrange(shape::Letter) = (shape.zmin, shape.zmax)
