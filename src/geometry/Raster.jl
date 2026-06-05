struct BBox
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
end

function get_bbox(poly::Vector{Vec2})
    xs = map(vx, poly)
    ys = map(vy, poly)
    return BBox(minimum(xs), maximum(xs), minimum(ys), maximum(ys))
end

function polygon_area(poly::Vector{Vec2})
    n = length(poly)
    n >= 3 || return 0.0
    acc = 0.0
    j = n
    for i in 1:n
        acc += poly[j].x * poly[i].y - poly[i].x * poly[j].y
        j = i
    end
    return 0.5 * acc
end

function is_inside_polygon(x::Real, y::Real, poly_pts::Vector{Vec2})
    inside = false
    xf, yf = Float64(x), Float64(y)
    j = length(poly_pts)
    for i in 1:length(poly_pts)
        xi, yi = poly_pts[i].x, poly_pts[i].y
        xj, yj = poly_pts[j].x, poly_pts[j].y
        if ((yi > yf) != (yj > yf)) && (xf < (xj - xi) * (yf - yi) / (yj - yi) + xi)
            inside = !inside
        end
        j = i
    end
    return inside
end

function is_inside_any(x::Real, y::Real, polys::Vector{Vector{Vec2}}, bboxes::Vector{BBox})
    for (poly, box) in zip(polys, bboxes)
        if box.xmin <= x <= box.xmax && box.ymin <= y <= box.ymax && is_inside_polygon(x, y, poly)
            return true
        end
    end
    return false
end

function fill_fraction(poly::Vector{Vec2}, xlo::Real, xhi::Real, ylo::Real, yhi::Real; subpixel::Integer=4)
    subpixel > 0 || throw(ArgumentError("subpixel must be positive"))
    box = get_bbox(poly)
    if Float64(xhi) < box.xmin || Float64(xlo) > box.xmax || Float64(yhi) < box.ymin || Float64(ylo) > box.ymax
        return 0.0
    end
    hits = 0
    total = Int(subpixel)^2
    dx = (Float64(xhi) - Float64(xlo)) / subpixel
    dy = (Float64(yhi) - Float64(ylo)) / subpixel
    for a in 0:(subpixel - 1), b in 0:(subpixel - 1)
        xs = Float64(xlo) + (a + 0.5) * dx
        ys = Float64(ylo) + (b + 0.5) * dy
        hits += is_inside_polygon(xs, ys, poly) ? 1 : 0
    end
    return hits / total
end
