_vec2(x) = x isa Vec2 ? x : Vec2(x)

function straight(p0, p1; width=nothing, role::Symbol=:core, name::Symbol=:straight)
    path = Vec2[_vec2(p0), _vec2(p1)]
    return (kind=:waveguide, name=name, role=role, path=path, width=width)
end

function taper(p0, p1; width_start, width_end, role::Symbol=:core, name::Symbol=:taper)
    path = Vec2[_vec2(p0), _vec2(p1)]
    return (kind=:tapered, name=name, role=role, path=path,
            width_start=Float64(width_start), width_end=Float64(width_end))
end

function cosine_bend(p0, p1; n::Integer=60, width=nothing, role::Symbol=:core, name::Symbol=:bend)
    a = _vec2(p0)
    b = _vec2(p1)
    nseg = max(2, Int(n))
    path = Vec2[]
    for q in 0:nseg
        t = q / nseg
        x = a.x + t * (b.x - a.x)
        s = 0.5 * (1.0 - cos(pi * t))
        y = a.y + s * (b.y - a.y)
        push!(path, Vec2(x, y))
    end
    return (kind=:waveguide, name=name, role=role, path=path, width=width)
end

function ybranch(p0, split, p_top, p_bottom; n::Integer=60, width=nothing, name::Symbol=:ybranch)
    trunk = straight(p0, split; width=width, role=:core, name=Symbol(name, "_trunk"))
    top = cosine_bend(split, p_top; n=n, width=width, role=:core, name=Symbol(name, "_top"))
    bottom = cosine_bend(split, p_bottom; n=n, width=width, role=:core, name=Symbol(name, "_bottom"))
    return (trunk, top, bottom)
end

function film_region(x0, x1; width=nothing, y::Real=0.0, role::Symbol=:film, name::Symbol=:film)
    return straight((Float64(x0), Float64(y)), (Float64(x1), Float64(y));
                    width=width, role=role, name=name)
end

function _flatten_segments!(out, item)
    if item isa NamedTuple && haskey(item, :kind)
        push!(out, item)
    elseif item isa AbstractVector || item isa Tuple
        for child in item
            _flatten_segments!(out, child)
        end
    else
        throw(ArgumentError("unsupported device segment $item"))
    end
    return out
end

function _segment_width(seg, default_width)
    haskey(seg, :width) && seg.width !== nothing && return Float64(seg.width)
    return Float64(default_width)
end

function _segment_name(seg, i)
    haskey(seg, :name) ? Symbol(seg.name) : Symbol("segment_", i)
end

function _segment_role(seg)
    haskey(seg, :role) ? Symbol(seg.role) : :core
end

function _segment_material(seg, core::Material, film_material::Material)
    return _segment_role(seg) === :film ? film_material : core
end

function _segment_shape(seg, width::Real, height::Real, zmin::Real)
    if seg.kind === :waveguide
        return Waveguide(seg.path, _segment_width(seg, width), Float64(zmin), Float64(zmin) + Float64(height))
    elseif seg.kind === :tapered
        return TaperedWaveguide(seg.path, Float64(seg.width_start), Float64(seg.width_end),
                                Float64(zmin), Float64(zmin) + Float64(height))
    end
    throw(ArgumentError("unsupported device segment kind $(seg.kind)"))
end

function _segment_polygon(seg, width::Real)
    if seg.kind === :waveguide
        return generate_waveguide_polygon(seg.path, _segment_width(seg, width))
    elseif seg.kind === :tapered
        return generate_tapered_polygon(seg.path, seg.width_start, seg.width_end)
    end
    return Vec2[]
end

function _film_bounds(segments)
    films = [seg for seg in segments if _segment_role(seg) === :film]
    isempty(films) && return (NaN, NaN)
    xs = Float64[]
    for seg in films, p in seg.path
        push!(xs, p.x)
    end
    return (minimum(xs), maximum(xs))
end

function _ports_from_segments(segments)
    isempty(segments) && return NamedTuple()
    first_path = first(segments).path
    last_path = last(segments).path
    return (input=first(first_path), output=last(last_path))
end

function _dict_named_tuple(d::Dict{Symbol,V}) where {V}
    names = Tuple(collect(keys(d)))
    vals = Tuple(d[k] for k in names)
    return NamedTuple{names}(vals)
end

function waveguide_device(segments...;
                          core::Material=Material("Si3N4"; epsr=4.0, color=:gray),
                          film_model::AbstractPhysicsModel=MagnetoOpticModel(),
                          film_material::Material=Material("GdFeCo"; epsr=1.0, model=film_model, color=:orange),
                          height::Real=0.40e-6,
                          width::Real=0.40e-6,
                          zmin::Real=0.0)
    flat = Any[]
    for seg in segments
        _flatten_segments!(flat, seg)
    end
    scene = Scene()
    paths = Dict{Symbol,Vector{Vec2}}()
    polygons = Dict{Symbol,Vector{Vec2}}()
    film_regions = NamedTuple[]
    for (i, seg) in enumerate(flat)
        name = _segment_name(seg, i)
        add_shape!(scene, _segment_shape(seg, width, height, zmin), _segment_material(seg, core, film_material))
        paths[name] = seg.path
        polygons[name] = _segment_polygon(seg, width)
        if _segment_role(seg) === :film
            push!(film_regions, (name=name, path=seg.path, polygon=polygons[name]))
        end
    end
    x0, x1 = _film_bounds(flat)
    return (;
        scene,
        ports=_ports_from_segments(flat),
        film_regions,
        paths=_dict_named_tuple(paths),
        polygons=_dict_named_tuple(polygons),
        x_film_start=x0,
        x_film_end=x1,
        wg_width=Float64(width),
        wg_height=Float64(height),
    )
end

function waveguide_device(; segments=(), kwargs...)
    return waveguide_device(segments...; kwargs...)
end
