function extrude_waveguide_mesh(poly_2d::Vector{Vec2}, z_min::Real, z_max::Real)
    n = length(poly_2d)
    vertices = Vector{NTuple{3,Float64}}(undef, 2n)
    for (i, p) in enumerate(poly_2d)
        vertices[i] = (p.x, p.y, Float64(z_min))
        vertices[n + i] = (p.x, p.y, Float64(z_max))
    end
    faces = NTuple{3,Int}[]
    for i in 2:(n - 1)
        push!(faces, (1, i, i + 1))
        push!(faces, (n + 1, n + i + 1, n + i))
    end
    for i in 1:n
        j = i == n ? 1 : i + 1
        push!(faces, (i, j, n + j))
        push!(faces, (i, n + j, n + i))
    end
    return (vertices=vertices, faces=faces)
end

function write_device_obj(filename::AbstractString, scene::Scene)
    open(filename, "w") do io
        println(io, "# MagnetoPhotonic device mesh")
        offset = 0
        for entry in scene.entries
            entry.shape isa Box && continue
            mesh = extrude_waveguide_mesh(polygon(entry.shape), zrange(entry.shape)...)
            println(io, "o ", replace(entry.material.name, ' ' => '_'))
            for v in mesh.vertices
                println(io, "v $(v[1]) $(v[2]) $(v[3])")
            end
            for f in mesh.faces
                println(io, "f $(f[1] + offset) $(f[2] + offset) $(f[3] + offset)")
            end
            offset += length(mesh.vertices)
        end
    end
    return filename
end

function write_plan_svg(filename::AbstractString, scene::Scene; width_px::Int=1200, height_px::Int=360)
    polys = Vector{Tuple{Vector{Vec2},String}}()
    for entry in scene.entries
        if entry.shape isa Box
            s = entry.shape
            push!(polys, ([Vec2(s.xmin, s.ymin), Vec2(s.xmax, s.ymin), Vec2(s.xmax, s.ymax), Vec2(s.xmin, s.ymax)], entry.material.name))
        elseif !(entry.shape isa Cylinder)
            push!(polys, (polygon(entry.shape), entry.material.name))
        end
    end
    allpts = reduce(vcat, first.(polys); init=Vec2[])
    box = get_bbox(allpts)
    sx = width_px / max(box.xmax - box.xmin, eps())
    sy = height_px / max(box.ymax - box.ymin, eps())
    scale = 0.90 * min(sx, sy)
    xoff = (width_px - scale * (box.xmax - box.xmin)) / 2 - scale * box.xmin
    yoff = (height_px + scale * (box.ymax - box.ymin)) / 2 + scale * box.ymin

    open(filename, "w") do io
        println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$width_px\" height=\"$height_px\" viewBox=\"0 0 $width_px $height_px\">")
        println(io, "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>")
        for (poly, name) in polys
            pts = join(["$(xoff + scale * p.x),$(yoff - scale * p.y)" for p in poly], " ")
            color = get(MATERIAL_COLORS, name, "#777777")
            println(io, "<polygon points=\"$pts\" fill=\"$color\" stroke=\"#111111\" stroke-width=\"1\"/>")
        end
        println(io, "</svg>")
    end
    return filename
end
