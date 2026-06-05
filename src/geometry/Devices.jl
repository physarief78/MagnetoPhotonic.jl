function not_gate_60um(; units::Symbol=:um, wg_width=0.40, wg_height=0.40, film_thickness=1.0)
    scale = units == :m ? 1e-6 : 1.0
    x_in = 0.0
    x_bend = 5.0
    x_merge = 25.0
    x_taper = 30.0
    x_out = 60.0
    x_film_start = 40.0
    x_film_end = x_film_start + film_thickness
    y_top = 0.75
    y_bot = -0.75
    y_merge = wg_width / 2.0

    path_top = [Vec2(x_in, y_top), Vec2(x_bend, y_top)]
    path_bot = [Vec2(x_in, y_bot), Vec2(x_bend, y_bot)]
    bend_resolution = 60
    for i in 1:bend_resolution
        t = i / bend_resolution
        x = x_bend + t * (x_merge - x_bend)
        y_t = y_merge + (y_top - y_merge) * 0.5 * (1.0 + cos(pi * t))
        y_b = -y_merge + (y_bot - (-y_merge)) * 0.5 * (1.0 + cos(pi * t))
        push!(path_top, Vec2(x, y_t))
        push!(path_bot, Vec2(x, y_b))
    end

    path_taper = [Vec2(x_merge, 0.0), Vec2(x_taper, 0.0)]
    path_out_pre = [Vec2(x_taper, 0.0), Vec2(x_film_start, 0.0)]
    path_out_film = [Vec2(x_film_start, 0.0), Vec2(x_film_end, 0.0)]
    path_out_post = [Vec2(x_film_end, 0.0), Vec2(x_out, 0.0)]

    if scale != 1.0
        path_top = [scale * p for p in path_top]
        path_bot = [scale * p for p in path_bot]
        path_taper = [scale * p for p in path_taper]
        path_out_pre = [scale * p for p in path_out_pre]
        path_out_film = [scale * p for p in path_out_film]
        path_out_post = [scale * p for p in path_out_post]
        wg_width *= scale
        wg_height *= scale
    end

    core = Material("Si3N4"; epsr=4.0, color=:gray)
    active = Material("GdFeCo"; epsr=1.0, model=:magneto_optic, color=:orange)
    scene = Scene()
    add_shape!(scene, Waveguide(path_top, wg_width, 0.0, wg_height), core)
    add_shape!(scene, Waveguide(path_bot, wg_width, 0.0, wg_height), core)
    add_shape!(scene, TaperedWaveguide(path_taper, 2.0 * wg_width, wg_width, 0.0, wg_height), core)
    add_shape!(scene, Waveguide(path_out_pre, wg_width, 0.0, wg_height), core)
    add_shape!(scene, Waveguide(path_out_film, wg_width, 0.0, wg_height), active)
    add_shape!(scene, Waveguide(path_out_post, wg_width, 0.0, wg_height), core)

    return (;
        scene,
        paths=(top=path_top, bottom=path_bot, taper=path_taper, pre=path_out_pre, film=path_out_film, post=path_out_post),
        polygons=(
            top=generate_waveguide_polygon(path_top, wg_width),
            bottom=generate_waveguide_polygon(path_bot, wg_width),
            taper=generate_tapered_polygon(path_taper, 2.0 * wg_width, wg_width),
            film=generate_waveguide_polygon(path_out_film, wg_width),
        ),
        x_film_start=x_film_start * scale,
        x_film_end=x_film_end * scale,
        wg_width,
        wg_height,
    )
end

function passive_waveguide(; length=10.0, width=0.4, zmin=0.0, zmax=0.4, epsr=4.0)
    scene = Scene()
    add_shape!(scene, Waveguide([Vec2(0.0, 0.0), Vec2(length, 0.0)], width, zmin, zmax), Material("core"; epsr=epsr))
    return scene
end

function hm_test_pattern(; zmin=0.0, zmax=0.4, epsr=4.0)
    scene = Scene()
    add_shape!(scene, PolygonShape(generate_H_geometry(0.0, 0.0), zmin, zmax), Material("H"; epsr=epsr, color=:blue))
    add_shape!(scene, PolygonShape(generate_M_geometry(10.0, 0.0), zmin, zmax), Material("M"; epsr=epsr, color=:red))
    return scene
end
