module MagnetoPhotonicMakieExt

using MagnetoPhotonic
import CairoMakie

# 2-D plan view of a scene's shapes (extends the parent `plot_scene`).
function MagnetoPhotonic.plot_scene(scene::MagnetoPhotonic.Scene; filename=nothing, size=(1200, 360))
    fig = CairoMakie.Figure(size=size)
    ax = CairoMakie.Axis(fig[1, 1], aspect=CairoMakie.DataAspect(), xlabel="x", ylabel="y")
    for entry in scene.entries
        entry.shape isa MagnetoPhotonic.Cylinder && continue
        poly = entry.shape isa MagnetoPhotonic.Box ?
               [MagnetoPhotonic.Vec2(entry.shape.xmin, entry.shape.ymin), MagnetoPhotonic.Vec2(entry.shape.xmax, entry.shape.ymin),
                MagnetoPhotonic.Vec2(entry.shape.xmax, entry.shape.ymax), MagnetoPhotonic.Vec2(entry.shape.xmin, entry.shape.ymax)] :
               MagnetoPhotonic.polygon(entry.shape)
        pts = [CairoMakie.Point2f(p.x, p.y) for p in poly]
        col = get(MagnetoPhotonic.MATERIAL_COLORS, entry.material.name, "#777777")
        CairoMakie.poly!(ax, pts; color=col, strokecolor=:black, strokewidth=1)
    end
    filename === nothing || CairoMakie.save(filename, fig)
    return fig
end

# Heatmap animation of captured field slices (extends the parent `render_field_video`).
function MagnetoPhotonic.render_field_video(frames::AbstractVector{<:AbstractMatrix}, filename::AbstractString;
                                        fps::Integer=20, colormap=:turbo, colorrange=nothing)
    isempty(frames) && throw(ArgumentError("no frames to render"))
    cr = colorrange === nothing ? (0.0, maximum(maximum, frames)) : colorrange
    node = CairoMakie.Observable(Matrix{Float64}(frames[1]))
    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1, 1], aspect=CairoMakie.DataAspect())
    CairoMakie.heatmap!(ax, node; colormap=colormap, colorrange=cr)
    CairoMakie.record(fig, filename, eachindex(frames); framerate=fps) do i
        node[] = Matrix{Float64}(frames[i])
    end
    return filename
end

end
