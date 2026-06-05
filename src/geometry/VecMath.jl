struct Vec2
    x::Float64
    y::Float64
end

Vec2(t::Tuple{<:Real,<:Real}) = Vec2(Float64(t[1]), Float64(t[2]))
Vec2(v::AbstractVector{<:Real}) = Vec2(Float64(v[1]), Float64(v[2]))

vx(v::Vec2) = v.x
vy(v::Vec2) = v.y
as_tuple(v::Vec2) = (v.x, v.y)

Base.:+(a::Vec2, b::Vec2) = Vec2(a.x + b.x, a.y + b.y)
Base.:-(a::Vec2, b::Vec2) = Vec2(a.x - b.x, a.y - b.y)
Base.:-(v::Vec2) = Vec2(-v.x, -v.y)
Base.:*(s::Real, v::Vec2) = Vec2(Float64(s) * v.x, Float64(s) * v.y)
Base.:*(v::Vec2, s::Real) = Float64(s) * v
Base.:/(v::Vec2, s::Real) = Vec2(v.x / Float64(s), v.y / Float64(s))
Base.isapprox(a::Vec2, b::Vec2; kwargs...) = isapprox(a.x, b.x; kwargs...) && isapprox(a.y, b.y; kwargs...)

dot2(a::Vec2, b::Vec2) = a.x * b.x + a.y * b.y
norm2(v::Vec2) = sqrt(dot2(v, v))

function normalize2(v::Vec2)
    n = norm2(v)
    return n == 0.0 ? Vec2(0.0, 0.0) : v / n
end
