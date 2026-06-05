struct Medium
    name::String
    epsr::Float64
    poles::Vector{Any}
    color::Symbol
end

function Medium(; index::Real=1.0, epsr::Union{Nothing,Real}=nothing,
                name::AbstractString="medium", poles=(), color::Symbol=:gray)
    er = epsr === nothing ? Float64(index)^2 : Float64(epsr)
    return Medium(String(name), er, Any[poles...], color)
end

medium_epsr(m::Medium) = m.epsr
