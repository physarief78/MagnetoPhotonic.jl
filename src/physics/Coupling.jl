# Multiphysics coupling: EM-absorbed power -> 4TM heat -> LLB magnetization.
#
# The material-cell implementation lives in the KernelAbstractions kernel layer
# so CPU and GPU use the same update path. `:cycle_average` uses the optical
# cycle-average power density; `:ade_work` uses positive local E dot J_ADE when
# ADE states are supplied, otherwise it falls back to cycle-average absorption.

multiphysics_step!(thermal::ThermalState, ::Nothing, args...; kwargs...) = thermal
multiphysics_step!(thermal::ThermalState, mag, fields, region, ::NullModel, lut, dt::Real; kwargs...) = thermal

# `absorption` is an AbsorptionState whose window_sum (pabs·dt integrated by
# accumulate_absorption! since the last call) supplies the heating term; pass
# `absorption=nothing` for a pure cool-down step (the reference relaxation path).
function multiphysics_step!(thermal::ThermalState, mag::MagnetizationState, fields, region,
                            model::MagnetoOpticModel, lut, dt::Real;
                            subcycles::Integer=1, absorption=nothing,
                            brillouin_iters::Integer=2,
                            backend::AbstractBackend=CPUBackend(),
                            compute_T::Type=default_compute_type(backend),
                            _legacy_kwargs...)
    subcycles > 0 || throw(ArgumentError("subcycles must be positive"))
    return _ka_multiphysics_step!(backend, thermal, mag, fields, region, model, lut, dt;
                                  subcycles=subcycles, absorption=absorption,
                                  brillouin_iters=brillouin_iters, compute_T=compute_T)
end

# Per-EM-step accumulation of the absorbed power density into `absorption`
# (U_abs, window_sum, peak) — the reference's kernel_em_pabs_accumulate!.
function accumulate_absorption!(absorption::AbsorptionState, fields, region,
                                model::MagnetoOpticModel, dt::Real;
                                absorption_model::Symbol=:cycle_average,
                                eps0::Real=FDTDParams().eps0,
                                backend::AbstractBackend=CPUBackend(),
                                ade_x=nothing, ade_y=nothing, ade_z=nothing)
    return _ka_pabs_accumulate!(backend, absorption, fields, region, model, dt;
                                absorption_model=absorption_model, eps0=eps0,
                                ade_x=ade_x, ade_y=ade_y, ade_z=ade_z)
end
