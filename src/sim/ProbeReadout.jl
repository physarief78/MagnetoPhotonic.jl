# Gold-standard probe readout.
#
# Per-cell running DFT of (Ey, Ez, Hy, Hz) at the pre- and post-planes, plus the
# per-cell time-integrated net Poynting flux. This mirrors the reference
# `kernel_probe_dft_accumulate!`: the spectral quantities are built from the true
# complex cross-power Sx(ω) = Σ (Êy·conj(Ĥz) − Êz·conj(Ĥy))·dA — not from |E|² —
# so `Sx_*_im` are non-trivial and `R_omega` is a genuine reference-normalized
# reflection, not identically zero. Scalar T/R/A use net flux with bare-waveguide
# reference normalization (transmitted = post-plane net, reflected = incident −
# pre-plane net), exactly as the production gold-standard path.
mutable struct ProbeReadout <: AbstractMonitor
    x_pre::Float64
    x_post::Float64
    mode::Any
    omega_bins::Vector{Float64}
    center_bin::Int
    trace_stride::Int
    frame_target::Int
    frame_skip::Int
    store_frames::Bool
    lambda0::Float64
    probe_amplitude::Float64
    probe_tau::Float64
    # plane geometry, allocated lazily on the first recorded step
    initialized::Bool
    Ny::Int
    Nz::Int
    area_yz::Matrix{Float64}
    mode_w::Matrix{Float64}
    mode_norm::Float64
    backend::Any
    omega_bins_dev::Any
    area_yz_dev::Any
    mode_w_dev::Any
    # per-cell running complex DFT at the pre (…_pre) and post (…_post) planes
    dft_Ey_pre::Any
    dft_Ez_pre::Any
    dft_Hy_pre::Any
    dft_Hz_pre::Any
    dft_Ey_post::Any
    dft_Ez_post::Any
    dft_Hy_post::Any
    dft_Hz_post::Any
    # per-cell signed forward flux (Ey·Hz − Ez·Hy), time-integrated (× dt)
    energy_pre::Any
    energy_post::Any
    work_Ey_pre::Any
    work_Ez_pre::Any
    work_Ey_post::Any
    work_Ez_post::Any
    # mode-overlap-weighted scalar traces (for group delay / pulse broadening)
    t_trace::Vector{Float64}
    Ey_pre::Vector{Float64}
    Ez_pre::Vector{Float64}
    Ey_post::Vector{Float64}
    Ez_post::Vector{Float64}
    # strided field movies
    frame_indices::Vector{Int}
    frame_times::Vector{Float64}
    Ey_xy::Vector{Any}
    Ez_xy::Vector{Any}
    Ey_yz_pre::Vector{Any}
    Ez_yz_pre::Vector{Any}
    Ey_yz_post::Vector{Any}
    Ez_yz_post::Vector{Any}
    # explicit step indices to capture (the reference uses
    # unique(round.(Int, range(1, steps; length=target)))); empty = stride fallback
    frame_plan::Vector{Int}
    frame_ptr::Int
    # z position of the xy movie slice (the reference uses the waveguide
    # mid-height, wg_height/2); NaN = domain-centre fallback
    slice_z_position::Float64
    # last probe step (the reference also samples the traces at n==1 and n==steps,
    # not only at the stride multiples); 0 = unknown
    final_step::Int
end

function ProbeReadout(; pre::Real, post::Real, mode=nothing, lambda0::Real=532e-9,
                      omega_bins=nothing, spectrum_bins::Integer=5,
                      trace_stride::Integer=10, frame_target::Integer=0,
                      frame_skip::Integer=0, store_frames::Bool=true,
                      probe_amplitude::Real=1.0,
                      probe_tau::Real=24e-15,
                      frame_indices=Int[],
                      slice_z_position::Real=NaN,
                      final_step::Integer=0)
    p = FDTDParams(lambda0)
    omega_probe = 2pi * p.c0 / Float64(lambda0)
    bins = omega_bins === nothing ?
           collect(omega_probe .* (1 .+ range(-0.04, 0.04; length=Int(spectrum_bins)))) :
           Float64.(collect(omega_bins))
    center = argmin(abs.(bins .- omega_probe))
    empty3 = Array{ComplexF64}(undef, 0, 0, 0)
    empty2 = Matrix{Float64}(undef, 0, 0)
    plan = Int.(collect(frame_indices))
    skip = Int(frame_skip)
    if skip <= 0 && length(plan) > 1
        skip = max(1, plan[2] - plan[1])
    end
    return ProbeReadout(Float64(pre), Float64(post), mode, bins, Int(center), Int(trace_stride),
                        Int(frame_target), skip, store_frames, Float64(lambda0),
                        Float64(probe_amplitude), Float64(probe_tau),
                        false, 0, 0, copy(empty2), copy(empty2), 1.0,
                        CPUBackend(), Float64[], copy(empty2), copy(empty2),
                        copy(empty3), copy(empty3), copy(empty3), copy(empty3),
                        copy(empty3), copy(empty3), copy(empty3), copy(empty3),
                        copy(empty2), copy(empty2), copy(empty2), copy(empty2), copy(empty2), copy(empty2),
                        Float64[], Float64[], Float64[], Float64[], Float64[],
                        Int[], Float64[], Any[], Any[], Any[], Any[], Any[], Any[],
                        plan, 1, Float64(slice_z_position), Int(final_step))
end

# Reference trace sampling: first step, every trace_stride-th step, and the final
# step (`n == 1 || n % stride == 0 || n == steps_probe`).
_trace_due(m::ProbeReadout, n::Integer) =
    n == 1 || n % m.trace_stride == 0 || (m.final_step > 0 && Int(n) == m.final_step)

# Allocate the plane buffers once the grid is known. The mode weight defaults to a
# uniform profile (area-weighted plane mean) when no eigenmode is supplied.
function _probe_init!(m::ProbeReadout, sim)
    grid = sim.grid
    st = _state_from_monitor_view(sim)
    # Device path whenever the underlying FDTDState runs on a GPU backend. (An earlier
    # `!hasproperty(sim, :fields)` clause wrongly excluded the standard run_phase!
    # monitor view — which always carries :fields — so on GPU runs the readout silently
    # fell to the HOST path: 4 plane downloads + CPU DFT trig EVERY step, ~35 ms/step
    # and ~230 KB/step of host garbage vs ~1 ms on-device.)
    use_device = st !== nothing && is_gpu_backend(st.backend)
    b = use_device ? st.backend : CPUBackend()
    Ny = length(grid.y.centers)
    Nz = length(grid.z.centers)
    area = Matrix{Float64}(undef, Ny, Nz)
    modw = Matrix{Float64}(undef, Ny, Nz)
    mode_host = m.mode === nothing ? nothing : to_host(m.mode)
    have_mode = mode_host !== nothing && size(mode_host) == (Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny
        area[j, k] = _cell_width(grid.y, j) * _cell_width(grid.z, k)
        modw[j, k] = have_mode ? Float64(mode_host[j, k]) : 1.0
    end
    nb = length(m.omega_bins)
    m.Ny = Ny; m.Nz = Nz
    m.area_yz = area
    m.mode_w = modw
    norm = 0.0
    @inbounds for k in 1:Nz, j in 1:Ny
        norm += abs(modw[j, k]) * area[j, k]
    end
    m.mode_norm = norm > 0.0 ? norm : 1.0
    m.backend = b
    m.omega_bins_dev = use_device ? adapt_backend(b, Float64.(m.omega_bins)) : m.omega_bins
    m.area_yz_dev = use_device ? adapt_backend(b, area) : area
    m.mode_w_dev = use_device ? adapt_backend(b, modw) : modw
    m.dft_Ey_pre = zeros_backend(b, ComplexF64, Ny, Nz, nb)
    m.dft_Ez_pre = zeros_backend(b, ComplexF64, Ny, Nz, nb)
    m.dft_Hy_pre = zeros_backend(b, ComplexF64, Ny, Nz, nb)
    m.dft_Hz_pre = zeros_backend(b, ComplexF64, Ny, Nz, nb)
    m.dft_Ey_post = zeros_backend(b, ComplexF64, Ny, Nz, nb)
    m.dft_Ez_post = zeros_backend(b, ComplexF64, Ny, Nz, nb)
    m.dft_Hy_post = zeros_backend(b, ComplexF64, Ny, Nz, nb)
    m.dft_Hz_post = zeros_backend(b, ComplexF64, Ny, Nz, nb)
    m.energy_pre = zeros_backend(b, Float64, Ny, Nz)
    m.energy_post = zeros_backend(b, Float64, Ny, Nz)
    m.work_Ey_pre = zeros_backend(b, Float64, Ny, Nz)
    m.work_Ez_pre = zeros_backend(b, Float64, Ny, Nz)
    m.work_Ey_post = zeros_backend(b, Float64, Ny, Nz)
    m.work_Ez_post = zeros_backend(b, Float64, Ny, Nz)
    m.initialized = true
    return m
end

# Stack 2-D frames into a frame-LAST cube: (d1, d2, n_frames), the reference's
# HDF5 layout (Nx, Ny, frames) / (Ny, Nz, frames).
function _stack_frames(frames)
    isempty(frames) && return Array{Float32}(undef, 0, 0, 0)
    out = Array{Float32}(undef, size(frames[1])..., length(frames))
    for i in eachindex(frames)
        out[axes(frames[i])..., i] = Float32.(frames[i])
    end
    return out
end

function _probe_frame_due(m::ProbeReadout, sim)
    m.frame_target > 0 || return false
    if !isempty(m.frame_plan)
        m.frame_ptr <= length(m.frame_plan) || return false
        return sim.n == m.frame_plan[m.frame_ptr]
    end
    skip = m.frame_skip > 0 ? m.frame_skip : max(1, cld(max(sim.n, 1), m.frame_target))
    return sim.n % skip == 0 && length(m.frame_indices) < m.frame_target
end

function _record_probe_frame!(m::ProbeReadout, sim)
    _probe_frame_due(m, sim) || return m
    isempty(m.frame_plan) || (m.frame_ptr += 1)
    fields = _view_fields(sim)
    grid = sim.grid
    kc = isnan(m.slice_z_position) ? cld(length(grid.z.centers), 2) :
         _source_index(grid.z, m.slice_z_position)
    ipre = _plane_index(grid, :x, m.x_pre)
    ipost = _plane_index(grid, :x, m.x_post)
    push!(m.frame_indices, sim.n)
    push!(m.frame_times, sim.t)
    m.store_frames || return m
    # |E| Float32 slices in native (Nx, Ny) / (Ny, Nz) orientation, matching the
    # reference's abs.(view(...)) movie frames.
    push!(m.Ey_xy, Float32.(abs.(plane_to_host(fields.Ey, :z, kc))))
    push!(m.Ez_xy, Float32.(abs.(plane_to_host(fields.Ez, :z, kc))))
    push!(m.Ey_yz_pre, Float32.(abs.(plane_to_host(fields.Ey, :x, ipre))))
    push!(m.Ez_yz_pre, Float32.(abs.(plane_to_host(fields.Ez, :x, ipre))))
    push!(m.Ey_yz_post, Float32.(abs.(plane_to_host(fields.Ey, :x, ipost))))
    push!(m.Ez_yz_post, Float32.(abs.(plane_to_host(fields.Ez, :x, ipost))))
    return m
end

# Must mirror the `use_device` gate in _probe_init! (buffers and record path have to
# agree on where the DFT accumulators live).
_probe_device_path(m::ProbeReadout, sim) = begin
    st = _state_from_monitor_view(sim)
    st !== nothing && is_gpu_backend(st.backend)
end

function _record_probe_host!(m::ProbeReadout, sim)
    t = Float64(sim.t)
    dt = Float64(sim.dt)
    grid = sim.grid
    fields = _view_fields(sim)
    Ey = _field_array(fields, :Ey); Ez = _field_array(fields, :Ez)
    Hy = _field_array(fields, :Hy); Hz = _field_array(fields, :Hz)
    ipre = _plane_index(grid, :x, m.x_pre)
    ipost = _plane_index(grid, :x, m.x_post)
    nb = length(m.omega_bins)
    phases = Vector{ComplexF64}(undef, nb)
    @inbounds for q in 1:nb
        phases[q] = cis(-m.omega_bins[q] * t) * dt
    end
    @inbounds for k in 1:m.Nz, j in 1:m.Ny
        eyp = Float64(Ey[ipre, j, k]); ezp = Float64(Ez[ipre, j, k])
        hyp = Float64(Hy[ipre, j, k]); hzp = Float64(Hz[ipre, j, k])
        eyq = Float64(Ey[ipost, j, k]); ezq = Float64(Ez[ipost, j, k])
        hyq = Float64(Hy[ipost, j, k]); hzq = Float64(Hz[ipost, j, k])
        m.energy_pre[j, k] += (eyp * hzp - ezp * hyp) * dt
        m.energy_post[j, k] += (eyq * hzq - ezq * hyq) * dt
        for q in 1:nb
            phq = phases[q]
            m.dft_Ey_pre[j, k, q] += eyp * phq
            m.dft_Ez_pre[j, k, q] += ezp * phq
            m.dft_Hy_pre[j, k, q] += hyp * phq
            m.dft_Hz_pre[j, k, q] += hzp * phq
            m.dft_Ey_post[j, k, q] += eyq * phq
            m.dft_Ez_post[j, k, q] += ezq * phq
            m.dft_Hy_post[j, k, q] += hyq * phq
            m.dft_Hz_post[j, k, q] += hzq * phq
        end
    end
    if _trace_due(m, sim.n)
        eypre = 0.0; ezpre = 0.0; eypost = 0.0; ezpost = 0.0
        @inbounds for k in 1:m.Nz, j in 1:m.Ny
            w = m.mode_w[j, k] * m.area_yz[j, k]
            eypre += Float64(Ey[ipre, j, k]) * w
            ezpre += Float64(Ez[ipre, j, k]) * w
            eypost += Float64(Ey[ipost, j, k]) * w
            ezpost += Float64(Ez[ipost, j, k]) * w
        end
        inv = 1.0 / m.mode_norm
        push!(m.t_trace, t)
        push!(m.Ey_pre, eypre * inv); push!(m.Ez_pre, ezpre * inv)
        push!(m.Ey_post, eypost * inv); push!(m.Ez_post, ezpost * inv)
    end
    _record_probe_frame!(m, sim)
    return m
end

function _record_probe_device!(m::ProbeReadout, sim)
    t = Float64(sim.t)
    dt = Float64(sim.dt)
    grid = sim.grid
    fields = _view_fields(sim)
    ipre = _plane_index(grid, :x, m.x_pre)
    ipost = _plane_index(grid, :x, m.x_post)
    _ka_probe_dft_accumulate!(m.backend, fields,
                              m.dft_Ey_pre, m.dft_Ez_pre, m.dft_Hy_pre, m.dft_Hz_pre,
                              m.dft_Ey_post, m.dft_Ez_post, m.dft_Hy_post, m.dft_Hz_post,
                              m.energy_pre, m.energy_post, m.omega_bins_dev,
                              ipre, ipost, t, dt)
    if _trace_due(m, sim.n)
        _ka_probe_trace_plane!(m.backend,
                               m.work_Ey_pre, m.work_Ez_pre, m.work_Ey_post, m.work_Ez_post,
                               fields, m.mode_w_dev, m.area_yz_dev, ipre, ipost)
        inv = 1.0 / m.mode_norm
        push!(m.t_trace, t)
        push!(m.Ey_pre, reduce_sum(m.work_Ey_pre) * inv)
        push!(m.Ez_pre, reduce_sum(m.work_Ez_pre) * inv)
        push!(m.Ey_post, reduce_sum(m.work_Ey_post) * inv)
        push!(m.Ez_post, reduce_sum(m.work_Ez_post) * inv)
    end
    _record_probe_frame!(m, sim)
    return m
end

function record!(m::ProbeReadout, sim)
    m.initialized || _probe_init!(m, sim)
    _probe_device_path(m, sim) ? _record_probe_device!(m, sim) : _record_probe_host!(m, sim)
    return m
end

_monitor_due(::ProbeReadout, n::Integer) = true
_monitor_needs_sync(::ProbeReadout, n::Integer) = false

function _trace_moments(t, y)
    isempty(t) && return (delay_fs=NaN, rms_fs=NaN)
    w = abs2.(Float64.(y))
    sw = sum(w)
    sw <= 0 && return (delay_fs=NaN, rms_fs=NaN)
    mu = sum(Float64.(t) .* w) / sw
    sig = sqrt(max(sum(((Float64.(t) .- mu).^2) .* w) / sw, 0.0))
    return (delay_fs=mu * 1e15, rms_fs=sig * 1e15)
end

# Per-bin complex Poynting cross-power, Σ_jk (Êy·conj(Ĥz) − Êz·conj(Ĥy))·dA.
function _plane_poynting(m::ProbeReadout, dEy, dEz, dHy, dHz)
    nb = length(m.omega_bins)
    Sx = zeros(ComplexF64, nb)
    @inbounds for q in 1:nb
        acc = ComplexF64(0)
        for k in 1:m.Nz, j in 1:m.Ny
            acc += (dEy[j, k, q] * conj(dHz[j, k, q]) -
                    dEz[j, k, q] * conj(dHy[j, k, q])) * m.area_yz[j, k]
        end
        Sx[q] = acc
    end
    return Sx
end

function _plane_net_flux(energy, area)
    net = 0.0
    @inbounds for k in axes(energy, 2), j in axes(energy, 1)
        net += energy[j, k] * area[j, k]
    end
    return net
end

# Mode-overlap-weighted complex amplitude of a component at one bin (Jones readout).
function _jones_overlap(m::ProbeReadout, dft, bin::Integer)
    acc = ComplexF64(0)
    @inbounds for k in 1:m.Nz, j in 1:m.Ny
        acc += dft[j, k, bin] * (m.mode_w[j, k] * m.area_yz[j, k])
    end
    return acc / m.mode_norm
end

function _probe_host_buffers(m::ProbeReadout)
    m.initialized || return nothing
    is_gpu_backend(m.backend) && synchronize(m.backend)
    return (;
        dft_Ey_pre=to_host(m.dft_Ey_pre),
        dft_Ez_pre=to_host(m.dft_Ez_pre),
        dft_Hy_pre=to_host(m.dft_Hy_pre),
        dft_Hz_pre=to_host(m.dft_Hz_pre),
        dft_Ey_post=to_host(m.dft_Ey_post),
        dft_Ez_post=to_host(m.dft_Ez_post),
        dft_Hy_post=to_host(m.dft_Hy_post),
        dft_Hz_post=to_host(m.dft_Hz_post),
        energy_pre=to_host(m.energy_pre),
        energy_post=to_host(m.energy_post),
    )
end

function probe_shot(m::ProbeReadout; state_label::AbstractString="probe",
                    is_reference::Bool=false, incident_energy=nothing, incident_sx=nothing,
                    normalization_method::AbstractString="gold_standard_bare_waveguide_reference_netflux",
                    dt::Real=NaN, steps_probe::Integer=0, absorbed_energy=nothing)
    nb = length(m.omega_bins)
    buf = _probe_host_buffers(m)
    Sx_pre = buf !== nothing ? _plane_poynting(m, buf.dft_Ey_pre, buf.dft_Ez_pre, buf.dft_Hy_pre, buf.dft_Hz_pre) :
             zeros(ComplexF64, nb)
    Sx_post = buf !== nothing ? _plane_poynting(m, buf.dft_Ey_post, buf.dft_Ez_post, buf.dft_Hy_post, buf.dft_Hz_post) :
              zeros(ComplexF64, nb)
    net_pre = buf !== nothing ? _plane_net_flux(buf.energy_pre, m.area_yz) : 0.0
    net_post = buf !== nothing ? _plane_net_flux(buf.energy_post, m.area_yz) : 0.0

    # Bare-waveguide reference normalization: the reference shot defines incident as
    # its own pure-forward pre-plane net; subsequent shots reuse it.
    incident = incident_energy === nothing ? net_pre : Float64(incident_energy)
    Sx_inc = incident_sx === nothing ? copy(Sx_pre) : convert(Vector{ComplexF64}, incident_sx)
    denom = abs(incident) > 0.0 ? incident : eps(Float64)
    reflected = incident - net_pre          # forward energy missing at the pre-plane
    transmitted = net_post                  # post-plane net forward
    T = transmitted / denom
    R = reflected / denom
    # The reference's A is the PHYSICAL film absorption (per-cell E·J work × cell
    # volume, integrated over the run), not the 1−T−R closure — so T+R+A < 1 by the
    # ring-down/scattering remainder, and the bare reference shot has A = 0 exactly.
    absorbed = absorbed_energy === nothing ? max(incident - transmitted - reflected, 0.0) :
               Float64(absorbed_energy)
    A = absorbed / denom

    # Per-bin spectral T/R from the true complex Poynting, with a RELATIVE floor on
    # the (physically tiny) incident cross-power so only empty bins go NaN.
    inc_re = real.(Sx_inc)
    peak = isempty(inc_re) ? 0.0 : maximum(abs.(inc_re))
    den_floor = 1.0e-12 * peak
    Tomega = fill(NaN, nb)
    Romega = fill(NaN, nb)
    @inbounds for q in 1:nb
        den = inc_re[q]
        if abs(den) > den_floor
            Tomega[q] = real(Sx_post[q]) / den
            Romega[q] = (den - real(Sx_pre[q])) / den
        end
    end

    q0 = m.center_bin
    if buf !== nothing
        jEy_far = _jones_overlap(m, buf.dft_Ey_post, q0)
        jEz_far = _jones_overlap(m, buf.dft_Ez_post, q0)
        jEy_kerr = _jones_overlap(m, buf.dft_Ey_pre, q0)
        jEz_kerr = _jones_overlap(m, buf.dft_Ez_pre, q0)
    else
        jEy_far = jEz_far = jEy_kerr = jEz_kerr = ComplexF64(0)
    end
    far = probe_jones_angles_deg(jEy_far, jEz_far)
    ker = probe_jones_angles_deg(jEy_kerr, jEz_kerr)

    input_stats = _trace_moments(m.t_trace, m.Ez_pre)
    trans_stats = _trace_moments(m.t_trace, m.Ez_post)
    return (;
        T=T, R=R, A=A, T_plus_R_plus_A=T + R + A,
        T_omega=Tomega, R_omega=Romega,
        theta_faraday_deg=far.rotation_deg, theta_kerr_deg=ker.rotation_deg,
        ellipticity_faraday_deg=far.ellipticity_deg, ellipticity_kerr_deg=ker.ellipticity_deg,
        jones_faraday_Ey_re_im=[real(jEy_far), imag(jEy_far)],
        jones_faraday_Ez_re_im=[real(jEz_far), imag(jEz_far)],
        jones_kerr_Ey_re_im=[real(jEy_kerr), imag(jEy_kerr)],
        jones_kerr_Ez_re_im=[real(jEz_kerr), imag(jEz_kerr)],
        Sx_incident_re=real.(Sx_inc), Sx_incident_im=imag.(Sx_inc),
        Sx_pre_net_re=real.(Sx_pre), Sx_pre_net_im=imag.(Sx_pre),
        Sx_post_net_re=real.(Sx_post), Sx_post_net_im=imag.(Sx_post),
        Sx_inc=Sx_inc,
        incident_energy_J=incident,
        transmitted_energy_J=transmitted,
        reflected_energy_J=reflected,
        absorbed_energy_J=absorbed,
        net_energy_pre_J=net_pre,
        net_energy_post_J=net_post,
        group_delay_fs=trans_stats.delay_fs - input_stats.delay_fs,
        pulse_broadening_fs=trans_stats.rms_fs - input_stats.rms_fs,
        throughput_loss=1.0 - T,
        input_pulse_rms_fs=input_stats.rms_fs,
        transmitted_pulse_rms_fs=trans_stats.rms_fs,
        Ey_trace_pre=copy(m.Ey_pre), Ez_trace_pre=copy(m.Ez_pre),
        Ey_trace_post=copy(m.Ey_post), Ez_trace_post=copy(m.Ez_post),
        trace_time_s=copy(m.t_trace),
        omega_bins_rad_s=copy(m.omega_bins),
        x_pre_um=m.x_pre * 1e6, x_post_um=m.x_post * 1e6,
        is_reference=Int8(is_reference ? 1 : 0),
        state_label=String(state_label),
        normalization_method=String(normalization_method),
        dt=Float64(dt), lambda_nm=m.lambda0 * 1e9,
        probe_amplitude_V_m=m.probe_amplitude,
        probe_tau_fs=m.probe_tau * 1e15,
        probe_trace_stride=m.trace_stride,
        probe_frame_target=m.frame_target,
        probe_frame_skip=m.frame_skip,
        probe_frame_count=length(m.frame_indices),
        probe_frame_indices=copy(m.frame_indices),
        probe_frame_times_s=copy(m.frame_times),
        steps_probe=Int(steps_probe),
        pole_fit_note="probe diagonal and magneto-optic ADE poles fitted at the probe wavelength",
        frames=(;
            Ey_xy=_stack_frames(m.Ey_xy), Ez_xy=_stack_frames(m.Ez_xy),
            Ey_yz_pre=_stack_frames(m.Ey_yz_pre), Ez_yz_pre=_stack_frames(m.Ez_yz_pre),
            Ey_yz_post=_stack_frames(m.Ey_yz_post), Ez_yz_post=_stack_frames(m.Ez_yz_post),
        ),
    )
end

# Reference contrast sign convention: deltas are INITIAL − SWITCHED.
function probe_contrast(initial, switched)
    return (;
        delta_T=initial.T - switched.T,
        delta_R=initial.R - switched.R,
        delta_A=initial.A - switched.A,
        delta_theta_faraday_deg=initial.theta_faraday_deg - switched.theta_faraday_deg,
        delta_theta_kerr_deg=initial.theta_kerr_deg - switched.theta_kerr_deg,
        delta_ellipticity_faraday_deg=initial.ellipticity_faraday_deg - switched.ellipticity_faraday_deg,
        delta_ellipticity_kerr_deg=initial.ellipticity_kerr_deg - switched.ellipticity_kerr_deg,
        delta_group_delay_fs=initial.group_delay_fs - switched.group_delay_fs,
        delta_pulse_broadening_fs=initial.pulse_broadening_fs - switched.pulse_broadening_fs,
        initial_T_R_A=[initial.T, initial.R, initial.A],
        switched_T_R_A=[switched.T, switched.R, switched.A],
        initial_T_plus_R_plus_A=initial.T_plus_R_plus_A,
        switched_T_plus_R_plus_A=switched.T_plus_R_plus_A,
    )
end

monitor_data(m::ProbeReadout) = probe_shot(m)
