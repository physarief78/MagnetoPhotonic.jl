# Generates the result figures used in the README / docs by running the actual
# simulations and emitting lightweight SVG plots (no plotting dependencies).
# Run with:  julia --project=. figs/make_figures.jl
using MagnetoPhotonic
using Printf

const ASSETS = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(ASSETS)

# ---------------------------------------------------------------- colormaps
_vir = ((0.267,0.005,0.329),(0.231,0.322,0.545),(0.128,0.567,0.551),(0.369,0.789,0.383),(0.993,0.906,0.144))
function _cmap(v)
    v = clamp(v, 0.0, 1.0); s = v*(length(_vir)-1); i = clamp(floor(Int,s)+1,1,length(_vir)-1); f = s-(i-1)
    a,b = _vir[i],_vir[i+1]
    @sprintf("rgb(%d,%d,%d)", round(Int,255*(a[1]+f*(b[1]-a[1]))), round(Int,255*(a[2]+f*(b[2]-a[2]))), round(Int,255*(a[3]+f*(b[3]-a[3]))))
end

# Signed diverging map (blue -0 white +0 red), v in [-1,1]. Used for oscillating
# wave fields so crests/troughs (and the interference pattern) are visible.
function _div(v)
    v = clamp(v, -1.0, 1.0); t = abs(v)
    w = (247,247,247)
    c = v < 0 ? (33,102,172) : (178,24,43)
    @sprintf("rgb(%d,%d,%d)", round(Int,w[1]+t*(c[1]-w[1])), round(Int,w[2]+t*(c[2]-w[2])), round(Int,w[3]+t*(c[3]-w[3])))
end

# ---------------------------------------------------------------- tiny line plotter
function lineplot_svg(path, series; title="", xlabel="", ylabel="", xlog=false, ylog=false, legend=true)
    W,H,ml,mr,mt,mb = 760,440,72,28,42,56; pw,ph = W-ml-mr, H-mt-mb
    cols = ("#1f77b4","#d62728","#2ca02c","#9467bd")
    fx = xlog ? log10 : identity; fy = ylog ? log10 : identity
    allx = reduce(vcat,[fx.(s.x) for s in series]); ally = reduce(vcat,[fy.(s.y) for s in series])
    xmin,xmax = extrema(allx); ymin,ymax = extrema(ally)
    xmax==xmin && (xmax+=1); ymax==ymin && (ymax+=1)
    pad=0.04*(ymax-ymin); ymin-=pad; ymax+=pad
    px(x)= ml + (fx(x)-xmin)/(xmax-xmin)*pw
    py(y)= mt + ph - (fy(y)-ymin)/(ymax-ymin)*ph
    open(path,"w") do io
        println(io,"""<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" viewBox="0 0 $W $H" font-family="sans-serif"><rect width="$W" height="$H" fill="white"/>""")
        println(io,"""<rect x="$ml" y="$mt" width="$pw" height="$ph" fill="none" stroke="#999"/>""")
        for t in 0:4
            X=ml+t/4*pw; Y=mt+ph-t/4*ph; xv=xmin+t/4*(xmax-xmin); yv=ymin+t/4*(ymax-ymin)
            println(io,"""<line x1="$X" y1="$mt" x2="$X" y2="$(mt+ph)" stroke="#eee"/><line x1="$ml" y1="$Y" x2="$(ml+pw)" y2="$Y" stroke="#eee"/>""")
            xl = xlog ? @sprintf("%.0e",10.0^xv) : @sprintf("%.3g",xv)
            yl = ylog ? @sprintf("%.0e",10.0^yv) : @sprintf("%.3g",yv)
            println(io,"""<text x="$X" y="$(mt+ph+18)" font-size="12" text-anchor="middle" fill="#333">$xl</text>""")
            println(io,"""<text x="$(ml-8)" y="$(Y+4)" font-size="12" text-anchor="end" fill="#333">$yl</text>""")
        end
        for (k,s) in enumerate(series)
            c = cols[(k-1)%length(cols)+1]
            pts = join((@sprintf("%.1f,%.1f",px(s.x[i]),py(s.y[i])) for i in eachindex(s.x))," ")
            dash = get(s,:dash,false) ? " stroke-dasharray=\"7 5\"" : ""
            println(io,"""<polyline points="$pts" fill="none" stroke="$c" stroke-width="2.4"$dash/>""")
            if get(s,:markers,false)
                for i in eachindex(s.x); println(io,"""<circle cx="$(px(s.x[i]))" cy="$(py(s.y[i]))" r="3.5" fill="$c"/>"""); end
            end
        end
        println(io,"""<text x="$(ml+pw/2)" y="24" font-size="16" text-anchor="middle" font-weight="bold">$title</text>""")
        println(io,"""<text x="$(ml+pw/2)" y="$(H-14)" font-size="13" text-anchor="middle">$xlabel</text>""")
        println(io,"""<text x="18" y="$(mt+ph/2)" font-size="13" text-anchor="middle" transform="rotate(-90 18 $(mt+ph/2))">$ylabel</text>""")
        if legend
            for (k,s) in enumerate(series)
                c=cols[(k-1)%length(cols)+1]; ly=mt+16+(k-1)*18
                println(io,"""<line x1="$(ml+pw-150)" y1="$ly" x2="$(ml+pw-126)" y2="$ly" stroke="$c" stroke-width="3"/><text x="$(ml+pw-120)" y="$(ly+4)" font-size="12">$(s.label)</text>""")
            end
        end
        println(io,"</svg>")
    end
    println("wrote ", path)
end

# ------------------------------------------ signed field heatmap (+ optional barrier)
function field_svg(path, M; title="", xlabel="", cell=3, clip=0.5, mask=nothing)
    nx,ny = size(M); mx = maximum(abs,M); mx==0 && (mx=1.0); sc = 1.0/(clip*mx)
    ml,mt,cb = 8,34,26; W = ml + nx*cell + cb + 50; H = mt + ny*cell + 30
    open(path,"w") do io
        println(io,"""<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" viewBox="0 0 $W $H" font-family="sans-serif"><rect width="$W" height="$H" fill="white"/>""")
        for j in 1:ny, i in 1:nx
            x = ml+(i-1)*cell; y = mt+(ny-j)*cell
            col = (mask !== nothing && mask[i,j]) ? "#111111" : _div(M[i,j]*sc)
            println(io,"""<rect x="$x" y="$y" width="$cell" height="$cell" fill="$col"/>""")
        end
        cbx = ml+nx*cell+16
        for s in 0:49
            y = mt + (ny*cell)*(s/50); h = ceil(ny*cell/50)+1
            v = 1.0 - 2.0*(s/49)               # +1 (top) .. -1 (bottom)
            println(io,"""<rect x="$cbx" y="$y" width="14" height="$h" fill="$(_div(v))"/>""")
        end
        println(io,"""<text x="$(cbx+20)" y="$(mt+8)" font-size="11">+</text><text x="$(cbx+20)" y="$(mt+ny*cell)" font-size="11">−</text>""")
        println(io,"""<text x="$(ml+nx*cell/2)" y="22" font-size="16" text-anchor="middle" font-weight="bold">$title</text>""")
        println(io,"""<text x="$(ml+nx*cell/2)" y="$(H-8)" font-size="12" text-anchor="middle">$xlabel</text>""")
        println(io,"</svg>")
    end
    println("wrote ", path)
end

# --------------------------------------- two-panel all-optical-switching figure
function switching_svg(path, t, mTM, mRE, Te, Tl, Ts; MsTM=1.10, MsRE=0.55)
    W = 760; ml,mr = 78,150; pw = W-ml-mr
    p1t,p1h = 46, 230     # magnetization panel
    p2t,p2h = 330, 150    # temperature panel
    H = p2t + p2h + 40
    xmin,xmax = extrema(t)
    # ---- panel coordinate helpers
    px(x) = ml + (x-xmin)/(xmax-xmin)*pw
    # transient-ferromagnetic window = between FeCo and Gd zero crossings
    zc(y) = begin tc=NaN; for i in 2:length(y); if (y[i-1]<0)!=(y[i]<0); tc=t[i-1]+(t[i]-t[i-1])*(0-y[i-1])/(y[i]-y[i-1]); break; end; end; tc end
    Mfe = MsTM .* mTM; Mgd = MsRE .* mRE; Mnet = Mfe .+ Mgd
    cFe = zc(Mfe); cGd = zc(Mgd)
    open(path,"w") do io
        println(io,"""<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" viewBox="0 0 $W $H" font-family="sans-serif"><rect width="$W" height="$H" fill="white"/>""")
        # ===== panel 1: magnetization =====
        ymin,ymax = -1.25, 1.25
        py1(y) = p1t + p1h - (y-ymin)/(ymax-ymin)*p1h
        println(io,"""<rect x="$ml" y="$p1t" width="$pw" height="$p1h" fill="none" stroke="#999"/>""")
        if isfinite(cFe) && isfinite(cGd)
            xa,xb = px(min(cFe,cGd)), px(max(cFe,cGd))
            println(io,"""<rect x="$xa" y="$p1t" width="$(xb-xa)" height="$p1h" fill="#ffe08a" fill-opacity="0.55"/>""")
            println(io,"""<text x="$((xa+xb)/2)" y="$(p1t+16)" font-size="11" font-style="italic" text-anchor="middle" fill="#7a5c00">transient</text>""")
            println(io,"""<text x="$((xa+xb)/2)" y="$(p1t+28)" font-size="11" font-style="italic" text-anchor="middle" fill="#7a5c00">ferromagnet</text>""")
        end
        println(io,"""<line x1="$ml" y1="$(py1(0.0))" x2="$(ml+pw)" y2="$(py1(0.0))" stroke="#bbb" stroke-dasharray="3 3"/>""")
        for yv in (-1.0,-0.5,0.5,1.0)
            Y=py1(yv); println(io,"""<text x="$(ml-8)" y="$(Y+4)" font-size="11" text-anchor="end" fill="#333">$(@sprintf("%.1f",yv))</text>""")
        end
        polyline(io,c,xs,ys,py;dash=false) = begin
            pts=join((@sprintf("%.1f,%.1f",px(xs[i]),py(ys[i])) for i in eachindex(xs))," ")
            d = dash ? " stroke-dasharray=\"7 5\"" : ""
            println(io,"""<polyline points="$pts" fill="none" stroke="$c" stroke-width="2.2"$d/>""")
        end
        polyline(io,"#d62728",t,Mfe,py1)
        polyline(io,"#1f77b4",t,Mgd,py1)
        polyline(io,"#222222",t,Mnet,py1;dash=true)
        println(io,"""<text x="$(ml+pw/2)" y="26" font-size="16" text-anchor="middle" font-weight="bold">All-optical switching (0-D 4TM + LLB, single cell)</text>""")
        println(io,"""<text x="22" y="$(p1t+p1h/2)" font-size="12" text-anchor="middle" transform="rotate(-90 22 $(p1t+p1h/2))">M_x  (MA/m)</text>""")
        lx = ml+pw+12
        for (k,(lab,c)) in enumerate((("FeCo (TM)","#d62728"),("Gd (RE)","#1f77b4"),("Net M_x","#222222")))
            ly=p1t+18+(k-1)*18
            println(io,"""<line x1="$lx" y1="$ly" x2="$(lx+22)" y2="$ly" stroke="$c" stroke-width="3"/><text x="$(lx+27)" y="$(ly+4)" font-size="12">$lab</text>""")
        end
        # ===== panel 2: reservoir temperatures =====
        tmax = maximum(Te); ytop = ceil(tmax/500)*500
        py2(y) = p2t + p2h - (y-0.0)/(ytop-0.0)*p2h
        println(io,"""<rect x="$ml" y="$p2t" width="$pw" height="$p2h" fill="none" stroke="#999"/>""")
        for yv in 0:500:ytop
            Y=py2(yv); println(io,"""<line x1="$ml" y1="$Y" x2="$(ml+pw)" y2="$Y" stroke="#eee"/><text x="$(ml-8)" y="$(Y+4)" font-size="11" text-anchor="end" fill="#333">$(round(Int,yv))</text>""")
        end
        polyline(io,"#ff8c1a",t,Te,py2)
        polyline(io,"#2ca02c",t,Tl,py2)
        polyline(io,"#7a3fb3",t,Ts,py2)
        println(io,"""<text x="$(ml+pw/2)" y="$(p2t-8)" font-size="14" text-anchor="middle" font-weight="bold">Reservoir temperatures (4TM)</text>""")
        println(io,"""<text x="22" y="$(p2t+p2h/2)" font-size="12" text-anchor="middle" transform="rotate(-90 22 $(p2t+p2h/2))">T  (K)</text>""")
        println(io,"""<text x="$(ml+pw/2)" y="$(H-12)" font-size="13" text-anchor="middle">time  (ps)</text>""")
        for (k,(lab,c)) in enumerate((("electron T_e","#ff8c1a"),("lattice T_l","#2ca02c"),("spin T_s","#7a3fb3")))
            ly=p2t+18+(k-1)*18
            println(io,"""<line x1="$lx" y1="$ly" x2="$(lx+22)" y2="$ly" stroke="$c" stroke-width="3"/><text x="$(lx+27)" y="$(ly+4)" font-size="12">$lab</text>""")
        end
        # time ticks (shared)
        for f in 0:0.25:1
            xv = xmin+f*(xmax-xmin); X=px(xv)
            println(io,"""<text x="$X" y="$(p2t+p2h+16)" font-size="11" text-anchor="middle" fill="#333">$(@sprintf("%.1f",xv))</text>""")
        end
        println(io,"</svg>")
    end
    println("wrote ", path)
end

# ============================================================ 1) single-slit diffraction
let
    p = FDTDParams()
    pw = GaussianPulse(; amplitude=1.0, tau=8e-15, t0=24e-15, omega=2pi*p.c0/700e-9)
    Lx, Ly, dx = 4e-6, 3e-6, 20e-9
    yc = Ly/2; half_slit = 0.20e-6; bx0, bx1 = 1.95e-6, 2.05e-6
    sim = Simulation(; cell=((0.0,Lx),(0.0,Ly)), dx=dx, dimension=2, mode=:TM,
                     sources=[PlaneSource(pw, :Ez; axis=:x, position=0.5e-6)],
                     boundary=PML(10), courant=0.45, params=p)
    xs = sim.grid.x.centers; ys = sim.grid.y.centers
    mask = falses(length(xs), length(ys))
    for i in eachindex(xs), j in eachindex(ys)
        (bx0 <= xs[i] <= bx1 && !(yc-half_slit <= ys[j] <= yc+half_slit)) && (mask[i,j]=true)
    end
    run!(sim, 2200; callback = s->(s.fields.Ez[mask] .= 0.0))
    field_svg(joinpath(ASSETS,"slit.svg"), Float64.(sim.fields.Ez); mask=mask, clip=0.45,
              title="Single-slit diffraction — plane wave Ez through a PEC wall",
              xlabel="200 x 150 cells (dx=20 nm); incident wave from left, slit at center")
end

# ============================================================ 2) point source (sine x gaussian)
let
    p = FDTDParams()
    pw = GaussianPulse(; amplitude=1.0, tau=20e-15, t0=30e-15, omega=2pi*p.c0/700e-9)
    L, dx = 4e-6, 20e-9
    sim = Simulation(; cell=((0.0,L),(0.0,L)), dx=dx, dimension=2, mode=:TM,
                     sources=[PointSource(pw, :Ez, (L/2, L/2))],
                     boundary=PML(10), courant=0.45, params=p)
    run!(sim, 1300)
    field_svg(joinpath(ASSETS,"pointsource.svg"), Float64.(sim.fields.Ez); clip=0.5,
              title="Point source — sine wave modulated by a Gaussian pulse, radiating from center",
              xlabel="200 x 200 cells (dx=20 nm); cylindrical wavefronts absorbed by CPML")
end

# ============================================================ 3) convergence
let
    p = FDTDParams(); eta = sqrt(p.mu0/p.eps0); L=4e-6; x0=1e-6; sigma=0.25e-6; Tprop=3e-15
    function l2err(dx)
        grid = uniform_grid((0.0,L),dx); f = allocate_fields(grid); dt = cfl_dt(grid,p;courant=0.45)
        for i in eachindex(grid.x.centers)
            E = exp(-((grid.x.centers[i]-x0)/sigma)^2); f.Ez[i]=E; f.Dz[i]=p.eps0*E
            xh = i<length(grid.x.centers) ? grid.x.edges[i+1] : grid.x.edges[end]
            f.Hy[i] = -exp(-((xh+p.c0*dt/2-x0)/sigma)^2)/eta
        end
        for _ in 1:round(Int,Tprop/dt); update_H_1d!(f,grid,p,dt); update_E_1d!(f,grid,p,dt,ones(length(grid.x.centers))); end
        tf=round(Int,Tprop/dt)*dt; e=0.0; n=0.0
        for i in eachindex(grid.x.centers)
            ex=exp(-((grid.x.centers[i]-(x0+p.c0*tf))/sigma)^2); dc=grid.x.edges[i+1]-grid.x.edges[i]; e+=(f.Ez[i]-ex)^2*dc; n+=ex^2*dc
        end
        sqrt(e/n)
    end
    dxs = [40e-9,30e-9,20e-9,14e-9,10e-9,7e-9]; errs = l2err.(dxs)
    ref = errs[1] .* (dxs ./ dxs[1]).^2
    lineplot_svg(joinpath(ASSETS,"convergence.svg"),
        [(x=dxs.*1e9, y=errs, label="measured L2", markers=true),
         (x=dxs.*1e9, y=ref, label="O(dx^2) reference", dash=true)];
        title="Grid convergence (source-free Yee)", xlabel="dx  (nm)", ylabel="relative L2 error",
        xlog=true, ylog=true)
end

# ============================================================ 4) asymmetric AOS switching
# Single material cell driven by the full four-temperature model: an absorbed-power
# Gaussian heats the electron bath, which feeds the TM (FeCo, 100 fs) and RE (Gd,
# 430 fs) spin baths on their *distinct* demagnetization timescales. The element-
# resolved coupling + branch-selection channel make FeCo reverse FIRST and Gd follow,
# opening the transient-ferromagnetic window — exactly the reference behavior.
let
    gd = MagnetoOpticModel().params
    tm,re,Tmin,invdT,N = build_m_eq_lut(gd)
    m0TM = lookup_m_eq_lut(tm,gd.T0,Tmin,invdT,N)
    m0RE = lookup_m_eq_lut(re,gd.T0,Tmin,invdT,N)
    mTM=(m0TM,0.0,1e-4); mRE=(m0RE,0.0,1e-4)
    Te=gd.T0; Tl=gd.T0; TsTM=gd.T0; TsRE=gd.T0
    dt=0.5e-15; nst=16000
    pabs_peak=4.0e21; t_pulse=0.4e-12; tau_pulse=0.2e-12
    t=Float64[]; aTM=Float64[]; aRE=Float64[]; Tev=Float64[]; Tlv=Float64[]; Tsv=Float64[]
    for n in 1:nst
        tt=n*dt
        pabs = pabs_peak*exp(-((tt-t_pulse)/tau_pulse)^2)
        r = llb_step(mTM..., mRE..., Te, TsTM, TsRE, gd, tm,re,Tmin,invdT,N, dt, 2)
        mTM=(r[1],r[2],r[3]); mRE=(r[4],r[5],r[6])
        Q_TM = gd.Ms_TM*r[9]*(r[7]/dt); Q_RE = gd.Ms_RE*r[10]*(r[8]/dt)
        Te,Tl,TsTM,TsRE = update_4tm(Te,Tl,TsTM,TsRE,pabs,Q_TM,Q_RE,gd,dt)
        if n % 40 == 0
            push!(t, tt*1e12); push!(aTM, mTM[1]); push!(aRE, mRE[1])
            push!(Tev, Te); push!(Tlv, Tl); push!(Tsv, 0.5*(TsTM+TsRE))
        end
    end
    switching_svg(joinpath(ASSETS,"switching.svg"), t, aTM, aRE, Tev, Tlv, Tsv)
    @printf("  AOS: FeCo m_x %.2f->%.2f, Gd m_x %.2f->%.2f, Te_peak=%.0f K\n",
            aTM[1], aTM[end], aRE[1], aRE[end], maximum(Tev))
end

# ============================================================ 5) device geometry
let
    dev = not_gate_60um()
    write_plan_svg(joinpath(ASSETS,"device.svg"), dev.scene)
    println("wrote ", joinpath(ASSETS,"device.svg"))
end

println("done.")
