# Generates the result figures used in the README / docs by running the actual
# simulations and emitting lightweight SVG plots (no plotting dependencies).
# Run with:  julia --project=. figs/make_figures.jl
using MagnetoPhotonic
using Printf

const ASSETS = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(ASSETS)

# ---------------------------------------------------------------- tiny SVG plotter
_vir = ((0.267,0.005,0.329),(0.231,0.322,0.545),(0.128,0.567,0.551),(0.369,0.789,0.383),(0.993,0.906,0.144))
function _cmap(v)
    v = clamp(v, 0.0, 1.0); s = v*(length(_vir)-1); i = clamp(floor(Int,s)+1,1,length(_vir)-1); f = s-(i-1)
    a,b = _vir[i],_vir[i+1]
    @sprintf("rgb(%d,%d,%d)", round(Int,255*(a[1]+f*(b[1]-a[1]))), round(Int,255*(a[2]+f*(b[2]-a[2]))), round(Int,255*(a[3]+f*(b[3]-a[3]))))
end

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

function heatmap_svg(path, M; title="", xlabel="", ylabel="", cell=5)
    nx,ny = size(M); mx = maximum(abs,M); mx==0 && (mx=1.0)
    ml,mt,cb = 8,34,26; W = ml + nx*cell + cb + 44; H = mt + ny*cell + 30
    open(path,"w") do io
        println(io,"""<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" viewBox="0 0 $W $H" font-family="sans-serif"><rect width="$W" height="$H" fill="white"/>""")
        for j in 1:ny, i in 1:nx
            x = ml+(i-1)*cell; y = mt+(ny-j)*cell
            println(io,"""<rect x="$x" y="$y" width="$cell" height="$cell" fill="$(_cmap(abs(M[i,j])/mx))"/>""")
        end
        cbx = ml+nx*cell+16
        for s in 0:49
            y = mt + (ny*cell)*(1-(s+1)/50); h = ceil(ny*cell/50)+1
            println(io,"""<rect x="$cbx" y="$y" width="14" height="$h" fill="$(_cmap(s/49))"/>""")
        end
        println(io,"""<text x="$(cbx+20)" y="$(mt+8)" font-size="11">max</text><text x="$(cbx+20)" y="$(mt+ny*cell)" font-size="11">0</text>""")
        println(io,"""<text x="$(ml+nx*cell/2)" y="22" font-size="16" text-anchor="middle" font-weight="bold">$title</text>""")
        println(io,"""<text x="$(ml+nx*cell/2)" y="$(H-8)" font-size="12" text-anchor="middle">$xlabel</text>""")
        println(io,"</svg>")
    end
    println("wrote ", path)
end

# ------------------------------------------------------------------- 1) 2D field
let
    p = FDTDParams()
    pw = GaussianPulse(; amplitude=1.0, tau=10e-15, t0=40e-15, omega=2pi*p.c0/800e-9)
    sim = Simulation(; cell=(2e-6,1.5e-6), dx=20e-9, dimension=2, mode=:TM,
                     sources=[PlaneSource(pw,:Ez; axis=:x, position=0.3e-6)], boundary=PML(10), courant=0.4)
    run!(sim, 6000)
    heatmap_svg(joinpath(ASSETS,"field2d.svg"), Float64.(sim.fields.Ez);
                title="2-D TM plane wave  |Ez|", xlabel="x  (100 x 75 cells, CPML borders)")
end

# ------------------------------------------------------------------- 2) convergence
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

# ------------------------------------------------------------------- 3) switching
let
    gd = MagnetoOpticModel().params
    tm,re,Tmin,invdT,N = build_m_eq_lut(gd)
    mTM = (lookup_m_eq_lut(tm,gd.T0,Tmin,invdT,N), 0.0, 1e-4)
    mRE = (lookup_m_eq_lut(re,gd.T0,Tmin,invdT,N), 0.0, 1e-4)
    dt = 0.5e-15; nst = 5600
    t = Float64[]; a = Float64[]; b = Float64[]
    for n in 1:nst
        T = n < 1700 ? 1200.0 : max(300.0, 1200.0-(n-1700)*0.4)
        r = llb_step(mTM..., mRE..., T,T,T, gd, tm,re,Tmin,invdT,N, dt, 2)
        mTM=(r[1],r[2],r[3]); mRE=(r[4],r[5],r[6])
        if n % 20 == 0; push!(t, n*dt*1e12); push!(a, mTM[1]); push!(b, mRE[1]); end
    end
    lineplot_svg(joinpath(ASSETS,"switching.svg"),
        [(x=t, y=a, label="m_TM,x  (FeCo)"), (x=t, y=b, label="m_RE,x  (Gd)")];
        title="All-optical switching (4TM + LLB)", xlabel="time  (ps)", ylabel="reduced magnetization m_x")
end

# ------------------------------------------------------------------- 4) device geometry
let
    dev = not_gate_60um()
    write_plan_svg(joinpath(ASSETS,"device.svg"), dev.scene)
    println("wrote ", joinpath(ASSETS,"device.svg"))
end

println("done.")
