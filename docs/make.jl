using Documenter

makedocs(
    sitename = "MagnetoPhotonic.jl",
    authors = "MIPA UNPAD Lab",
    # No git remote configured yet; once the repo is pushed, set e.g.
    #   repo = "github.com/<USER>/MagnetoPhotonic.jl"
    # and remove `remotes = nothing` to enable "edit on GitHub" source links.
    remotes = nothing,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        # canonical = "https://<USER>.github.io/MagnetoPhotonic.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "EM-FDTD Tutorial" => "em_fdtd.md",
        "Magneto-Optic Switching" => "magneto_optics.md",
        "API Reference" => "api.md",
    ],
    warnonly = true,
)

# To publish to GitHub Pages, set <USER> and uncomment (configure a DOCUMENTER_KEY secret):
# deploydocs(repo = "github.com/<USER>/MagnetoPhotonic.jl.git", devbranch = "main")
