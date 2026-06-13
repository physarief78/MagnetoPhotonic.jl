using Documenter

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ BEFORE YOU PUSH: verify the GitHub username/repo in the three places       │
# │ marked "VERIFY" below. It is set to "physarief" (from the author's email   │
# │ handle); change it if your GitHub account or repo name differs. The        │
# │ Documentation GitHub Action then deploys to                                │
# │   https://<USER>.github.io/MagnetoPhotonic.jl                              │
# │ (you must also add a DOCUMENTER_KEY deploy secret — see the README).       │
# └──────────────────────────────────────────────────────────────────────────┘
const GH_USER = "physarief"          # VERIFY: your GitHub username
const GH_REPO = "MagnetoPhotonic.jl" # VERIFY: your repository name

makedocs(
    sitename = "MagnetoPhotonic.jl",
    authors = "Muhammad Arief Mulyana",
    repo = Remotes.GitHub(GH_USER, GH_REPO),
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://$(GH_USER).github.io/$(GH_REPO)",  # VERIFY
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Introduction" => "introduction.md",
        "Fundamentals" => "fundamentals.md",
        "Getting Started" => "getting_started.md",
        "Tutorials" => [
            "EM-FDTD" => "em_fdtd.md",
            "Magneto-Optic Switching" => "magneto_optics.md",
        ],
        "Capabilities" => "capabilities.md",
        "API Reference" => "api.md",
    ],
    # Keep the build resilient: missing-docstring / cross-ref issues warn instead of
    # erroring, so a local `docs/make.jl` run always produces a site.
    warnonly = true,
)

# Deploys to GitHub Pages (gh-pages branch) on push to `main` via .github/workflows/Documentation.yml.
deploydocs(
    repo = "github.com/$(GH_USER)/$(GH_REPO).git",  # VERIFY
    devbranch = "main",
    push_preview = true,
)
