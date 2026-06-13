using Documenter

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ The Documentation GitHub Action deploys the built site to                  │
# │   https://physarief78.github.io/MagnetoPhotonic.jl                         │
# │ via the gh-pages branch. For that to work the repo needs EITHER            │
# │ "Read and write" workflow permissions (Settings → Actions → General) so    │
# │ the default GITHUB_TOKEN can push gh-pages, OR a DOCUMENTER_KEY deploy      │
# │ secret (see the README). GH_USER/GH_REPO must match the actual repo.       │
# └──────────────────────────────────────────────────────────────────────────┘
const GH_USER = "physarief78"        # GitHub username (repo owner)
const GH_REPO = "MagnetoPhotonic.jl" # repository name

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
