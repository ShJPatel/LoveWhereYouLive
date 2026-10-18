# 🏡 San Antonio Apartment & Townhome Hunter (`sa-apartment-hunter`)

An automated, extensible geospatial + AI pipeline to find and score apartments and townhomes in San Antonio, TX, tailored for commuting to **Southwest Research Institute (SwRI)**.

## ✨ Features
- **Open-in-Colab One-Click Notebook:** Open `notebooks/San_Antonio_Apartment_Hunter_Pipeline-v2.ipynb` directly in Google Colab via the `Open in Colab` badge once pushed to GitHub.
- **Headless CLI & GitHub Actions Runner:** Executes automatically via `python -m src.pipeline` or every Monday morning via `.github/workflows/weekly_hunt.yml`.
- **Agent-Ready Architecture (`AGENTS.md`):** Modular criteria functions in `src/criteria/` and declarative weights in `config.yaml` make it effortless for AI coding agents (Claude Code, GitHub Copilot Workspace, Cursor, Gemini) to pick up a GitHub Issue and submit a PR.

## 🚀 Quick Start (Push to Your GitHub in 30 Seconds)
Unzip this repository, open a terminal in the folder, and run:
```bash
chmod +x bootstrap_github.sh
./bootstrap_github.sh sa-apartment-hunter
```
*(Requires the [GitHub CLI (`gh`)](https://cli.github.com/) authenticated via `gh auth login`. It will create your private repo, push the code, and automatically populate 5 GitHub Issues on your board!)*
