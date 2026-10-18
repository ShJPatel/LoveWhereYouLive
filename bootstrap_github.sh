#!/usr/bin/env bash
set -e

REPO_NAME=${1:-"sa-apartment-hunter"}
echo "Initializing Git repository and pushing to GitHub as $REPO_NAME..."

git init
git branch -M main
git add .
git commit -m "feat: initial San Antonio multi-criteria rental hunter pipeline (v2)"

# Create repository on GitHub using GitHub CLI (requires `gh auth login`)
gh repo create "$REPO_NAME" --private --source=. --remote=origin --push

echo "Creating initial GitHub Issues from ROADMAP backlog..."

gh issue create --title "[CRITERION] Distinguish W/D Machines Included vs Hookups Only" \
  --label "enhancement" \
  --body "Refine \`criterion_required_appliances_gatekeeper\` in \`src/criteria/__init__.py\` to distinguish between units with actual washer/dryer appliances included vs. units offering W/D connections/hookups only."

gh issue create --title "[CRITERION] Edwards Aquifer Water Softener & RO Filtration Detector" \
  --label "enhancement" \
  --body "Add a scoring module that rewards built-in water softeners, softener loops, or reverse osmosis filtration to protect appliances from San Antonio hard water."

gh issue create --title "[CRITERION] Pet-Friendliness & 100% Hard Flooring Scorer" \
  --label "enhancement" \
  --body "Add a module scoring cat-friendly policies, flagging high monthly pet rent, and rewarding 100% luxury vinyl plank (LVP) / tile / hardwood flooring (no carpet)."

gh issue create --title "[CRITERION] Airport Flight Path & Freight Rail Horn Buffer" \
  --label "enhancement" \
  --body "Pull \`railway=rail\` from OpenStreetMap and airport approach corridors to penalize listings within 400m of train horn zones or low-altitude flight paths."

gh issue create --title "[FEATURE] Automated Gemini 2.5 Flash Kitchen & Floorplan Vision Grader" \
  --label "enhancement" \
  --body "Integrate multi-modal image inspection in \`src/pipeline.py\` for the top 25 surviving candidates to verify open floor plans, kitchen islands, and stove/countertop quality."

echo "Done! Your repository and Issue Board are live."
