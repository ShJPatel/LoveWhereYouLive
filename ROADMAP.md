# Project Roadmap, Features & Issue Backlog

Use this file (or run `scripts/bootstrap_github_issues.sh` with the GitHub CLI `gh`) to track milestones and assign tasks to coding agents or collaborators.

## Milestone 1: Core Geospatial & Scraper Engine (Completed in v1/v2)
- [x] **M1.1:** OpenStreetMap (`OSMnx`) spatial KD-Tree indexing for San Antonio grocery stores, parks/greenway trails, and major highways (`EPSG:32614`).
- [x] **M1.2:** Automated rental listing harvest via `HomeHarvest` with bed/bath/budget filtering.
- [x] **M1.3:** Directional rush-hour commute model to Southwest Research Institute (SwRI - 6220 Culebra Rd).
- [x] **M1.4:** Hard requirement gatekeeper for **In-Unit Laundry** and **Dishwasher**.
- [x] **M1.5:** Bonus scoring modules for **Garage / Bike Storage** and **Costco Proximity**.

## Milestone 2: San Antonio Environmental & Utility Refinements (Next Up)
- [ ] **Issue #1: Distinguish "W/D Machines Included" vs "W/D Hookups Only"**
  - *Goal:* Parse listing descriptions and photo captions to separate units that provide actual washer/dryer appliances from those that only provide connections.
- [ ] **Issue #2: Edwards Aquifer Water Softener & RO Filtration Detector**
  - *Goal:* Add a bonus criterion detecting built-in water softeners, softener loops, or reverse-osmosis taps (critical for protecting espresso boilers and dishwashers from San Antonio's ~18 gpg hard limestone water).
- [ ] **Issue #3: Pet-Friendliness & 100% Hard Flooring Scorer**
  - *Goal:* Score cat-friendly policies, flag excessive monthly pet rent, and reward units with 100% luxury vinyl plank (LVP) / hardwood / tile (no bedroom carpet).
- [ ] **Issue #4: Airport Flight Path & Freight Rail Horn Buffer**
  - *Goal:* Query OpenStreetMap `railway=rail` and SAT / Lackland AFB approach vectors to penalize high-decibel noise corridors.

## Milestone 3: Live API & Multi-Modal Vision Upgrades
- [ ] **Issue #5: Automated Gemini 2.5 Flash Kitchen & Floorplan Vision Grader**
  - *Goal:* Fetch the top 25 surviving listings' photo URLs and pass kitchen/living room images to `gemini-2.5-flash` to verify open-concept island layouts, gas/induction ranges, and modern cabinetry.
- [ ] **Issue #6: Effective Rent ("Junk Fee") Calculator**
  - *Goal:* Estimate true monthly cost by detecting mandatory valet trash, bulk internet/tech packages, and Fetch package fees vs. private townhome leases.
- [ ] **Issue #7: Scheduled GitHub Action Weekly Scrape & Diff Alert**
  - *Goal:* Run `.github/workflows/weekly_hunt.yml` every Monday morning, commit newly discovered top-scoring rentals, and highlight new arrivals.
