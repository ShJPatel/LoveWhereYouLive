# Project Roadmap, Features & Issue Backlog

## Milestone 1: Core Geospatial & Scraper Engine (Completed)
- [x] **M1.1:** OpenStreetMap (`OSMnx`) spatial KD-Tree indexing for San Antonio grocery stores (excluding Walmart), parks/greenway trails, major highways, and railways (`EPSG:32614`).
- [x] **M1.2:** Automated rental listing harvest via `HomeHarvest` with bed/bath/budget filtering.
- [x] **M1.3:** Directional rush-hour commute model to Southwest Research Institute (SwRI - 6220 Culebra Rd).
- [x] **M1.4:** Hard requirement gatekeeper for **In-Unit Laundry** and **Dishwasher**.
- [x] **M1.5:** Bonus scoring modules for **Garage / Bike Storage** and **Costco Proximity**.

## Milestone 2: San Antonio Environmental, Utility & Financial Refinements (Completed)
- [x] **M2.1:** Exclude Walmart from grocery store proximity KD-Tree.
- [x] **M2.2:** Edwards Aquifer Water Softener & RO Filtration Detector (`criterion_water_softener_and_filtration`).
- [x] **M2.3:** Pet-Friendliness & 100% Hard Flooring Scorer (`criterion_pet_friendly_and_hard_flooring`).
- [x] **M2.4:** Spatial Railway & Airport Flight-Path Noise Buffer (`criterion_spatial_noise_buffer`).
- [x] **M2.5:** Regex-based Junk Fee & Utility Inclusion Parser (`criterion_advanced_junk_fee_parser`).

## Milestone 3: Zero-Quota Visual & LLM Grading
- [x] **M3.1 (Active - Strategy 1): Local OpenAI CLIP Vision Grader (`score_photos_with_local_clip`)**
  - *Implementation:* Runs `openai/clip-vit-base-patch32` locally on Google Colab CPU/GPU with zero API keys and zero quota limits. Evaluates listing photos for modern stainless/quartz open-concept kitchens and hard flooring vs. dated coil stoves and carpet, then blends the visual score into `modern_interior_score` and `composite_score`.
- [ ] **Backlog Issue #1 (Saved for Later - Strategy 3): Keyless `google.colab.ai` & Single-Prompt Batch LLM Auditor**
  - *Goal:* Concatenate the descriptions of the top 15–20 spatial finalists into a single batched prompt using `google.colab.ai` (or a single free-tier Gemini Flash call) to extract complex lease nuances, exact W/D machine inclusion vs. hookups-only, and itemized effective rent in 1 request.
- [ ] **Backlog Issue #2 (Optional): Distinguish "W/D Machines Included" vs "W/D Hookups Only" in Gatekeeper**
  - *Goal:* Refine `criterion_required_appliances_gatekeeper` to flag or filter units that only provide washer/dryer connections rather than physical machines.
