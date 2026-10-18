---
name: New Scoring Criterion or Filter
about: Propose a new spatial, environmental, or listing criterion for the rental scorer
title: '[CRITERION] '
labels: enhancement, criterion
---

### Criterion Name & Goal
What lifestyle, environmental, or property factor should this measure?

### Data Source
- [ ] OpenStreetMap tag (`ox.features_from_place`)
- [ ] Listing text / metadata (`HomeHarvest` columns)
- [ ] Gemini Vision API photo inspection
- [ ] External API / Shapefile (e.g., EPA AirNow, TxDOT, Fiber coverage)

### Scoring Logic / Hard Cutoff
- **Hard Requirement Gate?** (Yes / No)
- **Score Formula (0 to 100):** (e.g., 100 pts if < 5 mins, 0 pts if > 20 mins)
- **Suggested Weight in `config.yaml`:** (e.g., `0.08`)
