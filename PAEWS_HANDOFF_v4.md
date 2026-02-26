# PAEWS Handoff v4 — Session 7 Ready

**Peruvian Anchovy Early Warning System**
**Date:** February 26, 2026
**Status:** Biomass extraction in progress, model baseline locked

---

## 1. What PAEWS Is

A satellite-based early warning system that predicts whether Peru's anchovy fishing season will be NORMAL or REDUCED, using only freely available remote sensing data (SST, Chlorophyll-a). Target: issue warnings 1–2 months before IMARPE's official cruise results.

**GitHub:** `https://github.com/jdante2023/PAEWS`
**Local path:** `c:\Users\josep\Documents\paews\`

---

## 2. Current Model Performance (LOCKED BASELINE)

| Metric | Value |
|--------|-------|
| PR-AUC | **0.682** |
| ROC-AUC | 0.639 |
| Recall | 67% (8/12 at-risk caught) |
| False Positives | 6 |
| Samples | **30** (2010 S1 – 2024 S2) |
| Mask | 40% (22,939 coastal pixels) |
| Health Check | 33 passes, 0 failures |
| Commit | ae21acd (mask tuning) + SST 2024 added |

**Feature weights:** Chl 34.0%, SST 24.9%, SSTa 20.7%, Chla 20.4%

---

## 3. Data Sources & Pipeline

### Satellite Data (all downloaded)
- **Chlorophyll-a:** Copernicus GlobColour L4 (300m reprocessed), 30 files (1997–2024), stored in `data/copernicus_chl/`
- **SST:** NOAA OISST v2.1 via ERDDAP, `data/sst/sst_YYYY.nc`, 2010–2024

### Pipeline Scripts (run in order)
```
python chl_migration.py      # Process 30 Copernicus Chl files
python composite_score.py    # Generate risk scores for all 30 samples
python health_check.py       # Validate: expect 33 passes, 0 failures
```

### Ground Truth
File: `data/ground_truth.csv`
- 30 rows (2010 S1 through 2024 S2)
- Columns: year, season, outcome (NORMAL/REDUCED), notes
- 12 REDUCED seasons, 18 NORMAL seasons

### Key Technical Details
- **Coastal mask:** 40% threshold — keeps 22,939 pixels within ~100km of coast
- **Season windows:** S1 = Jan–Jun composite, S2 = Jul–Dec composite
- **Scoring:** Logistic regression on 4 features (Chl mean, SST mean, Chl anomaly, SST anomaly)
- **Anomaly baseline:** 2010–2024 seasonal climatology

---

## 4. Biomass Integration — IN PROGRESS (Highest Priority)

### Why It Matters
The model's main weakness is false alarms: environment looks stressed but the stock is healthy enough to sustain fishing. Adding lagged biomass (previous survey's estimate) should help the model distinguish "stressed environment + healthy stock = NORMAL" from "stressed environment + depleted stock = REDUCED." Target: PR-AUC > 0.75.

### Verified Biomass Numbers (from primary IMARPE PDFs)

| Year | S1 (MMT) | S2 (MMT) | Source | Verified |
|------|----------|----------|--------|----------|
| 2018 | 11.21 | 8.78 | Castillo et al. (2021) Bol IMARPE 35(2) | ✅ TRUE |
| 2019 | 8.82 | 8.38* | Castillo et al. (2021) Bol IMARPE 35(2) | ✅ TRUE |
| 2021 | 12.03 | 8.03 | Castillo et al. (2023) Bol IMARPE 38(1) — **PDF uploaded & read** | ✅ TRUE |
| 2022 | 10.20 | 7.18 | Castillo et al. (2025) Inf Inst Mar Peru 52(1) | ✅ TRUE |

*2019 S2: IMARPE officials investigated for allegedly inflating biomass from ~4 MMT to 8.3 MMT. Add `integrity_flag=0` for this row.

### Partially Verified (secondary sources)

| Year | Season | Biomass | Source | Notes |
|------|--------|---------|--------|-------|
| 2015 | S2 | 3.38 or 6.07 | Oceana report | Contested methodology |
| 2016 | S1 | 4.42 or 7.28 | Oceana 2016 | Two cruises, post-El Niño |
| 2017 | S1 | 7.78 | Oceana blog | Norte-Centro only |

### Still Missing: ~18 seasons
2010–2014 (all), 2015 S1, 2016 S2, 2017 S2, 2020 (both), 2023 (both), 2024 (both), 2025 S1

### Papers to Find
Each year should have a Castillo et al. paper in Boletín IMARPE or Informe IMARPE:
- Search `revistas.imarpe.gob.pe` for each year
- Or find a single paper with a historical comparison table (the 2025 paper covering 2022 was expected to have one but we haven't confirmed)
- The PDF uploaded this session (Bol 38(1), 2021 paper) did NOT have a historical table — it only covers 2021

### CRITICAL: Verification Protocol
**Rule 1:** Nothing enters the CSV without a source URL or PDF page number.
**Rule 2:** Never ask LLMs for specific biomass numbers — they fabricate them. An earlier attempt produced a complete 2010–2017 table citing "Castillo et al. (2022)" which does not exist.
**Rule 3:** Primary sources only — `repositorio.imarpe.gob.pe` PDFs, `revistas.imarpe.gob.pe` articles.

### Biomass CSV Template
File created: `data/external/imarpe_biomass_verified.csv`
Columns: year, season, cruise_code, biomass_mmt, region, source_type, source_detail, verified, notes

---

## 5. What the Next Chat Should Do

### Immediate (biomass extraction)
1. **Find remaining biomass numbers.** Best approach: search IMARPE repositorio for Castillo papers covering 2010–2017, 2020, 2023–2025. Each year has a dedicated cruise report paper.
2. **Verify every number** against the actual PDF text — do not trust any number from an LLM without a page reference.
3. **Update `imarpe_biomass_verified.csv`** with verified numbers only.
4. **Add biomass as Feature #6** to `composite_score.py` using lagged value (previous survey's biomass predicts current season outcome).
5. **Rerun pipeline** and check if PR-AUC improves toward 0.75.

### Medium Priority
6. **2026 S1 live prediction script** — uses current satellite data to generate a risk probability before the April cruise. Time-sensitive.
7. **Sentinel-3 Collection 4 monitoring** — Copernicus reprocessing was expected Feb 26, 2026. Check if new data breaks the pipeline.

### Lower Priority
8. **Updated handoff** after biomass integration (v5 with final PR-AUC)
9. **Research paper editing** — user has a draft on GitHub

---

## 6. Known Issues & Lessons

### Performance Drop Is Healthy
PR-AUC went from 0.705 (28 samples) → 0.682 (30 samples) when SST 2024 was added. This is not regression — it's a more honest estimate. The two new 2024 seasons are ambiguous (S1 normal prob=0.30 ✓, S2 borderline FP at 0.55), and the additional samples shifted the decision boundary slightly.

### Hallucination Risk Is Real
- Gemini was measured at 65% biomass error rate in Session 4
- A complete 2010–2017 biomass table was generated from a nonexistent paper citation
- Even official IMARPE numbers can be unreliable (2019 S2 investigation)
- Primary scientific publications (Boletín IMARPE, peer-reviewed papers) are more trustworthy than quota resolutions

### Dead Ends (Don't Repeat)
- **SLA (Sea Level Anomaly):** Tested in Session 5. CMEMS SEALEVEL product only goes back to 2014 at appropriate resolution. Not worth the coverage gap. Confirmed dead end.
- **Mask >50%:** Drops PR-AUC. 40% is optimal.
- **Mask <30%:** Also drops performance. 40% is the sweet spot.

---

## 7. File Locations

```
c:\Users\josep\Documents\paews\
├── data/
│   ├── copernicus_chl/          # 30 NetCDF files (1997-2024)
│   ├── sst/                     # SST NetCDF files (2010-2024)
│   ├── ground_truth.csv         # 30 seasons, NORMAL/REDUCED labels
│   └── external/
│       └── imarpe_biomass_verified.csv  # Biomass template (8 verified, 18 missing)
├── chl_migration.py             # Chl processing
├── composite_score.py           # Main scoring pipeline
├── health_check.py              # Validation (33 passes expected)
├── masks/coastal_mask_40pct.nc  # Active mask
└── PAEWS_HANDOFF_v4.md          # This file
```

---

## 8. Session History

| Session | Key Achievement |
|---------|----------------|
| 1–3 | Initial build, Copernicus migration, ground truth creation |
| 4 | Gemini hallucination discovery (65% biomass error rate), verification protocols established |
| 5 | Mask tuning (50%→40%), SLA dead end confirmed, PR-AUC 0.705 with 28 samples |
| 6 | SST 2024 added (30 samples), PR-AUC baseline locked at 0.682, biomass verification crisis, CSV template created |
| 7 (current) | Uploaded & read Castillo et al. (2023) PDF — confirmed 2021 S1=12.03, S2=8.03. No historical table found in this paper. Handoff v4 created. |

---

## 9. Quick Start for New Chat

Paste this:

> I'm working on PAEWS — a satellite-based anchovy early warning system for Peru. I've uploaded PAEWS_HANDOFF_v4.md which has all the context. Current PR-AUC is 0.682 with 30 samples. My highest priority is finding and verifying IMARPE anchovy biomass estimates to add as a model feature. I have 8 verified numbers (2018-2019, 2021-2022) and need ~18 more (2010-2017, 2020, 2023-2025). Can you help me search for the remaining Castillo et al. papers in the IMARPE repositorio?
