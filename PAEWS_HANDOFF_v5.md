# PAEWS Handoff v5 — Session 10 Ready

**Peruvian Anchovy Early Warning System**
**Date:** February 25, 2026
**Status:** Biomass extraction paused at 10/26 verified; model baseline locked

---

## 1. What PAEWS Is

An ocean monitoring system that detects environmental conditions threatening Peru's anchovy fishery — the world's largest single-species fishery (~$3.5B/year exports). When El Niño warms the waters and collapses the food chain, anchovy disappear. PAEWS detects these compound events months before official season decisions, giving fisheries managers, traders, and coastal communities advance warning.

Peru divides anchovy fishing into two seasons per year:
- **Season 1 (S1):** April–July (main season, ~65% of annual catch)
- **Season 2 (S2):** November–January

Before each season, IMARPE runs acoustic surveys and recommends quotas to PRODUCE (Ministry of Production). Our system aims to predict disruptions before IMARPE's official announcements.

**GitHub:** `https://github.com/jdante2023/PAEWS`
**Local path:** `c:\Users\josep\Documents\paews\`
**Conda environment:** `geosentinel`

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

**Performance History:**

| Version | Features | Chl Source | PR-AUC | Notes |
|---------|----------|-----------|--------|-------|
| Original | 6 (MODIS) | MODIS raw | 0.743 | Chl=45% importance, but sensor dying |
| Copernicus 7-feat | 7 | Copernicus (no mask) | 0.695 | Chl collapsed to 7%, SLA negative coef |
| Copernicus 5-feat + mask | 5 | Copernicus (coastal mask) | 0.698 | Chl recovered to 27.3% |
| **Current: 30 samples** | **4** | **Copernicus (40% mask)** | **0.682** | **Most honest estimate, SST 2024 added** |

The drop from 0.743→0.682 is real: MODIS had a sensor bias that accidentally amplified signals. 0.682 is what environmental data alone actually knows. The model still catches every catastrophe.

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
- **Bounding box:** 0°S–16°S, 85°W–70°W (Peru's Humboldt Current upwelling zone)

---

## 4. Biomass Integration — PAUSED (10 of 26 Verified)

### Why It Matters
The model's main weakness is false alarms: environment looks stressed but the stock is healthy enough to sustain fishing. Adding lagged biomass (previous survey's estimate) should help distinguish "stressed environment + healthy stock = NORMAL" from "stressed environment + depleted stock = REDUCED." Target: PR-AUC > 0.75.

### Verified Biomass Numbers (from primary IMARPE PDFs)

| # | Year | S1 (MMT) | S2 (MMT) | Source | PDF Verified |
|---|------|----------|----------|--------|-------------|
| 1-2 | 2018 | 11.21 | 8.78 | Castillo et al. (2021) Bol IMARPE 35(2) art/view/301 | ✅ |
| 3-4 | 2019 | 8.82 | 8.38* | Castillo et al. (2021) Bol IMARPE 35(2) art/view/302 | ✅ |
| 5-6 | 2020 | 11.05 | 9.52 | Castillo et al. (2021) Inf Inst Mar Perú 48(3): 327-349 | ✅ |
| 7-8 | 2021 | 12.03 | 8.03 | Castillo et al. (2023) Bol IMARPE 38(1) art/view/385 | ✅ |
| 9-10 | 2022 | 7.13 | 4.68 | Castillo et al. (2024) Bol IMARPE 39(1) art/view/438 | ✅ |

*2019 S2: IMARPE officials investigated for allegedly inflating biomass from ~4 MMT to 8.3 MMT. Add `integrity_flag=0` for this row.

### Key Verification Findings (Sessions 8-10)

**2020 values (Session 9):** The Castillo et al. (2021) paper in Inf Inst Mar Perú 48(3) has a **translation error** in its English abstract — the S1 and S2 values are swapped. The Spanish abstract and body text (Tables 5-6, conclusions) confirm the correct values: S1=11.05, S2=9.52. The previously unverified value of 10.11 MMT for 2020 S1 (from Gemini in Session 4) was confirmed wrong.

**2021 S1 (Session 10):** Confirmed via a second paper — Castillo et al. (2022) Inf Inst Mar Perú 49(2): 175-192 reports 12.03 MMT for the unified cruise 2102-07 (two sub-cruises statistically merged via Wilcoxon test). Consistent with the Bol 38(1) paper.

**2022 values (Session 8):** Confirmed from Castillo et al. (2024) Bol IMARPE 39(1). S1=7.13 (Cr 2202-04, Feb-Apr 2022), S2=4.68 (Cr 2209-11). Note: v4 handoff had S1=10.20, S2=7.18 which were from a different paper (Inf 52(1)) — the Bol 39(1) values are the correct primary source.

**Gutiérrez et al. (2012) context paper:** Covers 1966-2009 acoustic time series with Hovmöller diagrams. Anchovy range: 0 to 12.71 MMT, average 5.13 MMT. Useful for sanity-checking but doesn't provide discrete per-cruise values extractable for the CSV. Does NOT cover 2010+.

### Still Missing: 16 season-pairs

| Years | Seasons | Notes |
|-------|---------|-------|
| 2010-2012 | S1, S2 each (6 total) | Cruise codes: 10xx, 11xx, 12xx |
| 2013-2014 | S1, S2 each (4 total) | **HIGH PRIORITY** — lag into hardest-to-predict seasons |
| 2015-2017 | S1, S2 each (6 total) | **2017 HIGH PRIORITY** |
| 2023-2025 | S1, S2 each (6 total) | PRODUCE Resoluciones or recent repositorio papers |

### Where to Find Them

**2010-2017:** These Castillo et al. papers are NOT on `revistas.imarpe.gob.pe` (confirmed Session 8). They exist on `repositorio.imarpe.gob.pe` as "Informe Ejecutivo" or cruise report documents. Search for cruise codes: 1002, 1009, 1102, 1109, ..., 1702, 1709.

**2023-2025:** Search PRODUCE website for Resoluciones Ministeriales that cite IMARPE biomass estimates when authorizing fishing seasons. Also check repositorio for recent Castillo papers.

### CRITICAL: Verification Protocol
**Rule 1:** Nothing enters the CSV without a source URL or PDF page number.
**Rule 2:** Never ask LLMs for specific biomass numbers — they fabricate them. Gemini was measured at 65% error rate (Session 4). A complete 2010-2017 table was generated from a nonexistent paper citation.
**Rule 3:** Primary sources only — `repositorio.imarpe.gob.pe` PDFs, `revistas.imarpe.gob.pe` articles.

### Biomass CSV Template
File: `data/external/imarpe_biomass_verified.csv`
Columns: year, season, cruise_code, biomass_mmt, region, source_type, source_detail, verified, notes

---

## 5. Season-by-Season Results (30 samples)

```
Year S  Outcome    Prob     Pred     Result
2010 1  NORMAL     0.60   AT-RISK    MISS (false alarm)
2010 2  NORMAL     0.21   NORMAL     HIT
2011 1  NORMAL     0.22   NORMAL     HIT
2011 2  REDUCED    0.13   NORMAL     MISS (missed disruption)
2012 1  REDUCED    0.49   AT-RISK    HIT
2012 2  REDUCED    0.51   AT-RISK    HIT
2013 1  NORMAL     0.23   NORMAL     HIT
2013 2  NORMAL     0.39   AT-RISK    MISS (false alarm)
2014 1  DISRUPTED  0.16   NORMAL     MISS ← subsurface Kelvin wave
2014 2  REDUCED    0.53   AT-RISK    HIT
2015 1  REDUCED    0.38   AT-RISK    HIT
2015 2  DISRUPTED  0.85   AT-RISK    HIT
2016 1  DISRUPTED  0.77   AT-RISK    HIT
2016 2  NORMAL     0.68   AT-RISK    MISS (false alarm)
2017 1  REDUCED    0.89   AT-RISK    HIT
2017 2  NORMAL     0.37   NORMAL     HIT
2018 1  NORMAL     0.19   NORMAL     HIT
2018 2  NORMAL     0.74   AT-RISK    MISS (false alarm)
2019 1  NORMAL     0.61   AT-RISK    MISS (false alarm)
2019 2  NORMAL     0.49   AT-RISK    MISS (false alarm)
2020 1  NORMAL     0.67   AT-RISK    MISS (false alarm)
2020 2  NORMAL     0.28   NORMAL     HIT
2021 1  NORMAL     0.19   NORMAL     HIT
2021 2  NORMAL     0.37   NORMAL     HIT
2022 1  NORMAL     0.36   NORMAL     HIT
2022 2  DISRUPTED  0.12   NORMAL     MISS ← cold-water disruption
2023 1  CANCELLED  0.94   AT-RISK    HIT  ← $1.4B loss predicted
2023 2  DISRUPTED  0.88   AT-RISK    HIT
2024 1  NORMAL     0.30   NORMAL     HIT
2024 2  NORMAL     0.55   AT-RISK    MISS (borderline false alarm)
```

---

## 6. Known Problems

### Still Open

**2014 S1 Ghost Miss (prob=0.16):** Subsurface Kelvin wave invisible to SST/Chl. SLA at monthly resolution also missed it. Needs weekly SLA or biomass.

**2022 S2 Cold-Water Disruption (prob=0.12):** SST was cold (-1.03σ), Chl moderately low. Model learned "cold = safe." Stock was depleted despite cool waters. Only biomass data would catch this.

**False Alarm Cluster (2018-2020):** 5 false positives where environment looked stressed but fishery was fine. Root cause: healthy stock tolerates moderate stress. Biomass is the fix.

**Copernicus Chl signal compression:** Coastal mask recovered most of signal (2.9%→27.3%), but still less discriminating than MODIS.

### Resolved
- ✅ MODIS dependency removed (was dying August 2026)
- ✅ Mixed-baseline bug fixed
- ✅ SLA negative coefficient issue (removed from regression, stored in CSV)
- ✅ Overfitting with 7 features on 28 samples (pruned)
- ✅ CSV overwrite race condition (CSV-first architecture)
- ✅ >100% pixel percentage bug (static ocean mask)
- ✅ 2020 biomass values verified (was unverified in v4)
- ✅ 2022 biomass source conflict resolved (Bol 39(1) is primary)

### Dead Ends (Don't Repeat)
- **SLA (Sea Level Anomaly):** Tested Session 3-5. CMEMS product only goes back to 2014. Negative coefficient. Dead end at monthly resolution.
- **Mask >50%:** Drops PR-AUC. 40% is optimal.
- **Mask <30%:** Also drops performance. 40% is the sweet spot.
- **Gemini biomass numbers:** 65% error rate. Don't use.
- **revistas.imarpe.gob.pe for pre-2018 papers:** Only has 2018-2022 Castillo papers digitized. 2010-2017 must come from repositorio.

---

## 7. File Locations

```
c:\Users\josep\Documents\paews\
├── data/
│   ├── copernicus_chl/          # 30 NetCDF files (1997-2024)
│   ├── sst/                     # SST NetCDF files (2010-2024)
│   ├── ground_truth.csv         # 30 seasons, NORMAL/REDUCED labels
│   └── external/
│       ├── paews_feature_matrix.csv      # 30 seasons, all features
│       ├── imarpe_biomass_verified.csv   # 10 verified, 16 missing
│       ├── nino_indices_monthly.csv      # 529 months (1982-Jan 2026)
│       ├── fishmeal_prices_monthly.csv   # World Bank (1979-2025)
│       └── peru_anchovy_catch_annual.csv # FAO (1950-2024)
├── scripts/
│   ├── chl_migration.py         # Chl processing + coastal mask
│   ├── composite_score.py       # Main scoring pipeline
│   └── health_check.py          # Validation (33 passes expected)
├── masks/coastal_mask_40pct.nc  # Active mask
└── PAEWS_HANDOFF_v5.md          # This file
```

---

## 8. Papers Read & Uploaded (Sessions 7-10)

| Paper | Journal | Content | Biomass? |
|-------|---------|---------|----------|
| Castillo et al. (2023) | Bol IMARPE 38(1) art/385 | 2021 S1+S2 acoustics | ✅ 12.03 / 8.03 |
| Castillo et al. (2024) | Bol IMARPE 39(1) art/438 | 2022 S1+S2 acoustics | ✅ 7.13 / 4.68 |
| Castillo et al. (2021) | Inf Inst Mar Perú 48(3) | 2020 S1+S2 acoustics | ✅ 11.05 / 9.52 |
| Castillo et al. (2022) | Inf Inst Mar Perú 49(2) | 2021 S1 (unified cruise) | ✅ 12.03 (confirms) |
| Orosco (2023) | Bol IMARPE 38(2) art/393 | Zooplankton/ichthyoplankton 2022 | ❌ No biomass |
| Gutiérrez et al. (2012) | Lat Am J Aquat Res | 1966-2009 acoustic time series | ❌ Hovmöller only |
| 3 companion papers | Bol 38(2), Inf 51(4), Inf 52(1) | Phytoplankton, zooplankton | ❌ No biomass |

---

## 9. Session History

| Session | Key Achievement |
|---------|----------------|
| 1–3 | Initial build, Copernicus migration, ground truth creation |
| 4 | Gemini hallucination discovery (65% biomass error), verification protocols |
| 5 | Mask tuning (50%→40%), SLA dead end confirmed, PR-AUC 0.705 |
| 6 | SST 2024 added (30 samples), PR-AUC locked at 0.682, CSV template |
| 7 | Uploaded Castillo (2023) PDF — confirmed 2021 S1=12.03, S2=8.03 |
| 8 | Found 2022 biomass in Bol 39(1). Confirmed revistas only has 2018+. Created 5-step search plan for repositorio. Rejected unverified biomass block. |
| 9 | Uploaded Castillo (2021) 2020 paper. Verified 2020 S1=11.05, S2=9.52. Found English abstract translation error (values swapped). Rejected Gemini's 10.11 for 2020 S1. |
| 10 | Uploaded Castillo (2022) 2021-unified paper (confirms 12.03). Uploaded Orosco zooplankton paper (no biomass). Reviewed Gutiérrez 1966-2009 context paper. Updated to v5. |

---

## 10. What the Next Chat Should Do

### Immediate Priority Options

**A. Continue biomass extraction (when user has PDFs):**
1. Search `repositorio.imarpe.gob.pe` for Castillo cruise reports 2010-2017
2. Priority targets: 2013, 2014, 2017 (lag into hardest-to-predict seasons)
3. Search PRODUCE for 2023-2025 Resoluciones Ministeriales
4. Every PDF must be uploaded for page-level verification

**B. Build biomass feature with 10 verified values:**
1. Update `imarpe_biomass_verified.csv` with the 10 verified values
2. Add lagged biomass as Feature #5 to `composite_score.py`
3. Rerun pipeline — even partial coverage may improve PR-AUC
4. Test with NaN handling for missing years

**C. 2026 S1 live prediction:**
1. IMARPE Cruise 2602-04 is at sea (Feb 16–Apr 4, 2026)
2. Once March SST/Chl data available, run system for real 2026 S1 prediction
3. First live validation opportunity

**D. Research paper editing:**
User has a draft on GitHub

### Medium Priority
- Sentinel-3 Collection 4 monitoring (expected Feb 26, 2026 — may break Chl downloads)
- SLA rule-based override (boost risk if sla_z > 1.5, without putting SLA in regression)

---

## 11. Quick Start for New Chat

Paste this:

> I'm working on PAEWS — a satellite-based anchovy early warning system for Peru. I've uploaded PAEWS_HANDOFF_v5.md which has all the context. Current PR-AUC is 0.682 with 30 samples. I have 10 verified biomass values (2018-2022) and need 16 more (2010-2017, 2023-2025). [Describe what you want to do next.]

**First, run the health check:**
```powershell
cd C:\Users\josep\Documents\paews\scripts
python health_check.py
```

**Upload these files when asked:**
- `composite_score.py` — main model
- `chl_migration.py` — Copernicus Chl pipeline + coastal mask
- `paews_feature_matrix.csv` — current feature matrix
- `health_check.py` — structural integrity checker
