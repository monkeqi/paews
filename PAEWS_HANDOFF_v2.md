# PAEWS — Peru Anchovy Early Warning System
## Complete Project Handoff v2 (Feb 25, 2026)

---

## 1. WHAT IT IS

An ocean monitoring system that detects environmental conditions threatening Peru's anchovy fishery — the world's largest single-species fishery (~$3.5B/year exports). When El Niño warms the waters and collapses the food chain, anchovy disappear. PAEWS detects these compound events months before official season decisions, giving fisheries managers, traders, and coastal communities advance warning.

Peru divides anchovy fishing into two seasons per year:
- **Season 1 (S1):** April–July (main season, ~65% of annual catch)
- **Season 2 (S2):** November–January

Before each season, IMARPE (Peru's Marine Institute) runs acoustic surveys and recommends quotas to PRODUCE (Ministry of Production). Our system aims to predict disruptions before IMARPE's official announcements.

---

## 2. PROJECT SETUP

**Local machine:** Windows, PowerShell
**Working directory:** `C:\Users\josep\Documents\paews`
**Conda environment:** `geosentinel`
**Run command pattern:**
```powershell
cd C:\Users\josep\Documents\paews\scripts
python [script].py
```

**GitHub:** `github.com/monkeqi/paews`

**Key Python packages:** numpy, pandas, xarray, matplotlib, scikit-learn, netCDF4, requests, copernicusmarine, h5py

**Copernicus Marine account:** Registered at data.marine.copernicus.eu (free). Login stored locally via `copernicusmarine login`.

---

## 3. DIRECTORY STRUCTURE

```
C:\Users\josep\Documents\paews\
├── scripts/
│   ├── anomaly_detector.py        # SST anomaly detection + seasonal alerts
│   ├── chl_anomaly_detector.py    # Chlorophyll anomaly + compound events
│   ├── compute_climatology.py     # SST climatology builder
│   ├── chl_climatology.py         # Chlorophyll climatology builder (log-space)
│   ├── composite_score.py         # ★ MAIN: Logistic regression + validation
│   ├── chl_migration.py           # ★ NEW: Copernicus Chl baseline migration
│   ├── sla_pipeline.py            # ★ NEW: Sea Level Anomaly feature extraction
│   ├── gap_filler.py              # Legacy Copernicus Chl gap-fill (superseded by chl_migration.py)
│   ├── external_data_puller.py    # Niño indices, fishmeal prices downloader
│   └── data_pipeline.py           # SST download pipeline
├── data/
│   ├── baseline_v2/               # SST yearly .nc files (2003-2024, OISST 0.25°)
│   ├── baseline_v2_chl/           # Chl yearly .nc files (2004-2022, MODIS 4km) — LEGACY
│   ├── processed/
│   │   ├── sst_climatology_v2.nc           # Monthly mean/std (2003-2022)
│   │   ├── chl_climatology_v2.nc           # Monthly log-mean/std MODIS (2004-2022) — LEGACY
│   │   ├── chl_climatology_copernicus.nc   # ★ NEW: Copernicus log-mean/std (2003-2022)
│   │   └── sla_climatology.nc              # ★ NEW: SLA monthly mean/std (2010-2022)
│   └── external/
│       ├── paews_feature_matrix.csv        # ★ 30 seasons, all features
│       ├── imarpe_ground_truth.csv         # Season outcomes 2010-2024
│       ├── nino_indices_monthly.csv        # 529 months (1982-Jan 2026)
│       ├── fishmeal_prices_monthly.csv     # World Bank (1979-2025)
│       ├── peru_anchovy_catch_annual.csv   # FAO (1950-2024)
│       ├── chl_copernicus_full.nc          # ★ NEW: Full Copernicus Chl 2003-2025 (145 MB)
│       ├── chl_copernicus_2022_2023.nc     # Legacy gap-fill data (12.7 MB)
│       └── sla_monthly_2010_2024.nc        # ★ NEW: Monthly SLA (13.2 MB)
└── outputs/
    └── composite_score_validation.png      # 4-panel dashboard
```

---

## 4. BOUNDING BOX & BASELINE

**Region:** 0°S–16°S, 85°W–70°W (Peru's Humboldt Current upwelling zone)
**SST source:** NOAA OISST v2.1 (0.25° daily) via ERDDAP `ncdcOisst21Agg_LonPM180`
**Chl source (NEW):** Copernicus GlobColour L4 multi-sensor monthly (`cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M`) — replaces MODIS Aqua
**SLA source (NEW):** Copernicus multi-year monthly SLA (`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m`)
**Baseline period:** 2003-2022 (SST & Chl), 2010-2022 (SLA)
**Climatology:** Monthly mean and std (SST, SLA), monthly log10-mean and log10-std (Chl)
**Static ocean pixels:** SST=1,629, Chl=57,336 (Copernicus grid), SLA grid covers full region at 0.125°

---

## 5. CURRENT MODEL STATE (Latest Run — Feb 25, 2026)

### 5.1 What Changed This Session
1. **SLA integrated** — downloaded monthly SLA 2010-2024 from Copernicus, computed climatology, extracted Z-scores for all 30 seasons
2. **Full Copernicus Chl migration** — downloaded 2003-2025 history (145 MB, 276 months), replaced MODIS baseline entirely
3. **Architecture rewrite** — `composite_score.py` now uses CSV-first approach: reads ledger before building features, prioritizes Copernicus Chl, pulls SLA from ledger, only computes SST from raw files
4. **Feature pruning (PENDING)** — discovered 7 features on 28 samples = overfitting. SLA coefficient went negative (opposite of theory). Updated script drops SLA and thermal_shock from regression, keeping 5 core features. **This version has NOT been run yet.**

### 5.2 Features (currently 7 in CSV, regression uses 5)

Active in regression:
| # | Feature | Description | Source |
|---|---------|-------------|--------|
| 1 | `sst_z` | SST Z-score in decision month | OISST |
| 2 | `chl_z` | Chl Z-score (inverted: negative = bad) | Copernicus GlobColour |
| 3 | `nino12_t1` | Niño 1+2 anomaly, 1-month lag | NOAA CPC |
| 4 | `is_summer` | Binary: S1=1, S2=0 | Calendar |
| 5 | `bio_thresh_pct` | % pixels where absolute SST > 23°C | OISST |

Stored in CSV but NOT in regression (noise at n=28):
| # | Feature | Description | Why excluded |
|---|---------|-------------|--------------|
| 6 | `thermal_shock` | Binary: 1 if bio_thresh > 25% | Collinear with bio_thresh |
| 7 | `sla_z` | SLA Z-score in decision month | Coefficient went negative (opposite theory); only 2.7% importance when included; monthly resolution too coarse for transient Kelvin waves |

### 5.3 Last Run Performance (7 features, Copernicus Chl — the BAD run)
```
ROC-AUC:  0.656
PR-AUC:   0.695  ← REGRESSION from 0.743
Best threshold: 0.49 (F1=0.64)
Recall: 67% (8/12)
```
This is the run with all 7 features on the Copernicus baseline. Performance degraded because:
- Chl importance collapsed from 45% → 7% (Copernicus Z-scores are compressed)
- SLA got -0.554 coefficient (wrong sign — fitting noise)
- 2022 S2 completely missed (prob=0.06)

### 5.4 PENDING: Run with 5 features
The updated `composite_score.py` (already delivered, not yet run) drops SLA and thermal_shock from the regression. Expected to recover performance by reducing overfitting. **Run this first in the new chat.**

### 5.5 Data-Driven Weights (7-feature run, for reference)
```
SST:           31.6%
Chlorophyll:    7.0%  ← collapsed from 45%
Niño 1+2:     14.3%
Season flag:    4.7%
Bio >23°C:    18.6%
Thermal Shock:  4.7%
SLA:           19.1%  (negative coefficient — wrong sign)
```

### 5.6 Season-by-Season (7-feature Copernicus run)
```
Year S  Outcome    Prob     Pred     Result
2010 1  NORMAL     0.64   AT-RISK    MISS (false alarm)
2010 2  NORMAL     0.34   NORMAL     HIT
2011 1  NORMAL     0.38   NORMAL     HIT
2011 2  REDUCED    0.20   NORMAL     MISS
2012 1  REDUCED    0.49   AT-RISK    HIT
2012 2  REDUCED    0.60   AT-RISK    HIT
2013 1  NORMAL     0.31   NORMAL     HIT
2013 2  NORMAL     0.43   NORMAL     HIT
2014 1  DISRUPTED  0.27   NORMAL     MISS ← still missed
2014 2  REDUCED    0.49   AT-RISK    HIT
2015 1  REDUCED    0.36   NORMAL     MISS
2015 2  DISRUPTED  0.79   AT-RISK    HIT
2016 1  DISRUPTED  0.70   AT-RISK    HIT
2016 2  NORMAL     0.74   AT-RISK    MISS (false alarm)
2017 1  REDUCED    0.82   AT-RISK    HIT
2017 2  NORMAL     0.32   NORMAL     HIT
2018 1  NORMAL     0.22   NORMAL     HIT
2018 2  NORMAL     0.66   AT-RISK    MISS (false alarm)
2019 1  NORMAL     0.57   AT-RISK    MISS (false alarm)
2019 2  NORMAL     0.35   NORMAL     HIT
2020 1  NORMAL     0.60   AT-RISK    MISS (false alarm)
2020 2  NORMAL     0.33   NORMAL     HIT
2021 1  NORMAL     0.19   NORMAL     HIT
2021 2  NORMAL     0.47   NORMAL     HIT
2022 1  NORMAL     0.18   NORMAL     HIT
2022 2  DISRUPTED  0.06   NORMAL     MISS ← CRITICAL REGRESSION
2023 1  CANCELLED  0.87   AT-RISK    HIT
2023 2  DISRUPTED  0.86   AT-RISK    HIT
```

### 5.7 Copernicus vs MODIS Sensor Bias
The Copernicus multi-sensor product reads systematically lower than MODIS in log10 space:
```
Month   Cop_mean  MODIS_mean     Diff
    1    -0.3188     -0.2707  -0.0481
    2    -0.3298     -0.2838  -0.0460
    3    -0.2869     -0.2554  -0.0315
    4    -0.2453     -0.2125  -0.0328
    5    -0.3032     -0.2389  -0.0643 ← winter bias grows
    6    -0.4279     -0.3076  -0.1204
    7    -0.4784     -0.3452  -0.1332 ← max bias
    8    -0.4576     -0.3327  -0.1248
    9    -0.3939     -0.2685  -0.1254
   10    -0.3721     -0.2736  -0.0984
   11    -0.3546     -0.2540  -0.1006
   12    -0.3178     -0.2504  -0.0675
```
Root cause: multi-sensor product fills cloud gaps with more offshore (lower productivity) pixels, shifting the climatological mean downward. The Z-scores are internally consistent on the Copernicus baseline, but the variance is lower, compressing the signal.

---

## 6. KNOWN PROBLEMS & THEIR FIXES

### 6.1 The 2014 Ghost Miss (Unresolved)
**Problem:** 2014 S1 was DISRUPTED but model gives prob=0.27 (7-feature) or 0.11 (original). SST was cool (-0.64σ), Chl looked fine.
**What we tried:** SLA integration. March 2014 SLA Z-score was only -0.67 — not the expected +1.0+ Kelvin wave signal. Monthly SLA resolution likely averaged out the transient event.
**Remaining fix options:** Weekly SLA data, or accept this as an irreducible miss without biomass data.

### 6.2 False Alarm Cluster (2018-2020) — Unresolved
**Problem:** Multiple false positives where environment looked stressed but fishery was fine.
**Fix:** Acoustic biomass data from IMARPE cruise reports. Healthy stock tolerates moderate environmental stress. Without biomass, model can't distinguish "stressed environment + weak stock" from "stressed environment + strong stock."

### 6.3 Copernicus Chl Signal Compression — NEW ISSUE
**Problem:** Copernicus Z-scores are compressed vs MODIS. 2022 S2 Chl signal weakened from -0.77 (MODIS) → -0.25 (Copernicus). Chl importance dropped from 45% → 7%.
**Status:** The baseline is internally consistent (all 30 seasons on same Copernicus climatology), but the weaker variance means Chl has less discriminative power. The 5-feature run (pending) may partially recover this.
**Possible future fixes:** Restrict spatial mask to nearshore upwelling pixels only (where variance is higher); use 8-day or weekly Copernicus product instead of monthly.

### 6.4 Thermal Shock Collinearity
**Problem:** `thermal_shock` binary only gets 1.2-4.7% importance because it's perfectly predicted by `bio_thresh_pct > 25`.
**Status:** Dropped from regression. Kept in CSV for alert messaging.

### 6.5 MODIS Aqua Dying — ADDRESSED
**Problem:** MODIS Aqua decommissioning August 2026.
**Status:** ✅ Full Copernicus Chl migration complete. All 30 seasons now use Copernicus GlobColour L4 multi-sensor product. MODIS no longer needed for operations. Legacy MODIS data retained for reference.

### 6.6 SLA at Monthly Resolution — Insufficient
**Problem:** Monthly SLA averaged out transient Kelvin wave signals. 2014 S1 SLA_Z was -0.67 instead of expected +1.0+.
**What works:** SLA correctly flags major El Niño episodes (2015 S2: +1.80, 2023 S1: +1.40, 2023 S2: +2.15).
**What doesn't work:** Catches that are primarily detection-by-SLA events like 2014 where the signal is transient.
**Status:** SLA stored in CSV, excluded from regression (negative coefficient = fitting noise at n=28). Could revisit with weekly SLA or as sample size grows.

---

## 7. GROUND TRUTH DATA

### 7.1 IMARPE Season Outcomes (imarpe_ground_truth.csv)
30 rows covering 2010-2024. Binary target: NORMAL=0, AT-RISK (REDUCED/DISRUPTED/CANCELLED)=1.
- NORMAL: 18 seasons
- REDUCED: 4 seasons
- DISRUPTED: 7 seasons
- CANCELLED: 1 season (2023 S1)

### 7.2 Key Historical Events
- **2023 S1:** First season cancellation. 86% juveniles, $1.4B lost revenue. Model: prob=0.87-0.98
- **2015-2016:** Strong El Niño, multiple disruptions
- **2017 S1:** Coastal El Niño, quota cut
- **2024:** Recovery year after cancellation

---

## 8. EXTERNAL DATA SOURCES

| Data | Source | File | Status |
|------|--------|------|--------|
| Niño 1+2, 3, 3.4, 4 | NOAA CPC | nino_indices_monthly.csv | Active |
| Fishmeal prices | World Bank Pink Sheet | fishmeal_prices_monthly.csv | Active |
| Annual catch | FAO FishStatJ | peru_anchovy_catch_annual.csv | Active |
| SST (daily) | NOAA OISST via ERDDAP | baseline_v2/sst_YYYY.nc | Active |
| Chl (monthly L4) | Copernicus GlobColour | chl_copernicus_full.nc | ★ NEW PRIMARY |
| SLA (monthly) | Copernicus DUACS | sla_monthly_2010_2024.nc | ★ NEW (in CSV, not in regression) |
| Chl (8-day MODIS) | ERDDAP | baseline_v2_chl/ | LEGACY — decommissioning Aug 2026 |

### Copernicus Dataset IDs (verified working)
| Product | Dataset ID | Use |
|---------|-----------|-----|
| Merged Chl L4 monthly | `cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M` | Primary Chl baseline |
| SLA Multi-Year monthly | `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m` | SLA feature |
| SLA NRT daily | `SEALEVEL_GLO_PHY_L4_NRT_008_046` | Future: real-time Kelvin wave monitoring |

**Note:** The dataset ID in the original handoff (`cmems_obs-sl_glo_phy-ssh_my_static_all-sat-l4-duacs-0.25deg_P1D`) was incorrect. Use the IDs above.

---

## 9. PIPELINE EXECUTION ORDER (CURRENT)

```powershell
# 1. Copernicus Chl baseline (writes Chl Z-scores + chl_source='Copernicus' to CSV)
python chl_migration.py

# 2. SLA features (writes SLA Z-scores to CSV)
python sla_pipeline.py

# 3. Composite score (reads CSV ledger first, computes SST from raw files, runs regression)
python composite_score.py
```

**Architecture:** CSV-first. `composite_score.py` loads `paews_feature_matrix.csv` as a ledger before building features. For each season:
- If ledger has `chl_source=Copernicus` → uses that Chl_Z value (no raster computation)
- SLA pulled from ledger (written by sla_pipeline.py)
- SST always computed fresh from raw OISST .nc files (no sensor change)
- Falls back to MODIS Chl computation only if Copernicus unavailable

`gap_filler.py` is no longer needed in the regular workflow.

---

## 10. BIOMASS DATA (Phase 4 — Not Yet Implemented)

### Why It Matters
Acoustic biomass from IMARPE cruise reports is the "golden record" for resolving false alarms. The model currently can't distinguish "stressed environment + weak stock" from "stressed environment + strong stock."

### How to Find It
Search `repositorio.imarpe.gob.pe` for "crucero evaluación hidroacústica recursos pelágicos". Look for "biomasa de [X] millones de toneladas" in the Norte-Centro region. Two cruises per year (Feb-Apr for S1, Sep-Nov for S2).

### Implementation Notes
- Use lagged biomass (biomass_t-1) to maintain prediction lead time
- Gemini-provided biomass values were REJECTED — cross-check revealed up to 65% errors vs actual IMARPE reports
- Verified values: 2022 S1 = 10.20 MT, 2022 S2 = 7.18 MT (from IMARPE cruise reports)
- All values must be manually verified against IMARPE PDFs before ingestion

---

## 11. EXPERT REVIEWER FEEDBACK

### Adopted (in current code)
- ✅ Data-driven weights via logistic regression
- ✅ PR-AUC as primary metric (rare event detection)
- ✅ Season flag feature (summer anomalies deadlier)
- ✅ Biological threshold (absolute SST > 23°C)
- ✅ Niño 1+2 at t-1 lag (Peru-specific, not Niño 3.4)
- ✅ Static ocean mask (fixes >100% bug)
- ✅ Full Copernicus Chl history (replaces MODIS baseline)
- ✅ CSV-first architecture (prevents data overwrite race condition)
- ✅ SLA pipeline built (stored in CSV, monitoring only)

### Pending Implementation
- Persistence tracking (3 consecutive weeks before escalating)
- Contribution breakdown for alerts (transparency)
- North-Center vs South stock split (~10°S boundary)
- Weekly SLA for better Kelvin wave detection

### Deferred (Phase 4)
- Acoustic biomass integration
- AIS vessel tracking (Global Fishing Watch)
- Thermocline depth from reanalysis
- Wind/upwelling index

### Key Reviewer Intel
- Deep Chlorophyll Maximum trap: low satellite Chl + stable SLA = biomass at depth, not collapsed → encode as rule: `low_chl AND stable_sla = HOLD, not ALERT`
- IMARPE Cruise 2602-04 (Feb 16–Apr 4, 2026): live validation opportunity
- PRODUCE sometimes sets quotas 10-15% above IMARPE recommendations

---

## 12. IMMEDIATE NEXT STEPS (Priority Order)

### 12.1 ★ Run the 5-Feature Model (FIRST THING)
The updated `composite_score.py` has already been delivered. It drops SLA and thermal_shock from the regression (5 features instead of 7). Run:
```powershell
python composite_score.py
```
Expected: PR-AUC should recover from 0.695 toward ~0.74 by reducing overfitting. Post the full output.

### 12.2 Investigate Copernicus Chl Signal Compression
The core issue: Copernicus Z-scores have less variance than MODIS, so Chl went from 45% → 7% importance. Options:
- **Nearshore mask:** Restrict Chl computation to coastal upwelling pixels only (higher variance, more relevant to anchovy)
- **Higher temporal resolution:** Use 8-day or weekly Copernicus product instead of monthly
- **Hybrid approach:** Keep Copernicus for spatial coverage but weight nearshore pixels more heavily

### 12.3 Biomass Integration (Fixes False Alarm Cluster)
Manually extract acoustic biomass from IMARPE cruise reports for all 30 seasons. Use as lagged feature (biomass_t-1). This is the single biggest improvement available.

### 12.4 Persistence & Confidence Scoring
Rolling 3-week window: require compound condition for 3 consecutive weeks before escalating. Confidence = f(cloud-free pixel %).

---

## 13. PHASE ROADMAP

### Phase 1 ✅ COMPLETE: Detection
- SST baseline & anomaly detector
- Chlorophyll baseline & compound event detection

### Phase 2 ✅ MOSTLY COMPLETE: Composite Score & Validation
- Logistic regression with data-driven weights
- PR-AUC 0.695-0.743 (depending on feature set)
- IMARPE ground truth integration
- Full Copernicus Chl migration ✅
- SLA integration ✅ (built but excluded from regression — noise at n=28)
- **Remaining:** Run 5-feature model, investigate Chl signal compression

### Phase 3: Operational System
- Real-time monitoring dashboard
- Automated alert generation with lead time estimates
- Persistence tracking and confidence scores
- North-Center vs South stock split

### Phase 4: Enhanced Intelligence
- Acoustic biomass integration (IMARPE cruise reports)
- AIS vessel tracking (Global Fishing Watch)
- Thermocline depth from reanalysis
- Weekly SLA for Kelvin wave detection

---

## 14. KEY SCIENTIFIC INSIGHTS

1. **Chlorophyll dominates under MODIS (45%)** but compresses under Copernicus (7%). The multi-sensor product fills cloud gaps with offshore pixels, reducing coastal variance. Needs investigation — nearshore masking may restore signal.

2. **Biological threshold matters (14-19%):** Anchovy don't care about Z-scores — they care about exceeding 23°C absolute temperature. Once >25% of the shelf exceeds this, habitat is effectively closed.

3. **SLA at monthly resolution is insufficient for Kelvin wave detection.** 2014 March SLA was -0.67 (wrong sign). Monthly averaging smooths out transient pulses. SLA does correctly flag major El Niño episodes (2015-2016, 2023).

4. **Sensor bias is real and significant.** Copernicus reads 0.05-0.13 lower than MODIS in log10 space. The bias is largest in winter months (Jun-Sep). Z-scores are consistent within a single baseline, but switching baselines mid-series breaks the model.

5. **7 features on 28 samples = overfitting.** SLA coefficient went negative (opposite of theory). Pruning to 5 features is necessary until sample size grows.

6. **Deep Chlorophyll Maximum trap:** Low satellite Chl doesn't always mean collapsed food chain — phytoplankton may be at depth. If SLA is stable (no Kelvin wave), don't panic.

7. **False alarm pattern:** 2018-2020 false alarms = stressed environment but healthy stock. Only acoustic biomass data resolves this.

---

## 15. CURRENT CONDITIONS (Feb 25, 2026)

- **Niño 1+2:** -0.29°C (neutral, slightly cool)
- **Niño 3.4:** -0.04°C (neutral basin-wide)
- **IMARPE Cruise 2602-04:** Currently at sea (Feb 16–Apr 4)
- **Decision window:** We are in the S1 decision month. Season would start ~April.
- **MODIS status:** Degrading, scheduled decommission August 2026. No longer needed — Copernicus migration complete.
- **Sentinel-3:** Collection 4 update expected Feb 26, 2026

No current El Niño threat. System would likely read NORMAL for 2026 S1.

---

## 16. INSTRUCTIONS FOR NEW CHAT

Paste this document as the first message. Then say:

> "I'm continuing the PAEWS project. The handoff document above is current as of Feb 25, 2026. The immediate task is [describe what you want to do next]."

**If resuming from where we left off:**
The first thing to do is run `python composite_score.py` with the updated 5-feature version and post the output. The script has already been updated and delivered — it drops SLA and thermal_shock from the regression.

**Scripts the assistant may need:**
- `composite_score.py` — main model (5-feature version, CSV-first architecture)
- `chl_migration.py` — Copernicus Chl baseline builder
- `sla_pipeline.py` — SLA feature extraction

Upload whichever scripts are relevant to the task.
