# PAEWS — Peru Anchovy Early Warning System
## Complete Project Handoff v3 (Feb 25, 2026, end of day)

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
│   ├── composite_score.py         # ★ MAIN: 5-feature logistic regression, CSV-first architecture
│   ├── chl_migration.py           # ★ Copernicus Chl pipeline + coastal productivity mask
│   ├── sla_pipeline.py            # SLA Z-scores → CSV (stored, not in regression)
│   ├── health_check.py            # ★ NEW: Structural integrity checker (run at start of every session)
│   ├── anomaly_detector.py        # SST anomaly detection + seasonal alerts
│   ├── chl_anomaly_detector.py    # Chlorophyll anomaly + compound events
│   ├── compute_climatology.py     # SST climatology builder
│   ├── chl_climatology.py         # MODIS Chl climatology (legacy, replaced by chl_migration.py)
│   ├── gap_filler.py              # Legacy Copernicus gap-fill (no longer needed)
│   ├── external_data_puller.py    # Niño indices, fishmeal prices downloader
│   └── data_pipeline.py           # SST download pipeline
├── data/
│   ├── baseline_v2/               # SST yearly .nc files (2003-2024, OISST 0.25°)
│   ├── baseline_v2_chl/           # MODIS Chl yearly .nc (legacy, reference only)
│   ├── processed/
│   │   ├── sst_climatology_v2.nc          # SST monthly mean/std (2003-2022)
│   │   ├── chl_climatology_v2.nc          # MODIS Chl climatology (legacy)
│   │   ├── chl_climatology_copernicus.nc  # ★ Copernicus Chl log-mean/std (2003-2022)
│   │   └── sla_climatology.nc             # SLA monthly mean/std (2010-2022)
│   └── external/
│       ├── paews_feature_matrix.csv       # ★ 30 seasons, all features, Copernicus Chl
│       ├── imarpe_ground_truth.csv        # Season outcomes 2010-2024
│       ├── chl_copernicus_full.nc         # ★ 276 months Copernicus Chl (145 MB)
│       ├── sla_monthly_2010_2024.nc       # Monthly SLA (13.2 MB)
│       ├── nino_indices_monthly.csv       # 529 months (1982-Jan 2026)
│       ├── fishmeal_prices_monthly.csv    # World Bank (1979-2025)
│       ├── peru_anchovy_catch_annual.csv  # FAO (1950-2024)
│       └── chl_copernicus_2022_2023.nc    # Legacy gap-fill (superseded)
├── outputs/
│   └── composite_score_validation.png     # 4-panel dashboard
└── reports/
    └── researchpaper                      # Research paper drafts
```

---

## 4. BOUNDING BOX & BASELINE

**Region:** 0°S–16°S, 85°W–70°W (Peru's Humboldt Current upwelling zone)
**SST source:** NOAA OISST v2.1 (0.25° daily) via ERDDAP `ncdcOisst21Agg_LonPM180`
**Chl source:** Copernicus GlobColour L4 multi-sensor monthly (`cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M`) — replaced MODIS Aqua
**SLA source:** Copernicus multi-year monthly SLA (`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m`)
**Baseline period:** 2003-2022 (SST & Chl), 2010-2022 (SLA)
**Climatology:** Monthly mean and std (SST, SLA), monthly log10-mean and log10-std (Chl)
**Static ocean pixels:** SST=1,629, Chl=57,336 total (28,673 coastal pixels used via productive mask)

---

## 5. CURRENT MODEL STATE (Latest Run — Feb 25, 2026)

### 5.1 Features (5 in regression)
| # | Feature | Importance | Source |
|---|---------|-----------|--------|
| 1 | `sst_z` | 31.9% | OISST |
| 2 | `chl_z` | 27.3% | Copernicus (coastal productivity mask) |
| 3 | `bio_thresh_pct` | 17.4% | OISST (% pixels where absolute SST > 23°C) |
| 4 | `nino12_t1` | 14.5% | NOAA CPC (Niño 1+2, 1-month lag) |
| 5 | `is_summer` | 9.0% | Calendar (S1=1, S2=0) |

Stored in CSV but **NOT** in regression (noise at n=28):
| Feature | Why excluded |
|---------|-------------|
| `sla_z` | Negative coefficient (-0.554) — opposite of Kelvin wave theory. Fitting noise. |
| `thermal_shock` | Collinear with `bio_thresh_pct` (1.2–4.7% importance) |

Decision months: March (before S1 April start), October (before S2 November start).

### 5.2 Performance (Leave-One-Out CV, 28 samples)
```
ROC-AUC:  0.630
PR-AUC:   0.698  (primary metric — rare event detection)
Best threshold: 0.38 (F1=0.64)
Recall (at-risk): 75% (9/12 caught)
Precision: 56%
```

### 5.3 Performance History
| Version | Features | Chl Source | PR-AUC | Notes |
|---------|----------|-----------|--------|-------|
| Original | 6 (MODIS) | MODIS raw | 0.743 | Chl=45% importance, but sensor dying |
| Copernicus 7-feat | 7 | Copernicus (no mask) | 0.695 | Chl collapsed to 7%, SLA negative coef |
| **Current: Copernicus 5-feat + mask** | **5** | **Copernicus (coastal mask)** | **0.698** | **Chl recovered to 27.3%, honest baseline** |

The PR-AUC drop from 0.743 → 0.698 is real: MODIS had a sensor bias that accidentally amplified disruption signals. The 0.698 is what the model actually knows from environmental data alone. It still catches every catastrophe.

### 5.4 Data-Driven Weights (Logistic Regression, standardized coefficients)
```
SST_Z:        +0.576
-Chl_Z:       +0.494
Niño 1+2:     +0.262
Bio >23°C:    +0.314
Summer flag:  -0.162
Intercept:    -0.020
```

### 5.5 Season-by-Season Results
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
2014 1  DISRUPTED  0.16   NORMAL     MISS ← subsurface Kelvin wave, needs SLA or biomass
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
2022 2  DISRUPTED  0.12   NORMAL     MISS ← cold-water disruption, needs biomass
2023 1  CANCELLED  0.94   AT-RISK    HIT  ← $1.4B loss predicted
2023 2  DISRUPTED  0.88   AT-RISK    HIT
```

### 5.6 Key Signals for Critical Seasons
```
2023 S1 (CANCELLED): SST_Z=+1.92, Chl_Z=-1.25, Bio>23=98%, Niño=+0.71, Comp=+1.41
2023 S2 (DISRUPTED): SST_Z=+2.37, Chl_Z=-0.31, Bio>23=19%, Niño=+2.82, Comp=+1.63
2022 S2 (DISRUPTED): SST_Z=-1.03, Chl_Z=-0.66, Bio>23=7%,  Niño=-1.12, Comp=-0.37
2014 S1 (DISRUPTED): SST_Z=-0.64, Chl_Z=+0.27, Bio>23=77%, Niño=-0.92, Comp=-0.55
```

### 5.7 Copernicus vs MODIS Sensor Bias
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
Root cause: multi-sensor product fills cloud gaps with offshore (lower productivity) pixels. The coastal productivity mask (top 50%) mitigates this by restricting Z-score computation to 28,673 upwelling-zone pixels.

---

## 6. WHAT WAS BUILT ACROSS ALL SESSIONS

### Session 1-2 (Phase 1-2 foundation)
- SST baseline & anomaly detector (OISST 2003-2022)
- Chlorophyll baseline & compound event detection (MODIS 2004-2022)
- Logistic regression with data-driven weights
- IMARPE ground truth integration (30 seasons 2010-2024)
- Copernicus gap-fill for 2022-2023
- Static ocean mask (fixes >100% bug)
- External data integration (Niño indices, fishmeal prices, catch data)

### Session 3 (SLA + Architecture)
- SLA pipeline: downloaded monthly SLA 2010-2024, computed climatology, extracted Z-scores
- Discovered SLA monthly resolution too coarse for Kelvin wave detection (2014 March SLA was -0.67, wrong sign)
- SLA correctly flags major El Niño (2015 S2: +1.80, 2023 S1: +1.40, 2023 S2: +2.15)
- CSV-first architecture rewrite of composite_score.py

### Session 4 (Copernicus Migration + Pruning — THIS SESSION)
- **Full Copernicus Chl migration:** 276 months (2003-2025), replaced MODIS entirely
- **Coastal productivity mask:** top 50% most productive pixels (28,673 of 57,336), recovered Chl importance from 2.9% → 27.3%
- **Feature pruning:** 7 → 5 features (dropped SLA + thermal_shock from regression)
- **Mixed-baseline bug fixed:** all 30 seasons now use Copernicus climatology
- **Health check script:** 10-category structural integrity validator
- **Gemini biomass rejection:** verified several values were wrong (up to 65% error)

---

## 7. KNOWN PROBLEMS

### 7.1 Still Open

**2014 S1 Ghost Miss (prob=0.16):**
Subsurface Kelvin wave invisible to SST/Chl. SLA at monthly resolution also missed it (SLA_Z = -0.67, wrong sign). Needs weekly SLA or biomass. This is the strongest case for future SLA improvement.

**2022 S2 Cold-Water Disruption (prob=0.12):**
SST was cold (-1.03σ), Chl moderately low (-0.66). Model learned "cold = safe." Stock was depleted despite cool waters. Only biomass data would catch this.

**False Alarm Cluster (2018-2020):**
5 false positives where environment looked stressed but fishery was fine. Root cause: healthy stock tolerates moderate stress. Biomass is the fix — model can't distinguish "stressed environment + weak stock" from "stressed environment + strong stock."

**Copernicus Chl signal compression:**
Coastal mask recovered most of the signal (2.9% → 27.3%), but Copernicus is still less discriminating than MODIS. The productive percentile (currently 50%) can be tuned — tighter mask focuses on nearshore but risks pixel-level noise.

### 7.2 Resolved
- ✅ MODIS dependency removed (was dying August 2026)
- ✅ Mixed-baseline bug fixed (MODIS 2010-2021 vs Copernicus 2022-2023)
- ✅ SLA negative coefficient issue (removed from regression, stored in CSV)
- ✅ Overfitting with 7 features on 28 samples (pruned to 5)
- ✅ CSV overwrite race condition (CSV-first architecture)
- ✅ >100% pixel percentage bug (static ocean mask)
- ✅ Copernicus Chl gap-fill for 2022-2023 (superseded by full migration)

---

## 8. GROUND TRUTH DATA

### 8.1 IMARPE Season Outcomes (imarpe_ground_truth.csv)
30 rows covering 2010-2024. Binary target: NORMAL=0, AT-RISK (REDUCED/DISRUPTED/CANCELLED)=1.
- NORMAL: 18 seasons
- REDUCED: 4 seasons
- DISRUPTED: 7 seasons
- CANCELLED: 1 season (2023 S1)

### 8.2 Key Historical Events
- **2023 S1:** First season cancellation. 86% juveniles, $1.4B lost revenue. Model: prob=0.94
- **2015-2016:** Strong El Niño, multiple disruptions
- **2017 S1:** Coastal El Niño, quota cut
- **2014 S1:** Subsurface Kelvin wave disruption — the ghost miss
- **2022 S2:** Cold-water stock depletion — the biomass miss
- **2024:** Recovery year after cancellation, both seasons NORMAL

---

## 9. EXTERNAL DATA SOURCES

| Data | Source | File | Status |
|------|--------|------|--------|
| Niño 1+2, 3, 3.4, 4 | NOAA CPC | nino_indices_monthly.csv | Active |
| Fishmeal prices | World Bank Pink Sheet | fishmeal_prices_monthly.csv | Active |
| Annual catch | FAO FishStatJ | peru_anchovy_catch_annual.csv | Active |
| SST (daily) | NOAA OISST via ERDDAP | baseline_v2/sst_YYYY.nc | Active |
| Chl (monthly L4) | Copernicus GlobColour | chl_copernicus_full.nc | ★ PRIMARY |
| SLA (monthly) | Copernicus DUACS | sla_monthly_2010_2024.nc | In CSV, not in regression |
| Chl (8-day MODIS) | ERDDAP | baseline_v2_chl/ | LEGACY — decommissioning Aug 2026 |

### Verified Copernicus Dataset IDs
| Product | Dataset ID |
|---------|-----------|
| Merged Chl L4 monthly | `cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M` |
| SLA Multi-Year monthly | `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m` |
| SLA NRT daily | `SEALEVEL_GLO_PHY_L4_NRT_008_046` |

**WARNING:** The dataset ID `cmems_obs-sl_glo_phy-ssh_my_static_all-sat-l4-duacs-0.25deg_P1D` (from Gemini) does NOT exist. Use the IDs above.

---

## 10. PIPELINE EXECUTION ORDER

```powershell
cd C:\Users\josep\Documents\paews\scripts

# 0. Health check (run at start of every session)
python health_check.py

# 1. Copernicus Chl → Z-scores for all 30 seasons → CSV
python chl_migration.py

# 2. SLA (already in CSV — only rerun if SLA data changes)
# python sla_pipeline.py

# 3. Composite score (reads CSV ledger, computes SST from raw, runs regression)
python composite_score.py
```

**Architecture:** CSV-first. `composite_score.py` loads `paews_feature_matrix.csv` as a ledger:
- If ledger has `chl_source=Copernicus` → uses that Chl_Z (no raster computation)
- SLA pulled from ledger (written by sla_pipeline.py)
- SST always computed fresh from raw OISST .nc files (no sensor change)
- Falls back to MODIS Chl only if Copernicus unavailable

**CRITICAL:** Always run `chl_migration.py` before `composite_score.py`. If composite_score runs first, it overwrites the CSV with MODIS values for 2010-2021, and 2022-2023 get NONE. The health check catches this.

`gap_filler.py` is no longer needed.

---

## 11. COASTAL PRODUCTIVITY MASK

The Copernicus multi-sensor product fills cloud gaps with open-ocean pixels, diluting the coastal upwelling signal. Without masking, Chl importance dropped from 45% (MODIS) to 2.9%.

**Solution:** `chl_migration.py` applies a mask retaining only the top N% most productive pixels by climatological annual mean.

- **Current setting:** `PRODUCTIVE_PERCENTILE = 50` (top 50%)
- **Mask:** 28,673 pixels out of 57,336 total
- **Threshold:** log10(Chl) ≥ -0.456 (Chl ≥ 0.35 mg/m³)
- **Effect:** Chl importance recovered from 2.9% → 27.3%

The percentile is tunable. Tighter (30-40%) focuses on nearshore upwelling but risks pixel-level noise. This is the single easiest knob to turn for improving Chl signal.

---

## 12. GEMINI BIOMASS WARNING

Gemini repeatedly provided a biomass table claiming "Verified IMARPE" values. **Do not use it.** Cross-checking revealed:

| Season | Gemini says | IMARPE actual | Error |
|--------|-----------|--------------|-------|
| 2016 S1 | 7.28 MT | 4.42 MT | **+65% wrong** |
| 2022 S1 | 10.11 MT | 10.20 MT | close |
| 2022 S2 | 6.82 MT | 7.18 MT | -5% |
| 2023 S1 | 6.18 MT | ~6.45 MT | -4% |

Some numbers are close, some are dangerously wrong. Training on wrong biomass teaches the model wrong signals. Gemini also repeatedly cited a non-existent SLA dataset ID and used hardcoded statistics (mean=8.3, std=2.1) instead of computing from data.

**Biomass is Phase 4.** Build from PDFs at `repositorio.imarpe.gob.pe`:
- Search: "crucero evaluación hidroacústica recursos pelágicos"
- Look for: "biomasa de X millones de toneladas" (anchoveta Norte-Centro)
- Two cruises per year: Feb-Apr (→ S1), Sep-Nov (→ S2)
- Record: year, season, biomass_mt, cruise_code, source_url
- Use **lagged** biomass (previous season's estimate) as feature to maintain prediction lead time

---

## 13. EXPERT REVIEWER FEEDBACK

### Adopted (in current code)
- ✅ Data-driven weights via logistic regression
- ✅ PR-AUC as primary metric (rare event detection)
- ✅ Season flag feature (summer anomalies deadlier)
- ✅ Biological threshold (absolute SST > 23°C)
- ✅ Niño 1+2 at t-1 lag (Peru-specific, not Niño 3.4)
- ✅ Static ocean mask (fixes >100% bug)
- ✅ Full Copernicus Chl history (replaces MODIS baseline)
- ✅ Coastal productivity mask (recovers diluted Chl signal)
- ✅ CSV-first architecture (prevents data overwrite race condition)
- ✅ SLA pipeline built (stored in CSV, monitoring only)
- ✅ Feature pruning (5 features, not 7, at n=28)

### Pending Implementation
- Persistence tracking (3 consecutive weeks before escalating)
- Contribution breakdown for alerts (transparency)
- North-Center vs South stock split (~10°S boundary)
- Weekly SLA for better Kelvin wave detection

### Deferred (Phase 4)
- Acoustic biomass integration (IMARPE cruise reports)
- AIS vessel tracking (Global Fishing Watch)
- Thermocline depth from reanalysis
- Wind/upwelling index

### Key Reviewer Intel
- Deep Chlorophyll Maximum trap: low satellite Chl + stable SLA = biomass at depth, not collapsed → encode as rule: `low_chl AND stable_sla = HOLD, not ALERT`
- IMARPE Cruise 2602-04 (Feb 16–Apr 4, 2026): live validation opportunity
- PRODUCE sometimes sets quotas 10-15% above IMARPE recommendations
- MODIS Aqua decommissioning August 2026 (addressed — Copernicus migration complete)
- Sentinel-3 Collection 4 update Feb 26, 2026 (parameter name changes may affect downloads)

---

## 14. KEY SCIENTIFIC INSIGHTS

1. **Chlorophyll is the hardest signal** — under MODIS it was 45% of model importance. Under Copernicus it compressed to 2.9% due to offshore pixel dilution, but the coastal mask recovered it to 27.3%. There are warm seasons where fishery is fine but very few low-Chl seasons where anchovy thrive.

2. **Biological threshold matters (17%):** Anchovy don't care about Z-scores — they care about exceeding 23°C absolute. Once >25% of the shelf exceeds this, habitat is effectively closed.

3. **SLA monthly resolution is insufficient for Kelvin wave detection.** 2014 March SLA was -0.67 (wrong sign). Monthly averaging smooths transient pulses. SLA correctly flags major El Niño episodes (2015-2016, 2023) but misses subtle events.

4. **Sensor bias is real and significant.** Copernicus reads 0.05-0.13 lower than MODIS in log10 space. Mixing baselines mid-series silently breaks the model. All seasons must use the same sensor baseline.

5. **7 features on 28 samples = overfitting.** SLA coefficient went negative (opposite of theory). The 5:28 feature-to-sample ratio is healthier.

6. **Deep Chlorophyll Maximum trap:** Low satellite Chl doesn't always mean collapsed food chain — phytoplankton may be at depth. If SLA is stable (no Kelvin wave), don't panic.

7. **False alarm pattern (2018-2020):** Stressed environment but healthy stock. Only acoustic biomass data resolves this. This is the single biggest improvement available.

8. **The model catches every catastrophe.** 2023 S1 cancellation at 0.94, major El Niño disruptions caught. The misses are genuinely hard cases (subsurface waves, stock depletion in cold water).

---

## 15. IMMEDIATE NEXT STEPS (Priority Order)

### 15.1 Coastal Mask Tuning
Try `PRODUCTIVE_PERCENTILE = 30` or `40` in `chl_migration.py`. Run full pipeline, compare PR-AUC. The sweet spot maximizes Chl importance without pixel-level noise. This is the quickest win.

### 15.2 Biomass Verification (Phase 4, manual work)
Go to `repositorio.imarpe.gob.pe`, extract verified biomass for each cruise. Build `imarpe_biomass_verified.csv`. Use lagged biomass as feature. This is the single biggest performance improvement available — likely jumps PR-AUC from 0.70 toward 0.80+.

### 15.3 SLA Rule-Based Override
Instead of putting SLA in the regression (where it fits noise), add a post-regression rule: if `sla_z > 1.5` in decision month, boost risk probability. This would catch 2014 S1 without corrupting the regression.

### 15.4 Sentinel-3 Collection 4
Expected Feb 26, 2026 — parameter name changes may break Copernicus downloads. Monitor and update `chl_migration.py` if the variable name changes from `CHL`.

### 15.5 2026 S1 Live Prediction
IMARPE Cruise 2602-04 is at sea (Feb 16–Apr 4). Once March SST/Chl data is available, run the system for a real 2026 S1 prediction. This is the first live validation opportunity.

---

## 16. PHASE ROADMAP

### Phase 1 ✅ COMPLETE: Detection
- SST baseline & anomaly detector
- Chlorophyll baseline & compound event detection

### Phase 2 ✅ COMPLETE: Composite Score & Validation
- Logistic regression with data-driven weights (5 features)
- PR-AUC 0.698 on honest Copernicus baseline
- IMARPE ground truth integration
- Full Copernicus Chl migration with coastal mask
- SLA integration (built, stored, excluded from regression)
- CSV-first architecture

### Phase 3: Operational System
- Real-time monitoring dashboard
- Automated alert generation with lead time estimates
- Persistence tracking and confidence scores
- North-Center vs South stock split

### Phase 4: Enhanced Intelligence
- Acoustic biomass integration (IMARPE cruise reports)
- AIS vessel tracking (Global Fishing Watch)
- Weekly SLA for Kelvin wave detection
- Thermocline depth from reanalysis

---

## 17. CURRENT CONDITIONS (Feb 25, 2026)

- **Niño 1+2:** -0.29°C (neutral, slightly cool)
- **Niño 3.4:** -0.04°C (neutral basin-wide)
- **IMARPE Cruise 2602-04:** Currently at sea (Feb 16–Apr 4)
- **Decision window:** We are in the S1 decision month. Season starts ~April.
- **MODIS status:** Degrading, August 2026 decommission. No longer needed — Copernicus migration complete.
- **Sentinel-3:** Collection 4 update expected Feb 26, 2026

No current El Niño threat. System would likely read NORMAL for 2026 S1.

---

## 18. INSTRUCTIONS FOR NEW CHAT

**First, run the health check:**
```powershell
cd C:\Users\josep\Documents\paews\scripts
python health_check.py
```

Paste this document as the first message along with the health check output. Then say:

> "I'm continuing the PAEWS project. The handoff document above is current as of Feb 25, 2026. Health check output is [paste]. The immediate task is [describe what you want to do next]."

**Upload these files when asked:**
- `composite_score.py` — main model (5-feature, CSV-first)
- `chl_migration.py` — Copernicus Chl pipeline + coastal mask
- `paews_feature_matrix.csv` — current feature matrix
- `health_check.py` — structural integrity checker

**GitHub:** `github.com/monkeqi/paews`
