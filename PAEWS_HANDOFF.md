# PAEWS — Peru Anchovy Early Warning System
## Complete Project Handoff (Feb 24, 2026)

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
& C:/Users/josep/miniconda3/Scripts/conda.exe run -n geosentinel python c:/Users/josep/Documents/paews/scripts/[script].py
```
Or simply from the scripts folder:
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
│   ├── gap_filler.py              # Copernicus Chl integration for 2022-2023
│   ├── external_data_puller.py    # Niño indices, fishmeal prices downloader
│   └── data_pipeline.py           # SST download pipeline
├── data/
│   ├── baseline_v2/               # SST yearly .nc files (2003-2022, OISST 0.25°)
│   ├── baseline_v2_chl/           # Chl yearly .nc files (2004-2022, MODIS 4km)
│   ├── processed/
│   │   ├── sst_climatology_v2.nc  # Monthly mean/std (2003-2022)
│   │   └── chl_climatology_v2.nc  # Monthly log-mean/std (2004-2022)
│   └── external/
│       ├── paews_feature_matrix.csv        # ★ 30 seasons, all features
│       ├── imarpe_ground_truth.csv         # Season outcomes 2010-2024
│       ├── nino_indices_monthly.csv        # 529 months (1982-Jan 2026)
│       ├── fishmeal_prices_monthly.csv     # World Bank (1979-2025)
│       ├── peru_anchovy_catch_annual.csv   # FAO (1950-2024)
│       └── chl_copernicus_2022_2023.nc     # Gap-fill data (12.7 MB)
└── outputs/
    └── composite_score_validation.png      # 4-panel dashboard
```

---

## 4. BOUNDING BOX & BASELINE

**Region:** 0°S–16°S, 85°W–70°W (Peru's Humboldt Current upwelling zone)
**SST source:** NOAA OISST v2.1 (0.25° daily) via ERDDAP `ncdcOisst21Agg_LonPM180`
**Chl source:** MODIS Aqua 8-day (4km) via ERDDAP `erdMH1chla8day` (ends mid-2022)
**Chl gap-fill:** Copernicus GlobColour L4 multi-sensor monthly (`cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M`)
**Baseline period:** 2003-2022 (SST), 2004-2022 (Chl) — clean, no El Niño contamination in reference
**Climatology:** Monthly mean and std (SST), monthly log10-mean and log10-std (Chl)
**Static ocean pixels:** SST=1,629, Chl=57,747 (fixed denominator for % calculations)

---

## 5. CURRENT MODEL STATE (Latest Run)

### 5.1 Features (6 total)
| # | Feature | Description | Source |
|---|---------|-------------|--------|
| 1 | `sst_z` | SST Z-score in decision month | OISST |
| 2 | `chl_z` | Chl Z-score (inverted: negative = bad) | MODIS/Copernicus |
| 3 | `nino12_t1` | Niño 1+2 anomaly, 1-month lag | NOAA CPC |
| 4 | `is_summer` | Binary: S1=1, S2=0 | Calendar |
| 5 | `bio_thresh_pct` | % pixels where absolute SST > 23°C | OISST |
| 6 | `thermal_shock` | Binary: 1 if bio_thresh > 25% | Derived |

Decision months: March (before S1 April start), October (before S2 November start).

### 5.2 Data-Driven Weights (Logistic Regression, 28 samples)
```
SST:           24.1%
Chlorophyll:   45.0%  ← dominant feature
Niño 1+2:     14.6%
Season flag:    1.2%
Bio >23°C:    13.8%
Thermal Shock:  1.2%  ← collinear with bio_thresh, needs more samples
```

### 5.3 Performance (Leave-One-Out CV)
```
ROC-AUC:  0.740
PR-AUC:   0.743  (primary metric — rare event detection)
Best threshold: 0.33 (F1=0.69)
Recall (at-risk): 83% (10/12 caught)
Precision: 59%
```

### 5.4 Season-by-Season Results
```
Year S  Outcome    Prob     Pred     Result
2010 1  NORMAL     0.67   AT-RISK    MISS (false alarm)
2010 2  NORMAL     0.22   NORMAL     HIT
2011 1  NORMAL     0.17   NORMAL     HIT
2011 2  REDUCED    0.24   NORMAL     MISS (missed disruption)
2012 1  REDUCED    0.49   AT-RISK    HIT
2012 2  REDUCED    0.74   AT-RISK    HIT
2013 1  NORMAL     0.12   NORMAL     HIT
2013 2  NORMAL     0.56   AT-RISK    MISS (false alarm)
2014 1  DISRUPTED  0.11   NORMAL     MISS ← CRITICAL: SLA would fix
2014 2  REDUCED    0.54   AT-RISK    HIT
2015 1  REDUCED    0.48   AT-RISK    HIT
2015 2  DISRUPTED  0.77   AT-RISK    HIT
2016 1  DISRUPTED  0.84   AT-RISK    HIT
2016 2  NORMAL     0.31   NORMAL     HIT
2017 1  REDUCED    0.87   AT-RISK    HIT
2017 2  NORMAL     0.14   NORMAL     HIT
2018 1  NORMAL     0.24   NORMAL     HIT
2018 2  NORMAL     0.80   AT-RISK    MISS (false alarm)
2019 1  NORMAL     0.56   AT-RISK    MISS (false alarm)
2019 2  NORMAL     0.44   AT-RISK    MISS (false alarm)
2020 1  NORMAL     0.58   AT-RISK    MISS (false alarm)
2020 2  NORMAL     0.16   NORMAL     HIT
2021 1  NORMAL     0.05   NORMAL     HIT
2021 2  NORMAL     0.25   NORMAL     HIT
2022 1  NORMAL     0.50   AT-RISK    MISS (false alarm)
2022 2  DISRUPTED  0.33   AT-RISK    HIT
2023 1  CANCELLED  0.98   AT-RISK    HIT ← 🎯 $1.4B loss predicted
2023 2  DISRUPTED  0.98   AT-RISK    HIT ← 🎯
```

### 5.5 Key Signals for Critical Seasons
```
2023 S1 (CANCELLED): SST_Z=+1.92, Chl_Z=-0.90, Bio>23=98%, Niño=+0.71, Composite=+1.27
2023 S2 (DISRUPTED): SST_Z=+2.37, Chl_Z=-0.97, Bio>23=19%, Niño=+2.82, Composite=+1.90
2022 S2 (DISRUPTED): SST_Z=-1.03, Chl_Z=-0.77, Bio>23=7%,  Niño=-1.12, Composite=-0.33
```

---

## 6. KNOWN PROBLEMS & THEIR FIXES

### 6.1 The 2014 Ghost Miss (Physics Fail)
**Problem:** 2014 S1 was DISRUPTED but our model gave prob=0.11. SST was cool (-0.64σ), Chl looked healthy (+0.37). El Niño developed AFTER the decision month — a Kelvin wave was propagating subsurface.
**Fix:** Sea Level Anomaly (SLA). Kelvin waves show as +10-15cm bulge in SSH weeks before SST responds.

### 6.2 False Alarm Cluster (2018-2020)
**Problem:** 7 false positives where environment looked stressed but fishery was fine.
**Fix:** Acoustic biomass data. Healthy stock tolerates moderate environmental stress. Without biomass, model can't distinguish "stressed environment + weak stock" from "stressed environment + strong stock."

### 6.3 Thermal Shock Collinearity
**Problem:** `thermal_shock` binary only gets 1.2% importance because it's perfectly predicted by `bio_thresh_pct > 25`.
**Fix:** Keep for alert messaging ("HABITAT CLOSED") but it won't add predictive power until sample size grows or model goes nonlinear.

### 6.4 MODIS Aqua Dying
**Problem:** MODIS Aqua decommissioning August 2026. Our entire Chl baseline is MODIS. Sensor already degrading — 2026 data may be darker/noisier.
**Fix:** Transition to full Copernicus L4 multi-sensor product (MODIS+VIIRS+OLCI). Already used for 2022-2023 gap fill. Need to build unified 1997-2025 Chl history.

### 6.5 Chlorophyll Weight Shift
**Observation:** Chl jumped from 31% to 45% importance after adding 2023 data. This is real — chlorophyll collapse is a harder signal than SST warmth. There are warm seasons where fishery is fine (2016 S2: SST+1.26, NORMAL) but very few low-Chl seasons where anchovy thrive.

---

## 7. GROUND TRUTH DATA

### 7.1 IMARPE Season Outcomes (imarpe_ground_truth.csv)
30 rows covering 2010-2024. Columns: year, season, start_date, end_date, quota_mt, actual_catch_mt, catch_pct, outcome, el_nino_flag, notes.

Outcomes classified as:
- **NORMAL (18):** Full or near-full quota caught
- **REDUCED (4):** Significant quota cut or < 80% catch
- **DISRUPTED (7):** Major disruption, severe quota cut
- **CANCELLED (1):** 2023 S1 — first ever full cancellation

Binary target: NORMAL=0, AT-RISK (REDUCED/DISRUPTED/CANCELLED)=1

### 7.2 Key Historical Events
- **2023 S1:** First season cancellation. 86% juveniles, $1.4B lost revenue. Our model: prob=0.98
- **2015-2016:** Strong El Niño, multiple disruptions
- **2017 S1:** Coastal El Niño, quota cut. Detected in February as ALERT
- **1984:** Historical Super El Niño collapse: 23K MT catch (from 12.3M peak in 1970)
- **2024:** Recovery year after cancellation, quota doubled to pre-Niño levels

---

## 8. EXTERNAL DATA SOURCES

| Data | Source | File | Update Frequency |
|------|--------|------|-----------------|
| Niño 1+2, 3, 3.4, 4 | NOAA CPC `sstoi.indices` | nino_indices_monthly.csv | Monthly |
| ONI (ENSO official) | NOAA CPC | In Niño CSV | Monthly |
| Fishmeal prices | World Bank Pink Sheet | fishmeal_prices_monthly.csv | Monthly |
| Annual catch | FAO FishStatJ | peru_anchovy_catch_annual.csv | Annual |
| SST (daily) | NOAA OISST via ERDDAP | baseline_v2/sst_YYYY.nc | Daily |
| Chl (8-day) | MODIS Aqua via ERDDAP | baseline_v2_chl/chl_YYYY.nc | 8-day (ends mid-2022) |
| Chl (monthly L4) | Copernicus GlobColour | chl_copernicus_2022_2023.nc | Monthly |

Current Niño 1+2 value (Jan 2026): -0.29°C (neutral, slightly cool off Peru).

---

## 9. COPERNICUS DATA PRODUCTS IDENTIFIED

| Product | ID | Use |
|---------|-----|-----|
| Merged Chl L4 monthly | `cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M` | Gap-fill + future baseline |
| SLA NRT daily | `SEALEVEL_GLO_PHY_L4_NRT_008_046` | Real-time Kelvin wave monitoring |
| SLA Multi-Year daily | `SEALEVEL_GLO_PHY_L4_MY_008_047` | Backtesting SLA feature |

Download command for SLA (example):
```bash
copernicusmarine get -i cmems_obs-sl_glo_phy-ssh_my_all-sat-l4-duacs-0.25deg_P1D \
  --longitude-min -85 --longitude-max -70 \
  --latitude-min -20 --latitude-max 0 \
  --start-datetime 2014-01-01T00:00:00 \
  --end-datetime 2014-06-01T00:00:00 \
  -o ./data/external/sla_2014.nc
```

---

## 10. EXPERT REVIEWER FEEDBACK (3 independent reviewers)

### 10.1 Adopted (in current code)
- ✅ Data-driven weights via logistic regression
- ✅ PR-AUC as primary metric (rare event detection)
- ✅ Season flag feature (summer anomalies deadlier)
- ✅ Biological threshold (absolute SST > 23°C)
- ✅ Niño 1+2 at t-1 lag (Peru-specific, not Niño 3.4)
- ✅ Static ocean mask (fixes >100% bug)
- ✅ Copernicus multi-sensor Chl for gap-fill
- ✅ Thermal shock binary flag
- ✅ CSV-first architecture (prevents data overwrite race condition)

### 10.2 Adopted in principle, pending implementation
- SLA as #1 next feature (Kelvin wave detection, fixes 2014 miss)
- Persistence tracking (3 consecutive weeks before escalating)
- Contribution breakdown for alerts (transparency)
- Full Copernicus L4 Chl history (replace MODIS baseline)
- North-Center vs South stock split (~10°S boundary)

### 10.3 Deferred to later phases
- Thermocline depth (Z20) — requires TAO buoys or reanalysis
- Wind/upwelling index (Bakun) — SST already captures the effect
- Dissolved oxygen / OMZ compression — model data only
- Salinity — poor satellite resolution
- AIS vessel tracking (Global Fishing Watch) — Phase 4
- South Pacific High pressure index — atmospheric driver

### 10.4 Explicitly rejected
- Fishmeal-to-soybean price ratio as model feature (circular logic for trading use)
- Geopandas polygon mask (current bbox sufficient, code had bugs)

### 10.5 Critical intel from reviewers
- MODIS Aqua decommissioning **August 2026** (6-month deadline)
- Sentinel-3 Collection 4 update Feb 26, 2026 (parameter name changes)
- Deep Chlorophyll Maximum trap: low satellite Chl + stable SLA = biomass at depth, not collapsed → encode as rule: `low_chl AND stable_sla = HOLD, not ALERT`
- IMARPE Cruise 2602-04 (Feb 16–Apr 4, 2026): live validation opportunity
- PRODUCE sometimes sets quotas 10-15% above IMARPE recommendations (economic pressure buffer)
- Acoustic biomass (SSB) from IMARPE cruise reports = "golden record" for resolving false alarms

---

## 11. COMPOSITE SCORE FORMULA

### Hardcoded (physics-based baseline):
```
Composite_Z = 0.4 × SST_Z + 0.4 × (-Chl_Z) + 0.2 × Niño12(t-1)
```

### Data-driven (logistic regression):
6 features → StandardScaler → LogisticRegression(L2, C=1.0, class_weight='balanced')
Leave-One-Out CV for honest performance estimates.
Chl sign is flipped (negative Chl = bad = positive risk signal).

### Proposed with SLA:
```
Composite_Z = 0.3 × SST_Z + 0.3 × (-Chl_Z) + 0.2 × SLA_Z + 0.1 × Niño12(t-1) + bio_thresh
```
(Exact weights to be determined by re-fitting logistic regression once SLA is integrated.)

---

## 12. PIPELINE EXECUTION ORDER

```powershell
# 1. Gap-fill missing chlorophyll (Copernicus data → CSV)
python gap_filler.py

# 2. Composite score (reads CSV, fills gaps, runs regression)
python composite_score.py

# 3. Individual backtests (optional)
python anomaly_detector.py --backtest 2017 --month 3
python chl_anomaly_detector.py --backtest 2017 --month 2
```

**Critical flow:** gap_filler.py writes to `paews_feature_matrix.csv` → composite_score.py reads CSV first, only recomputes from raw .nc files for rows with missing SST/Chl, then merges gap-filled values for 2022-2023 seasons.

---

## 13. IMMEDIATE NEXT STEPS (Priority Order)

### 13.1 SLA Integration (fixes 2014 miss, adds prediction lead time)
1. Download SLA from Copernicus (multi-year product for 2010-2024 backtesting)
2. Compute SLA climatology (monthly mean/std)
3. Add `sla_z` column to feature matrix
4. Re-run logistic regression with 7 features
5. Expected: 2014 S1 catch probability rises from 0.11 → 0.5+

### 13.2 Full Copernicus Chl History (replaces dying MODIS)
1. Download `cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M` for 1997-2025
2. Compute new Chl climatology from unified sensor-merged data
3. Replace MODIS baseline entirely
4. Must complete before MODIS Aqua decommission Aug 2026

### 13.3 Persistence & Confidence Scoring
1. Rolling 3-week window: require compound condition for 3 consecutive weeks before escalating
2. Confidence = f(cloud-free pixel %). If 60% masked, confidence drops proportionally

### 13.4 Alert Log CSV
Track cumulative stress, spatial coverage, contribution breakdown per feature per alert.

---

## 14. PHASE ROADMAP

### Phase 1 ✅ COMPLETE: Detection
- SST baseline & anomaly detector
- Chlorophyll baseline & compound event detection
- Validated: 2017 Coastal Niño ALERT, 2019 NORMAL confirmed, 2023 cancellation predicted at 0.98

### Phase 2 ✅ MOSTLY COMPLETE: Composite Score & Validation
- Logistic regression with 6 features
- PR-AUC 0.743, 83% recall
- IMARPE ground truth integration
- Copernicus gap-fill for 2022-2023
- **Remaining:** SLA integration, full Copernicus Chl history

### Phase 3: Operational System
- Real-time monitoring dashboard
- Automated alert generation with lead time estimates
- Persistence tracking and confidence scores
- North-Center vs South stock split

### Phase 4: Enhanced Intelligence
- Acoustic biomass integration (IMARPE cruise reports)
- AIS vessel tracking (Global Fishing Watch)
- Thermocline depth from reanalysis
- SPH pressure index

---

## 15. KEY SCIENTIFIC INSIGHTS

1. **Chlorophyll dominates (45%):** Not SST. There are warm seasons where fishery is fine but very few low-Chl seasons where anchovy thrive. Food chain collapse is the harder signal.

2. **Biological threshold matters (14%):** Anchovy don't care about Z-scores — they care about exceeding 23°C. Once >25% of the shelf exceeds this, the habitat is effectively closed.

3. **2014 confirms SLA necessity:** Cool SST (-0.64σ) and healthy Chl (+0.37) in March, but season disrupted. The Kelvin wave was subsurface — SLA would have shown +10-15cm bulge weeks before SST responded.

4. **Nearshore resilience:** During El Niño, nearshore Chl stays resilient (+0.04σ in Feb 2017) while offshore collapses (-0.57σ). Coastal upwelling buffers nearshore productivity.

5. **False alarm pattern:** 2018-2020 false alarms = stressed environment but healthy stock. Acoustic biomass data would resolve — healthy stock tolerates moderate stress.

6. **Deep Chlorophyll Maximum trap:** Low satellite Chl doesn't always mean collapsed food chain — phytoplankton may be at depth. If SLA is stable (no Kelvin wave), don't panic.

---

## 16. CURRENT CONDITIONS (Feb 24, 2026)

- **Niño 1+2:** -0.29°C (neutral, slightly cool)
- **Niño 3.4:** -0.04°C (neutral basin-wide)
- **IMARPE Cruise 2602-04:** Currently at sea (Feb 16–Apr 4)
- **Decision window:** We are in the S1 decision month. Season would start ~April.
- **MODIS status:** Degrading, scheduled decommission August 2026
- **Sentinel-3:** Collection 4 update expected Feb 26, 2026

No current El Niño threat. System would likely read NORMAL for 2026 S1 based on neutral Niño indices.

---

## 17. INSTRUCTIONS FOR NEW CHAT

Paste this document as the first message. Then say:

> "I'm continuing the PAEWS project. The handoff document above is current as of Feb 24, 2026. The immediate task is [describe what you want to do next]."

The new assistant should:
1. Confirm it has context
2. Ask you to upload any files it needs (scripts are on your machine, not in chat)
3. Continue from wherever you left off

If you want to work on SLA integration, upload `composite_score.py` and `gap_filler.py` so the assistant can see the current code.
