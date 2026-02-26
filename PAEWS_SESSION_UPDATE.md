# PAEWS Session Update — Feb 25, 2026
## Continuation from PAEWS_HANDOFF.md (Feb 24, 2026)

---

## WHAT CHANGED THIS SESSION

### 1. Copernicus Chl Migration (COMPLETE)
The entire MODIS chlorophyll baseline has been replaced with the unified Copernicus GlobColour L4 multi-sensor product. This was the top infrastructure priority (MODIS Aqua decommissions August 2026).

- **Downloaded:** `chl_copernicus_full.nc` — 276 months (Jan 2003–Dec 2025), 384×360 grid, 145 MB
- **New climatology:** `chl_climatology_copernicus.nc` — 2003–2022 baseline, log-space, saved in `data/processed/`
- **All 30 seasons** now use Copernicus Chl Z-scores (tagged `chl_source=Copernicus` in the CSV)
- **MODIS raw files** still in `baseline_v2_chl/` for reference but are no longer used in the pipeline

### 2. Coastal Productivity Mask (NEW)
Copernicus fills cloud gaps with open-ocean pixels that diluted the upwelling signal. Without a mask, Chl importance dropped from 45% (MODIS) to 2.9% (Copernicus).

**Fix:** `chl_migration.py` now applies a productive pixel mask — only the top 50% most productive pixels (by climatological mean) are included in the area-mean Z-score.

- **Mask:** 28,673 coastal/upwelling pixels out of 57,336 total
- **Threshold:** log10(Chl) ≥ -0.456 (Chl ≥ 0.35 mg/m³)
- **Effect:** Chl importance recovered from 2.9% → 27.3%
- **Tunable:** `PRODUCTIVE_PERCENTILE = 50` at top of `chl_migration.py`

### 3. CSV-First Architecture (composite_score.py rewrite)
`composite_score.py` was rewritten so `build_feature_matrix()` loads the CSV as a "ledger" first:

- **Chl:** Checks if ledger has `chl_source=Copernicus` → uses it. Falls back to MODIS only if missing.
- **SLA:** Pulled from ledger (written by `sla_pipeline.py`).
- **SST:** Always recomputed from raw OISST `.nc` files (no sensor change issue).
- **Niño:** Always recomputed from indices CSV.
- **Gap-fill merge** now only triggers for SST-missing rows (2024).
- `gap_filler.py` is **no longer needed** in the regular pipeline flow.

### 4. Feature Pruning: 7 → 5 Features
SLA and thermal_shock were removed from the logistic regression:

- **SLA** had a negative coefficient (-0.554) — higher SLA = lower predicted risk, opposite of Kelvin wave theory. Fitting noise with 28 samples.
- **thermal_shock** was collinear with `bio_thresh_pct` (1.2–4.7% importance).
- Both are still computed and stored in the CSV for future use when more samples or verified biomass become available.

---

## CURRENT MODEL STATE

### Features (5 in regression)
| # | Feature | Importance | Source |
|---|---------|-----------|--------|
| 1 | `sst_z` | 31.9% | OISST |
| 2 | `chl_z` | 27.3% | Copernicus (coastal mask) |
| 3 | `nino12_t1` | 14.5% | NOAA CPC |
| 4 | `bio_thresh_pct` | 17.4% | OISST (% pixels > 23°C) |
| 5 | `is_summer` | 9.0% | Calendar |

### Performance (Leave-One-Out CV, 28 samples)
```
ROC-AUC:  0.630
PR-AUC:   0.698  (primary metric)
Best threshold: 0.38 (F1=0.64)
Recall (at-risk): 75% (9/12 caught)
Precision: 56%
```

### Comparison to Previous Baseline
| Metric | MODIS 6-feature (old) | Copernicus 5-feature (now) |
|--------|----------------------|---------------------------|
| PR-AUC | 0.743 | 0.698 |
| Chl weight | 45.0% | 27.3% |
| 2023 S1 (CANCELLED) | prob=0.98 ✓ | prob=0.94 ✓ |
| 2023 S2 (DISRUPTED) | prob=0.98 ✓ | prob=0.88 ✓ |
| 2022 S2 (DISRUPTED) | prob=0.33 ✓ | prob=0.12 ✗ |
| Valid features | 28/30 | 28/30 |

### Season-by-Season Results
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
2014 1  DISRUPTED  0.16   NORMAL     MISS ← still missed (needs SLA or biomass)
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

---

## PIPELINE EXECUTION ORDER (UPDATED)

```powershell
cd C:\Users\josep\Documents\paews\scripts

# 1. Copernicus Chl → Z-scores for all 30 seasons → CSV
python chl_migration.py

# 2. SLA (already in CSV from prior run — only rerun if SLA data changes)
# python sla_pipeline.py

# 3. Composite score (reads CSV ledger, computes SST from raw, runs regression)
python composite_score.py
```

`gap_filler.py` is no longer needed in the regular flow.

---

## DIRECTORY STRUCTURE (UPDATED)

```
C:\Users\josep\Documents\paews\
├── scripts/
│   ├── composite_score.py         # ★ MAIN: 5-feature logistic regression, CSV-first
│   ├── chl_migration.py           # ★ Copernicus Chl pipeline + coastal mask
│   ├── sla_pipeline.py            # SLA Z-scores → CSV (feature stored, not in regression)
│   ├── anomaly_detector.py        # SST anomaly detection + seasonal alerts
│   ├── chl_anomaly_detector.py    # Chlorophyll anomaly + compound events
│   ├── compute_climatology.py     # SST climatology builder
│   ├── chl_climatology.py         # MODIS Chl climatology (legacy, replaced)
│   ├── gap_filler.py              # Legacy Copernicus gap-fill (no longer needed)
│   ├── external_data_puller.py    # Niño indices, fishmeal prices downloader
│   └── data_pipeline.py           # SST download pipeline
├── data/
│   ├── baseline_v2/               # SST yearly .nc files (2003-2023, OISST 0.25°)
│   ├── baseline_v2_chl/           # MODIS Chl yearly .nc (legacy, reference only)
│   ├── processed/
│   │   ├── sst_climatology_v2.nc          # SST monthly mean/std (2003-2022)
│   │   ├── chl_climatology_v2.nc          # MODIS Chl climatology (legacy)
│   │   └── chl_climatology_copernicus.nc  # ★ NEW: Copernicus Chl climatology
│   └── external/
│       ├── paews_feature_matrix.csv       # ★ 30 seasons, all features, Copernicus Chl
│       ├── imarpe_ground_truth.csv        # Season outcomes 2010-2024
│       ├── chl_copernicus_full.nc         # ★ NEW: 276 months Copernicus Chl (145 MB)
│       ├── chl_copernicus_2022_2023.nc    # Legacy gap-fill (superseded)
│       ├── nino_indices_monthly.csv       # 529 months (1982-Jan 2026)
│       ├── fishmeal_prices_monthly.csv    # World Bank (1979-2025)
│       └── peru_anchovy_catch_annual.csv  # FAO (1950-2024)
└── outputs/
    └── composite_score_validation.png     # 4-panel dashboard
```

---

## KNOWN PROBLEMS (UPDATED)

### Still Open
1. **2014 S1 Ghost Miss (prob=0.16):** Subsurface Kelvin wave, invisible to SST/Chl. SLA would fix but was dropped due to noise with 28 samples. Revisit when sample size grows or with rule-based SLA override.

2. **2022 S2 Cold-Water Disruption (prob=0.12):** SST was *cold* (-1.03σ), Chl moderately low (-0.66). Model learned "cold = safe." This is the case biomass data would catch — stock was depleted despite cool waters.

3. **False Alarm Cluster (2018-2020):** 5 false positives where environment looked stressed but fishery was fine. Biomass is the fix — healthy stock tolerates moderate stress.

4. **Copernicus Chl signal compression:** Coastal mask recovered most of the signal (2.9% → 27.3%), but Copernicus Chl is still less discriminating than MODIS was. The productive percentile (currently 50%) can be tuned — a tighter mask (e.g., top 30%) would focus more on nearshore upwelling but risks small-sample noise at the pixel level.

### Resolved This Session
- ✅ MODIS dependency removed (was dying August 2026)
- ✅ Mixed-baseline bug fixed (MODIS 2010-2021 vs Copernicus 2022-2023)
- ✅ SLA negative coefficient issue (removed from regression, stored in CSV)
- ✅ Overfitting with 7 features on 28 samples (pruned to 5)
- ✅ CSV overwrite race condition (CSV-first architecture)

---

## GEMINI BIOMASS WARNING

Gemini repeatedly provided a biomass table claiming "Verified IMARPE" values. **Do not use it.** We verified several entries against actual IMARPE reports and found:

| Season | Gemini says | IMARPE actual | Error |
|--------|-----------|--------------|-------|
| 2016 S1 | 7.28 MT | 4.42 MT | +65% wrong |
| 2022 S1 | 10.11 MT | 10.20 MT | close |
| 2022 S2 | 6.82 MT | 7.18 MT | -5% |
| 2023 S1 | 6.18 MT | ~6.45 MT | -4% |

Some numbers are close, some are dangerously wrong. Training on wrong biomass would teach the model the wrong signals. Also: the SLA dataset ID Gemini cites (`cmems_obs-sl_glo_phy-ssh_my_static_all-sat-l4-duacs-0.25deg_P1D`) doesn't exist — we had to find the real one ourselves.

**Biomass is Phase 4.** Build it from PDFs you personally download from `repositorio.imarpe.gob.pe`. Search "crucero evaluación hidroacústica recursos pelágicos" for each year. Look for "biomasa de X millones de toneladas" for anchoveta Norte-Centro. Two cruises per year (Feb-Apr → S1, Sep-Nov → S2). Record: year, season, biomass_mt, cruise_code.

---

## IMMEDIATE NEXT STEPS (PRIORITY ORDER)

### 1. Coastal Mask Tuning
Try `PRODUCTIVE_PERCENTILE = 30` or `40` in `chl_migration.py` to tighten the mask further toward nearshore upwelling. Run full pipeline, compare PR-AUC. The sweet spot maximizes Chl importance without introducing pixel-level noise.

### 2. Biomass Verification (Phase 4, manual work)
Go to `repositorio.imarpe.gob.pe`, extract verified biomass for each cruise. Build `imarpe_biomass_verified.csv` with columns: `year, season, biomass_mt, cruise_code, source_url`. Use lagged biomass (previous season's estimate) as a feature — it's the "stock health bank account" that explains why stressed environments sometimes produce normal seasons.

### 3. SLA Rule-Based Override
Instead of putting SLA in the regression (where it fits noise), add a post-regression rule: if `sla_z > 1.5` in decision month, boost risk probability by a fixed amount. This would catch 2014 S1 without corrupting the regression. Requires more analysis of the SLA-outcome relationship first.

### 4. Sentinel-3 Collection 4
Expected Feb 26, 2026 — parameter name changes may break Copernicus downloads. Monitor and update `chl_migration.py` if the variable name changes from `CHL`.

---

## CURRENT CONDITIONS (Feb 25, 2026)

- **Niño 1+2:** -0.29°C (neutral, slightly cool)
- **Niño 3.4:** -0.04°C (neutral basin-wide)
- **IMARPE Cruise 2602-04:** At sea (Feb 16–Apr 4)
- **Decision window:** S1 decision month (March). Season starts ~April.
- **System prediction:** Would likely read NORMAL for 2026 S1 based on neutral indices.

---

## INSTRUCTIONS FOR NEW CHAT

Paste this document plus `PAEWS_HANDOFF.md` as the first message. Then say:

> "I'm continuing the PAEWS project. These documents are current as of Feb 25, 2026. The immediate task is [describe what you want to do next]."

Upload these files when asked: `composite_score.py`, `chl_migration.py`, `paews_feature_matrix.csv`.

**GitHub:** `github.com/monkeqi/paews` (should be current after the push)
