# PAEWS Handoff v7 — Complete Reference

**Peruvian Anchovy Early Warning System**
**Date:** March 2, 2026
**Session:** 13 (cumulative)
**Status:** Model v2 production. Niño 1+2 surge detected. Prediction shifting toward ELEVATED.

---

## 1. What PAEWS Is

A satellite-based early warning system that predicts whether Peru's anchovy fishing season will be NORMAL or DISRUPTED, using freely available remote sensing data and climate indices. Target audience: Norwegian salmon companies exposed to fishmeal price volatility (Peru = ~20% of global fishmeal).

**GitHub:** `https://github.com/monkeqi/paews`
**Local path:** `C:\Users\josep\Documents\paews\`
**Environment:** Windows, PowerShell, VS Code, conda (`paews` environment), Python

---

## 2. Model v2 Architecture (3 features, 32 samples)

### Core Specs

| Metric | Value |
|--------|-------|
| Samples | 32 seasons (2010 S1 – 2025 S2) |
| Positives | 12 disrupted seasons |
| Features | **sst_z, chl_z, nino12_t1** |
| Algorithm | Logistic Regression (StandardScaler) |
| LOO ROC-AUC | **0.629** |
| Framework | Risk tiers (no binary threshold) |

### Coefficients (standardized, from March 2 run)

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| sst_z | +0.390 | Warmer SST → higher risk |
| chl_z | -0.583 | Lower chlorophyll → higher risk |
| nino12_t1 | +0.363 | El Niño (coastal) → higher risk |
| intercept | -0.589 | |

### Risk Tiers (replaces old binary threshold=0.38)

| Tier | Prob Range | Historical Disruption Rate | Action |
|------|-----------|---------------------------|--------|
| LOW | <0.20 | ~30% | No action |
| MODERATE | 0.20–0.50 | ~29% | Monitor |
| ELEVATED | 0.50–0.70 | ~25% | Monitor closely |
| **SEVERE** | **≥0.70** | **100% (4/4, 0 false positives)** | **Act** |

**Critical insight:** Only SEVERE is actionable. LOW/MODERATE/ELEVATED are statistically indistinguishable (~25-33% disruption). The model is a binary extreme-event detector, not a gradient predictor. Middle-tier outcomes are driven by biomass state (which the model cannot see).

### Why 3 Features (not 5)

The previous model used 5 features: sst_z, chl_z, nino12_t1, is_summer, bio_thresh_pct. Sessions 12-13 discovered:

- **Multicollinearity:** is_summer and bio_thresh_pct had Pearson r=0.963. S1 bio_thresh mean=76%, S2 mean=8% — they encoded the same seasonality signal.
- **4-model comparison** showed dropping both collinear features gave best ROC-AUC (0.629 vs 0.583 baseline).
- **Lagged landings** (previous season catch) was tested and rejected — worsened performance, caused overfitting, produced ecologically backwards coefficients.
- **Data audit** (10-point integrity check) passed with zero errors before model change.

Legacy 5-feature model preserved in `predict_2026_s1_v1_backup.py`. The v2 prediction script runs both side-by-side.

---

## 3. 2026 S1 Live Prediction

### Latest Run Output (March 2, 2026)

```
Model v2 (3-feature):
  SST Z:    +0.837  (OISST through Feb 15)
  Chl Z:    +0.166  (Copernicus Dec 2025 proxy — STALE)
  Niño 1+2: -0.290  (CPC Jan 2026 monthly — STALE)

  Probability: 0.308  [MODERATE]
  Bootstrap median: 0.307, 95% CI: [0.062, 0.657]
  Tier distribution: LOW 26%, MODERATE 60%, ELEVATED 13%, SEVERE 2%

Legacy (5-feature):
  Probability: 0.363  [NORMAL, threshold=0.38]
```

### CRITICAL: Niño 1+2 Surge (not yet in model)

CPC weekly data (Feb 23, 2026) shows Niño 1+2 has jumped from **−0.29** (Jan monthly) to **+1.0°C** (weekly). A 1.3-point swing in one month — this is a phase shift, not noise. Subsurface warming expanding eastward. IRI forecasts El Niño probabilities 58-61% by May-Jul 2026.

The official CPC February monthly value is not yet published. Expected any day now (first week of March).

### Scenario Impact Table

| Scenario | Niño 1+2 | Chl Z | Prob | Tier |
|----------|----------|-------|------|------|
| Current run (stale data) | −0.29 | +0.17 (Dec proxy) | 0.308 | MODERATE |
| Updated Niño only | +0.70 | +0.17 | ~0.37 | MODERATE |
| Updated Niño high | +1.00 | +0.17 | ~0.40 | MODERATE |
| Niño +0.70, Feb Chl proxy | +0.70 | −0.40 | ~0.49 | MODERATE |
| **Niño +1.00, Feb Chl proxy** | **+1.00** | **−0.40** | **~0.57** | **ELEVATED** |
| Niño +1.00, bad Chl | +1.00 | −0.80 | ~0.67 | ELEVATED |
| Worst case (2017-like) | +1.50 | −1.00 | ~0.78 | **SEVERE** |

### What Resolves the Prediction

1. **CPC Feb Niño 1+2 monthly** — imminent. Will confirm +0.7 to +1.0.
2. **Copernicus Jan 2026 Chl monthly** — should be downloadable now. Replaces Dec proxy.
3. **Copernicus Feb 2026 Chl** — available ~mid-March.
4. **SST through end of Feb** — run data_pipeline.py.
5. **Threshold:** If prob crosses 0.70 → SEVERE → act. 100% historical accuracy.

### External Context (as of March 2, 2026)

- **ENFEN** (Feb 13): El Niño Costero Alert — weak starting March, possibly moderate by July
- **SNP** president warned "not normal conditions" for 2026 S1
- **CPC** (Feb 23): La Niña Advisory still active but transitioning to ENSO-neutral (60% chance Feb-Apr)
- **IRI** (Feb 19): La Niña only 4% probability for Feb-Apr; El Niño 58-61% by May-Jul
- Subsurface warming strengthening and expanding eastward across Pacific

---

## 4. Data Sources & Sensor Notes

### Satellite Data

| Dataset | File | Notes |
|---------|------|-------|
| SST (training) | `data/baseline/sst_YYYY.nc` | NOAA OISST v2.1, annual files 2010-2025 |
| SST (current) | `data/current/sst_current.nc` | OISST 2026, through Feb 15 |
| Chl (training) | `data/external/chl_copernicus_full.nc` | Copernicus GlobColour L4 monthly, 2003-2025 |
| Chl (current) | `data/current/chlorophyll_current.nc` | VIIRS daily 2026, **REQUIRES BIAS CORRECTION** |

**SENSOR BIAS WARNING:** VIIRS chlorophyll reads +0.4 in log10 space vs Copernicus. This was caught in Session 11 — it flipped the prediction from 7% to 50%. The prediction script uses Copernicus as primary source and applies −0.4 bias correction if falling back to VIIRS. **Never mix sensors without correction.**

### External Data

| File | Contents | Latest |
|------|----------|--------|
| `data/external/nino_indices_monthly.csv` | CPC Niño indices | Through Jan 2026 |
| `data/external/oni_monthly.csv` | Oceanic Niño Index | Through Jan 2026 |
| `data/external/CMO-Historical-Data-Monthly.xlsx` | World Bank fishmeal prices | Through Dec 2025 ($1824/MT) |
| `data/external/imarpe_ground_truth.csv` | Season outcomes | 32 rows (2010 S1 – 2025 S2) |
| `data/external/imarpe_biomass_verified.csv` | Biomass estimates | 11 verified values |
| `data/external/paews_feature_matrix.csv` | Model-ready features | 32 rows |

### Climatologies

| File | Description |
|------|-------------|
| `data/processed/sst_climatology_v2.nc` | Monthly SST mean/std by pixel |
| `data/processed/chl_climatology_copernicus.nc` | Monthly Chl log10 mean/std by pixel |

---

## 5. Scripts Reference

### Production Scripts (use these)

```
python scripts/predict_2026_s1.py           # 2026 S1 prediction (v2 + legacy comparison)
python scripts/scenario_analysis.py         # Sweep Niño/Chl combos, risk matrix
python scripts/data_pipeline.py             # Download current SST + Chl from ERDDAP
python scripts/external_data_puller.py      # Update Niño, ONI, fishmeal from CPC/WorldBank
```

### Analysis Scripts

```
python scripts/model_v2_audit.py            # 10-point data integrity audit
python scripts/model_v3_collinearity.py     # 4-model comparison, multicollinearity analysis
python scripts/model_diagnosis.py           # Hindcast validation, threshold analysis
python scripts/hindcast_validation.py       # Full LOO hindcast with metrics
python scripts/compute_2025_and_predict.py  # Add 2025 + retrain + bootstrap
```

### Other Pipeline Scripts

```
python chl_migration.py                     # Process Copernicus Chl files
python composite_score.py                   # Generate risk scores (original pipeline)
python health_check.py                      # Validate (33 passes expected)
```

### Scenario Analysis Usage

```powershell
# Default: current SST (+0.837), sweep full Niño/Chl range
python scripts/scenario_analysis.py

# Zoom into likely Feb 2026 range
python scripts/scenario_analysis.py --nino-min 0.5 --nino-max 1.5 --chl-min -0.8 --chl-max 0.2

# Override SST (e.g., if March SST changes)
python scripts/scenario_analysis.py --sst 1.2

# Export CSV for dashboard or spreadsheet
python scripts/scenario_analysis.py --csv outputs/scenario_matrix.csv
```

Outputs: probability matrix, tier map, key scenarios table, SEVERE threshold finder.

### Interactive Dashboard

`scripts/index.html` — React dashboard (also available as `paews_dashboard.jsx`). Four panels:
- Timeline (LOO predictions by season, colored by tier)
- Tier Calibration (disruption rates per tier)
- Feature Space (SST vs Chl scatter, dot size = Niño magnitude)
- Bootstrap (tier distribution, CI visualization)

Toggle between Dec Chl proxy (MODERATE) and Feb Chl proxy (ELEVATED) by clicking the gauges.

**Note:** Dashboard data is hardcoded from earlier sessions. Some values differ slightly from the March 2 run due to SST updates. Update the SEASONS array in the JSX to match latest `predict_2026_s1.py` output if needed.

---

## 6. Ground Truth (32 seasons)

File: `data/external/imarpe_ground_truth.csv`

| Year | S1 | S2 | Notes |
|------|----|----|-------|
| 2010 | NORMAL | NORMAL | |
| 2011 | NORMAL | DISRUPTED | No quota set |
| 2012 | DISRUPTED | DISRUPTED | Biomass-driven |
| 2013 | NORMAL | NORMAL | |
| 2014 | DISRUPTED | DISRUPTED | Recruitment failure |
| 2015 | DISRUPTED | DISRUPTED | El Niño onset |
| 2016 | DISRUPTED | NORMAL | El Niño peak |
| 2017 | DISRUPTED | NORMAL | Coastal El Niño |
| 2018 | NORMAL | NORMAL | Recovery |
| 2019 | NORMAL | NORMAL | |
| 2020 | NORMAL | NORMAL | COVID affected but quota met |
| 2021 | NORMAL | NORMAL | |
| 2022 | NORMAL | DISRUPTED | Biomass collapse |
| 2023 | DISRUPTED | DISRUPTED | El Niño + Costero |
| 2024 | NORMAL | NORMAL | Strong recovery |
| 2025 | NORMAL | NORMAL | |

---

## 7. Biomass Data

File: `data/external/imarpe_biomass_verified.csv`

**11 verified values** (from IMARPE Boletín publications by Castillo et al.):

| Year | S1 (MMT) | S2 (MMT) | Source |
|------|----------|----------|--------|
| 2018 | 11.21 | 8.78 | Castillo et al. (2021) Bol IMARPE 35(2) |
| 2019 | 8.82 | 8.38* | Castillo et al. (2021) Bol IMARPE 35(2) |
| 2020 | 11.05 | 9.52 | Castillo et al. (2021) Inf Inst Mar Peru 48(3) |
| 2021 | 12.03 | 8.03 | Castillo et al. (2023) Bol IMARPE 38(1) |
| 2022 | 7.13 | 4.68 | Castillo et al. (2024) Bol IMARPE 39(1) |
| 2025 S1 | 10.93 | — | MarinTrust WF02 + SeafoodSource |

*2019 S2 has integrity_flag=0 (IMARPE officials investigated for inflating biomass).

**Still missing:** 2010–2017 (all), 2023 (both), 2024 (both), 2025 S2.

---

## 8. Verification Protocol

1. Nothing enters CSVs without a source URL, PDF page number, or Ministerial Resolution number.
2. Never trust LLMs for specific numbers — 65% error rate proven (Session 4, Gemini hallucination).
3. Primary sources only: IMARPE PDFs, PRODUCE Resoluciones, NOAA/CPC official data.
4. Cross-sensor data must be calibrated before comparison (VIIRS vs Copernicus: −0.4 log10).
5. Web-sourced data gets verified against 2+ independent sources where possible.
6. Any new feature matrix rows must pass the 10-point audit (`model_v2_audit.py`).

---

## 9. Key Discoveries & Dead Ends

### Things That Worked
- Copernicus GlobColour L4 monthly as Chl training source (consistent with climatology)
- 40% coastal mask (balances upwelling signal vs data coverage)
- 3-feature model (avoids multicollinearity, best ROC-AUC)
- Risk tier framework (SEVERE = only actionable tier, 100% accuracy)
- Bootstrap CI for uncertainty quantification (2000 resamples)

### Dead Ends (Don't Repeat These)
- **SLA (Sea Level Anomaly):** Only goes back to 2014. Coverage gap kills it.
- **Coastal mask >50% or <30%:** Both degrade performance. 40% is optimal.
- **VIIRS as Chl training source:** +0.4 log10 bias vs Copernicus. Never mix without correction.
- **LLM-generated biomass numbers:** 65% error rate (Session 4).
- **Lagged landings feature:** Overfitting, backwards coefficients. Rejected Session 12.
- **is_summer + bio_thresh_pct together:** r=0.963 collinear. Use neither.
- **Binary threshold (0.38):** Middle range has flat ~30% disruption rate regardless of threshold.

---

## 10. Session History

| Session | Key Achievement |
|---------|----------------|
| 1–3 | Initial build, Copernicus migration, ground truth creation |
| 4 | Gemini hallucination discovery (65% biomass error), verification protocols |
| 5 | Mask tuning (50%→40%), SLA dead end, PR-AUC 0.705 |
| 6 | SST 2024 added (30 samples), PR-AUC baseline 0.682, biomass CSV created |
| 7 | Uploaded Castillo 2023 PDF — confirmed 2021 biomass |
| 8 | Searched for remaining biomass papers — only 2018-2021 digitized |
| 9 | Uploaded Castillo 2021 — confirmed 2020 biomass, English abstract error |
| 10 | Handoff v5, biomass integration script tested |
| 11 | 2026 S1 first prediction: 0.33-0.50. VIIRS sensor bias caught (+0.4 log10). 2025 added (32 samples). |
| 12 | Hindcast validation: ROC-AUC 0.583. Lagged landings rejected. Risk tier calibration: SEVERE 100%, middle tiers flat. |
| 13 | Multicollinearity fix (is_summer/bio_thresh r=0.963). 3-feature model (ROC-AUC 0.629). Data audit passed. Production script v2. Dashboard. Scenario sweep script. Niño surge to +1.0 detected. |

---

## 11. File Tree

```
C:\Users\josep\Documents\paews\
├── data/
│   ├── baseline/                     # Annual SST (sst_2010.nc – sst_2025.nc)
│   ├── current/                      # Live 2026 data
│   │   ├── sst_current.nc            #   OISST through Feb 15
│   │   └── chlorophyll_current.nc    #   VIIRS Jan-Feb 2026
│   ├── copernicus_chl/               # 30 annual Chl files (1997-2024)
│   ├── processed/
│   │   ├── sst_climatology_v2.nc
│   │   └── chl_climatology_copernicus.nc
│   └── external/
│       ├── paews_feature_matrix.csv  # 32 rows, model input
│       ├── imarpe_ground_truth.csv   # 32 season outcomes
│       ├── imarpe_biomass_verified.csv
│       ├── chl_copernicus_full.nc    # Copernicus monthly 2003-2025
│       ├── nino_indices_monthly.csv  # CPC through Jan 2026
│       ├── oni_monthly.csv
│       └── CMO-Historical-Data-Monthly.xlsx
├── scripts/
│   ├── predict_2026_s1.py            # PRODUCTION v2 (3-feat + legacy)
│   ├── predict_2026_s1_v1_backup.py  # Old 5-feature version
│   ├── scenario_analysis.py          # Niño/Chl sweep, risk matrix
│   ├── model_v2_audit.py             # Data integrity audit
│   ├── model_v3_collinearity.py      # Multicollinearity analysis
│   ├── model_diagnosis.py            # Hindcast validation
│   ├── hindcast_validation.py        # Full LOO hindcast
│   ├── compute_2025_and_predict.py   # Add 2025 + retrain
│   ├── external_data_puller.py       # Download Niño/ONI/fishmeal
│   ├── data_pipeline.py              # Download SST/Chl from ERDDAP
│   ├── index.html                    # React dashboard
│   └── PAEWS_HANDOFF_v7.md           # Previous handoff (replace with this)
├── masks/coastal_mask_40pct.nc
├── chl_migration.py
├── composite_score.py
└── health_check.py
```

---

## 12. Immediate Priority (March 2026)

**Daily until mid-March:**

```powershell
cd C:\Users\josep\Documents\paews
conda activate paews
python scripts/external_data_puller.py    # Check for Feb Niño monthly
python scripts/predict_2026_s1.py         # Re-predict if data updated
```

**When Feb Niño appears:**

```powershell
python scripts/scenario_analysis.py       # See full risk matrix
python scripts/predict_2026_s1.py         # Official updated prediction
```

**Also check:**
- Copernicus Marine for Jan 2026 monthly Chl composite (download manually)
- If prob crosses 0.70 → SEVERE → 100% historical accuracy → act

**Upcoming data milestones:**
- CPC Feb Niño monthly: any day now (first week March)
- Copernicus Jan 2026 Chl: should be available
- Copernicus Feb 2026 Chl: ~mid-March
- Next CPC ENSO Diagnostic: March 12, 2026
- IMARPE survey results: typically announced March-April

---

## 13. Quick Start for New Chat

Paste this to start a new session:

> I'm working on PAEWS — a satellite-based anchovy early warning system for Peru's fishing seasons. Upload PAEWS_HANDOFF_v7.md for full context. Current state: Model v2 uses 3 features (sst_z, chl_z, nino12_t1) after fixing multicollinearity in Session 13. LOO ROC-AUC 0.629. SEVERE tier (≥0.70) = 100% accuracy, 0 false positives. Latest prediction 0.308 MODERATE but Niño 1+2 has surged from −0.29 to +1.0 weekly (not yet in model). CPC Feb monthly Niño expected any day — once it lands plus updated Chl, prediction likely moves to ELEVATED (~0.57). Priority: update data and re-predict. Repo: github.com/monkeqi/paews. Environment: Windows, PowerShell, conda (paews), Python.
