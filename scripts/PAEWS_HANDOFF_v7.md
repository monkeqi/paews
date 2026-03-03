# PAEWS Handoff v7 — Session 13 Complete

**Peruvian Anchovy Early Warning System**
**Date:** March 2, 2026
**Status:** Model v2 production (3-feature), Niño 1+2 surge detected (+1.0 weekly), prediction shifting toward ELEVATED

---

## 1. What PAEWS Is

A satellite-based early warning system that predicts whether Peru's anchovy fishing season will be NORMAL or DISRUPTED, using freely available remote sensing data (SST, Chlorophyll-a) and climate indices (Niño 1+2). Target: issue warnings 1–2 months before IMARPE's official cruise results.

**GitHub:** `https://github.com/monkeqi/paews`
**Local path:** `C:\Users\josep\Documents\paews\`

---

## 2. Current Model v2 (3 features, 32 samples)

| Metric | Value |
|--------|-------|
| Samples | **32** (2010 S1 – 2025 S2) |
| Positives | 12 disrupted seasons |
| Features | **sst_z, chl_z, nino12_t1** |
| LOO ROC-AUC | **0.629** |
| Approach | Risk tiers, no binary threshold |

**Feature weights (standardized coefficients):**

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| sst_z | +0.390 | Warmer → higher risk |
| chl_z | -0.583 | Lower chlorophyll → higher risk |
| nino12_t1 | +0.363 | El Niño → higher risk |
| intercept | -0.589 | |

**Risk tiers (replaces old binary threshold):**

| Tier | Range | Historical Disruption Rate | Action |
|------|-------|---------------------------|--------|
| LOW | <0.20 | ~30% | No action needed |
| MODERATE | 0.20–0.50 | ~29% | Monitor |
| ELEVATED | 0.50–0.70 | ~25% | Monitor closely |
| **SEVERE** | **>0.70** | **100% (4/4)** | **Act — 0 false positives** |

**Critical insight:** Only the SEVERE tier is actionable. LOW/MODERATE/ELEVATED are statistically indistinguishable (~25-33%). The model is an extreme-event detector, not a gradient.

### Why 3 features (Sessions 12-13)

- **Multicollinearity discovered:** is_summer and bio_thresh_pct had r=0.963. S1 bio mean 76%, S2 bio mean 8% — they were the same signal.
- **4-model comparison:** Dropping both collinear features gave best ROC-AUC (0.629 vs 0.583 baseline) and simplest model.
- **Lagged landings rejected:** Adding previous season's catch worsened performance (overfitting, ecologically backwards coefficients).
- **Data audit passed:** 10-point integrity check, zero errors, all 32 targets match ground truth.

### Legacy model (5-feature, for reference)

Still available via `predict_2026_s1_v1_backup.py`. Uses sst_z, chl_z, nino12_t1, is_summer, bio_thresh_pct with threshold=0.38. ROC-AUC 0.583. The v2 script runs both models side-by-side for comparison.

---

## 3. 2026 S1 LIVE PREDICTION

### Latest Run (March 2, 2026)

```
Model v2 (3-feature):
  Probability: 0.308  [MODERATE]
  Bootstrap median: 0.307, 95% CI: [0.062, 0.657]
  Tier distribution: LOW 26%, MODERATE 60%, ELEVATED 13%, SEVERE 2%

Legacy (5-feature):
  Probability: 0.363  [NORMAL, threshold=0.38]
  Bootstrap median: 0.377, 95% CI: [0.037, 0.803]
```

### CRITICAL: Niño 1+2 Surge (data not yet in model)

CPC weekly data (Feb 23, 2026) shows Niño 1+2 has jumped from −0.29 (Jan monthly) to **+1.0°C weekly**. This is a 1.3-point swing in one month — a phase shift, not noise.

**Impact on prediction (estimated):**

| Scenario | Niño 1+2 | Chl Z | Prob | Tier |
|----------|----------|-------|------|------|
| Current (stale Jan) | −0.29 | +0.17 (Dec proxy) | 0.308 | MODERATE |
| Updated Niño only | +0.70 | +0.17 | ~0.37 | MODERATE |
| Updated Niño high | +1.00 | +0.17 | ~0.40 | MODERATE |
| **Worst-case realistic** | **+1.00** | **−0.40 (Feb proxy)** | **~0.57** | **ELEVATED** |

The official CPC Feb 2026 monthly Niño value is not yet published (expected first week of March). Once it is, run `python scripts/external_data_puller.py` then re-predict.

### Feature Values (from March 2 run)

| Feature | Value | Training Range | Source |
|---------|-------|---------------|--------|
| sst_z | +0.837 | [-1.36, +2.37] | OISST through Feb 15 |
| chl_z | +0.166 | [-1.25, +0.91] | Copernicus Dec 2025 (proxy) |
| nino12_t1 | −0.29 | [-1.43, +2.82] | CPC Jan 2026 (STALE) |

### External Context

- **ENFEN** (Feb 13): El Niño Costero Alert — weak, starting March, possibly moderate by July
- **SNP** president: warned "not normal conditions" for 2026 S1
- **CPC** (Feb 23): Niño 1+2 weekly at +1.0. La Niña → ENSO-neutral transition expected Feb-Apr (60%)
- **IRI** (Feb 19): El Niño probabilities 58-61% by May-Jul 2026
- Subsurface warming expanding eastward; Kelvin wave propagation underway

### What Resolves the Prediction

1. **CPC Feb Niño 1+2 monthly** — imminent (any day now). Will confirm +0.7 to +1.0 range.
2. **Copernicus Jan 2026 Chl monthly** — should be downloadable now. Replaces Dec proxy.
3. **Copernicus Feb 2026 Chl** — available ~mid-March.
4. **SST update** — currently through Feb 15. Run data_pipeline.py for full Feb.
5. If prob crosses 0.70 → SEVERE → that's the real signal (100% hist. accuracy).

### Run Command

```powershell
cd C:\Users\josep\Documents\paews
python scripts/predict_2026_s1.py
```

---

## 4. Data Sources & Pipeline

### Satellite Data
- **SST:** NOAA OISST v2.1, `data/baseline/sst_YYYY.nc` (2010-2025), `data/current/sst_current.nc` (2026 through Feb 15)
- **Chl training:** Copernicus GlobColour L4, `data/external/chl_copernicus_full.nc` (2003-2025 monthly)
- **Chl current:** VIIRS daily, `data/current/chlorophyll_current.nc` (2026 Jan-Feb, REQUIRES BIAS CORRECTION)

### External Data
- **Niño indices:** `data/external/nino_indices_monthly.csv` — 529 months (1982–Jan 2026)
- **ONI:** `data/external/oni_monthly.csv`
- **Fishmeal prices:** `data/external/CMO-Historical-Data-Monthly.xlsx` — World Bank, through Dec 2025
- **Ground truth:** `data/external/imarpe_ground_truth.csv` — 32 rows (2010 S1–2025 S2)
- **Biomass:** `data/external/imarpe_biomass_verified.csv` — 11 verified values
- **Feature matrix:** `data/external/paews_feature_matrix.csv` — 32 rows, model-ready

### Pipeline Scripts
```
python scripts/data_pipeline.py             # Download current SST/Chl from ERDDAP
python scripts/external_data_puller.py      # Update Niño, ONI, fishmeal prices
python scripts/predict_2026_s1.py           # Run 2026 S1 prediction (v2 + legacy)
python scripts/model_v2_audit.py            # 10-point data integrity audit
python scripts/model_v3_collinearity.py     # 4-model comparison, multicollinearity analysis
python scripts/model_diagnosis.py           # Hindcast validation, threshold analysis
python scripts/hindcast_validation.py       # Full LOO hindcast with metrics
python scripts/compute_2025_and_predict.py  # Add 2025 + retrain + bootstrap
python chl_migration.py                     # Process Copernicus Chl files
python composite_score.py                   # Generate risk scores (original pipeline)
python health_check.py                      # Validate (33 passes expected)
```

---

## 5. Key Discoveries & Dead Ends

### Critical Findings

1. **Sensor bias (Session 11):** VIIRS Chl reads +0.4 in log10 vs Copernicus. Flipped prediction from 7% to 50%. Script now uses Copernicus primary, VIIRS bias-corrected fallback.

2. **Multicollinearity (Session 13):** is_summer and bio_thresh_pct r=0.963. Dropping both improved ROC-AUC from 0.583 to 0.629. Model went from 5 to 3 features.

3. **SEVERE tier is the only signal (Session 12-13):** Historical calibration shows SEVERE (>0.70) = 100% disruption, 0 false positives. Everything below 0.70 is essentially random (~25-33% disruption rate regardless of tier).

4. **Niño 1+2 phase shift (Session 13, live):** Weekly data shows +1.0°C (Feb 23) vs −0.29 (Jan monthly). The protective La Niña factor in the model has flipped to a risk factor. Prediction will jump once monthly data updates.

### Dead Ends (Don't Repeat)
- **SLA (Sea Level Anomaly):** Only goes back to 2014. Coverage gap kills it.
- **Mask >50% or <30%:** Both drop performance. 40% is optimal.
- **VIIRS as Chl training source:** Sensor bias of +0.4 log10 vs Copernicus.
- **LLM-generated biomass numbers:** 65% error rate (Session 4).
- **Lagged landings feature:** Worsened performance, overfitting, backwards coefficients.
- **is_summer + bio_thresh_pct together:** r=0.963 collinear. Use neither.

---

## 6. Ground Truth (32 seasons)

File: `data/external/imarpe_ground_truth.csv`

### Recent seasons (web-verified Feb 28, 2026):

| Year | Season | Quota (MT) | Catch (MT) | % | Outcome |
|------|--------|-----------|-----------|---|---------|
| 2024 | S1 | 2,475,000 | 2,431,000 | 98 | NORMAL |
| 2024 | S2 | 2,510,000 | 2,380,000 | 95 | NORMAL |
| 2025 | S1 | 3,000,000 | 2,457,487 | 82 | NORMAL |
| 2025 | S2 | 1,630,000 | 1,596,013 | 98 | NORMAL |

---

## 7. Biomass Data

File: `data/external/imarpe_biomass_verified.csv` — 33 rows, **11 verified**

### Verified Values

| Year | S1 (MMT) | S2 (MMT) | Source |
|------|----------|----------|--------|
| 2018 | 11.21 | 8.78 | Castillo et al. (2021) Bol IMARPE 35(2) |
| 2019 | 8.82 | 8.38* | Castillo et al. (2021) Bol IMARPE 35(2) |
| 2020 | 11.05 | 9.52 | Castillo et al. (2021) Inf Inst Mar Peru 48(3) |
| 2021 | 12.03 | 8.03 | Castillo et al. (2023) Bol IMARPE 38(1) |
| 2022 | 7.13 | 4.68 | Castillo et al. (2024) Bol IMARPE 39(1) |
| 2025 | 10.93 | TBD | MarinTrust WF02 + SeafoodSource |

*2019 S2: integrity_flag=0 (IMARPE officials investigated for inflating biomass).

Still missing: 2010–2017 (all), 2023 (both), 2024 (both), 2025 S2.

---

## 8. Verification Protocol

**Rule 1:** Nothing enters CSVs without a source URL, PDF page number, or Ministerial Resolution number.
**Rule 2:** Never trust LLMs for specific numbers — 65% error rate proven.
**Rule 3:** Primary sources only: IMARPE PDFs, PRODUCE Resoluciones, NOAA/CPC data.
**Rule 4:** Cross-sensor data must be calibrated before comparison.
**Rule 5:** Web-sourced data gets verified against 2+ independent sources where possible.

---

## 9. Session History

| Session | Key Achievement |
|---------|----------------|
| 1–3 | Initial build, Copernicus migration, ground truth creation |
| 4 | Gemini hallucination discovery (65% biomass error), verification protocols |
| 5 | Mask tuning (50%→40%), SLA dead end, PR-AUC 0.705 |
| 6 | SST 2024 added (30 samples), PR-AUC baseline 0.682, biomass CSV created |
| 7 | Uploaded Castillo 2023 PDF — confirmed 2021 biomass |
| 8 | Searched for remaining biomass papers — only 2018-2021 digitized |
| 9 | Uploaded Castillo 2021 — confirmed 2020 biomass, English abstract error |
| 10 | Handoff v5, biomass integration script tested (works, needs more data) |
| 11 | 2026 S1 prediction: 0.33-0.50 BORDERLINE. Sensor bias caught (VIIRS +0.4). 2025 added (32 samples). |
| **12** | **Hindcast validation: ROC-AUC 0.583. Lagged landings rejected (overfitting). Severe target reframe (0.768 but N=4). Risk tier calibration: SEVERE 100%, middle tiers flat.** |
| **13** | **Multicollinearity fix: is_summer/bio_thresh r=0.963. 3-feature model wins (ROC-AUC 0.629). Data audit passed (0 errors). Production script v2 deployed. Interactive dashboard built. Niño 1+2 surge to +1.0 detected (not yet in model).** |

---

## 10. File Locations

```
C:\Users\josep\Documents\paews\
├── data/
│   ├── baseline/                   # Annual SST files (sst_2010.nc through sst_2025.nc)
│   ├── current/                    # Live data (sst_current.nc, chlorophyll_current.nc)
│   ├── copernicus_chl/             # 30 annual Chl NetCDF files (1997-2024)
│   ├── processed/                  # Climatologies (sst_climatology_v2.nc, chl_climatology_copernicus.nc)
│   ├── external/
│   │   ├── paews_feature_matrix.csv
│   │   ├── imarpe_ground_truth.csv
│   │   ├── imarpe_biomass_verified.csv
│   │   ├── chl_copernicus_full.nc
│   │   ├── nino_indices_monthly.csv
│   │   ├── oni_monthly.csv
│   │   └── CMO-Historical-Data-Monthly.xlsx
│   └── ground_truth.csv            # Original 30-row version (superseded)
├── scripts/
│   ├── predict_2026_s1.py          # PRODUCTION v2 (3-feature + legacy comparison)
│   ├── predict_2026_s1_v1_backup.py # Old 5-feature version
│   ├── model_v2_audit.py           # 10-point data integrity audit
│   ├── model_v3_collinearity.py    # 4-model multicollinearity comparison
│   ├── model_diagnosis.py          # Hindcast validation + threshold analysis
│   ├── hindcast_validation.py      # Full LOO hindcast
│   ├── compute_2025_and_predict.py # Add 2025 + retrain + bootstrap
│   ├── external_data_puller.py     # Download Niño, ONI, fishmeal
│   ├── data_pipeline.py            # Download current SST/Chl
│   └── index.html                  # Interactive dashboard (React)
├── chl_migration.py
├── composite_score.py
├── health_check.py
├── masks/coastal_mask_40pct.nc
└── PAEWS_HANDOFF_v7.md             # This file
```

---

## 11. Immediate Priority (March 2026)

1. **Check CPC for Feb 2026 Niño 1+2 monthly** — run `external_data_puller.py` daily until it updates
2. **Download Copernicus Jan 2026 monthly Chl** — should be available now on data.marine.copernicus.eu
3. **Re-run prediction** once either data source updates
4. **If Niño confirms +0.7 to +1.0 and Chl drops:** prediction likely moves to ELEVATED (0.50-0.57)
5. **If prediction crosses 0.70:** SEVERE tier = act. 100% historical accuracy.

---

## 12. Quick Start for New Chat

Paste this:

> I'm working on PAEWS — a satellite-based anchovy early warning system for Peru. Upload PAEWS_HANDOFF_v7.md for full context. Current state: Model v2 uses 3 features (sst_z, chl_z, nino12_t1) after fixing multicollinearity (dropped is_summer + bio_thresh_pct, r=0.963). LOO ROC-AUC 0.629. SEVERE tier (>0.70) has 100% accuracy, 0 false positives. Latest prediction 0.308 MODERATE but Niño 1+2 has surged from −0.29 to +1.0 (weekly, not yet in model). Once CPC publishes Feb monthly Niño and Copernicus Feb Chl becomes available, re-run prediction — likely moves to ELEVATED (~0.57). Priority: update data and re-predict.
