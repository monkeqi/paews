# PAEWS HANDOFF v9 — Session 18 Complete

**Date:** March 6, 2026
**Session:** 18 (GODAS pipeline, SLA evaluation, juvenile % data collection setup)
**Author:** Claude (Anthropic) with Josep
**Status:** GODAS tested and rejected. SLA tested with robustness checks — promising but not robust enough. Juvenile % data collection framework built, ready for data hunt. Model remains 3-feature production v2.
**Conda env:** `geosentinel` (NOT `paews` — that env does not exist)
**GitHub:** `github.com/monkeqi/paews`
**Local path:** `C:\Users\josep\Documents\paews\`

---

## 1. PRODUCTION MODEL STATE — UNCHANGED

### Model v2 (3-feature logistic regression) — CURRENT PRODUCTION

| Item | Value |
|---|---|
| Features | `sst_z`, `chl_z`, `nino12_t1` |
| Training samples | 32 (2010 S1 – 2025 S2) |
| Positives | 12 disrupted seasons |
| Validation | Leave-One-Out Cross-Validation |
| ROC-AUC | 0.629 (full 32), 0.690 (30-sample subset excluding 2025) |
| PR-AUC | 0.661 |
| SEVERE tier accuracy | 4/4 = 100% (zero false positives) |
| Class weights | Balanced |
| Scaler | StandardScaler (per LOO fold) |
| Solver | lbfgs, max_iter=1000 |

**Full-model coefficients (trained on all 32 samples):**

| Feature | Coefficient |
|---|---|
| sst_z | +0.390 |
| chl_z | −0.583 |
| nino12_t1 | +0.363 |
| intercept | −0.589 |

**Tier thresholds:**

| Tier | Probability range | Historical disruption rate |
|---|---|---|
| SEVERE | ≥ 0.70 | 100% (4/4) |
| ELEVATED | 0.50–0.69 | 25% (variable) |
| MODERATE | 0.20–0.49 | ~30% (statistically flat) |
| LOW | < 0.20 | 40% (too few samples) |

---

## 2. LIVE PREDICTION — 2026 S1

**Prediction: 0.398 MODERATE** (Updated March 4, 2026 with confirmed Feb Niño 1+2)

| Feature | Value | Source | Freshness |
|---|---|---|---|
| sst_z | +0.837 | sst_current.nc (Feb 15 snapshot) | 3 weeks old |
| chl_z | +0.166 | Copernicus Dec 2025 proxy | **STALE — 3 months old** |
| nino12_t1 | +0.920 | CPC Feb 2026 monthly | Fresh |

- Bootstrap 95% CI: [0.136, 0.738]
- Bootstrap IQR: [0.307, 0.500]
- Tier distribution: LOW 6.9%, MODERATE 68.0%, ELEVATED 21.4%, SEVERE 3.6%

**Scenario Analysis:**

| Scenario | Prob | Tier |
|---|---|---|
| Live (Feb Niño +0.92, Dec Chl proxy) | 0.398 | MODERATE |
| If Chl drops to −0.40σ | 0.596 | ELEVATED |
| If Chl drops to −0.80σ | 0.718 | SEVERE |
| Niño +1.50, Chl −0.40σ | 0.665 | ELEVATED |
| Niño +1.50, Chl −0.80σ | 0.788 | SEVERE |
| Worst case (2017-like) | 0.807 | SEVERE |

**ENFEN context:** El Niño Costero alert active since Feb 14 (Comunicado N° 03-2026). Weak magnitude, possibly moderate by July. Two Kelvin waves forecast Mar-May. NRT chlorophyll (Feb 17–Mar 3) showed healthy upwelling (log10 CHL = +0.201 vs climatology −0.330). Real risk inflection expected March-April.

---

## 3. SESSION 18 WORK COMPLETED

### 2.1 GODAS Thermocline Pipeline — BUILT AND REJECTED

**Script:** `scripts/godas_thermocline.py`
**What it does:** Downloads GODAS potential temperature via OPeNDAP from NOAA THREDDS (`psl.noaa.gov/thredds/dodsC/Datasets/godas/pottmp.YYYY.nc`), computes Z20 (depth of 20°C isotherm = 293.15K) for Peru coastal box, builds climatology, outputs z20_z feature.

**Bugs fixed during session:**
- Filename: `pottmp` not `potmp` (two t's)
- Temperature units: GODAS stores Kelvin, not Celsius. TARGET_TEMP = 293.15, not 20.0
- 2026 file doesn't exist (updated annually in January, latest = 2025)
- Empty cache handling from failed runs

**Results:** Pipeline works correctly. Z20 varies from ~5m (S2 seasons, shallow thermocline) to ~47m (2016 S1, deep thermocline during El Niño). BUT:

**LOO test results (test_godas_feature.py):**
- ROC-AUC: 0.663 → 0.638 (−0.025) — **WORSE**
- SEVERE: 5/5 (100%) — passes
- 2014 S1 (target case): 0.158 → 0.171 — barely moves (+0.013)
- z20_z coefficient: +0.318 (positive, physically correct)
- Collinearity: r=0.735 with nino12_t1, r=0.601 with sst_z

**Verdict: REJECTED.** Z20 in the Peru coastal box is too blunt. The box-averaged thermocline mostly tracks what SST and Niño already tell the model. The 2014 S1 Kelvin wave signal was either too offshore or too early to show up in a coastal box average for February.

**Portfolio value:** Keep the pipeline as a technical demonstration of subsurface oceanography skills. Don't add to production model.

### 2.2 SLA Feature Evaluation — PROMISING BUT NOT ROBUST

**Script:** `test_sla_feature.py` (ran in-session, not saved to scripts/)
**What it does:** Tests `sla_z` (already in feature matrix) as 4th feature.

**LOO results (30 samples, 2025 missing SLA):**
- ROC-AUC: 0.690 → 0.741 (+0.051) — best improvement seen from any feature
- SEVERE: 5/5 (100%)
- Coefficient: sla_z = −0.795 (negative — higher SLA reduces risk)
- Collinearity: r=0.805 with sst_z (HIGH)

**Robustness checks (test_sla_robustness.py):**

| Test | Result | Pass? |
|---|---|---|
| Bootstrap stability (500 resamples) | Helps 53% of time | ✗ (need >60%) |
| Permutation test (1000 shuffles) | p=0.115 | ✗ (need <0.10) |
| VIF (collinearity) | max=3.64 | ✓ (below 5) |
| Residualized SLA (remove SST/Niño overlap) | AUC +0.079, but SEVERE drops to 86% | ✓ (signal is real) |
| Paired LOO comparison | 17 better vs 10 worse | ✓ |
| S1 vs S2 split | S1: −0.074, S2: +0.037 | ✗ (hurts S1!) |

**Score: 3/5 checks passed — MODERATE evidence**

**Verdict: DO NOT ADD TO PRODUCTION.** The improvement is real but not robust with 30 samples. Bootstrap stability is barely better than chance, permutation test fails significance, and it hurts S1 predictions (which is what we need for 2026 S1). Revisit if sample size reaches 50+.

### 2.3 Juvenile % Data Collection Framework — BUILT

**Scripts created:**
- `scripts/build_juvenile_feature.py` — data entry template with 3/32 pre-filled
- `scripts/test_juvenile_feature.py` — LOO evaluation (same framework as others)

**Pre-filled values:**

| Season | Juv % | Source | Type |
|---|---|---|---|
| 2017 S2 | 96% | IMARPE cruise CR1709-11 | Pre-season |
| 2022 S1 | 4% | Produce statement via La República | Pre-season |
| 2023 S1 | 86% | IMARPE surveys, multiple news sources | Pre-season |

**Additional data found from web searches (during-season, LEAKAGE RISK — need pre-season values):**
- 2019 S2: 97.8% juveniles in IMARPE prospection Jan 2020 (during season closure)
- 2021 S2: 12.6% juveniles in number during season (from RM 00008-2022-PRODUCE)
- 2024 S1: ~36% during season (CooperAccion)
- 2024 S2: 14.5% early season (CooperAccion)
- 2025 S1: 24.4% accumulated during season (Oceana Peru)

**Where to find pre-season values:**

The best source is the PRODUCE Resolución Ministerial that **opens** each season (not closes it). Search for titles containing "Autorizan el inicio" or "Establecen la Primera/Segunda Temporada de Pesca" on:
- `https://busquedas.elperuano.pe/normaslegales/` — search "anchoveta temporada"
- `https://www.gob.pe/institucion/produce/normas-legales` — search "anchoveta temporada"

**Note:** Josep had trouble accessing both sites at end of session. May need to try different browser, VPN, or search El Peruano by RM number directly.

**Alternative:** Search IMARPE repositorio for "Crucero de Evaluación Hidroacústica" reports by year. The pre-season cruise report always has juvenile %.

**Priority order for collection:** Disrupted seasons first (2014 S1, 2015 S2, 2016 S1, 2022 S2), then everything else.

### 2.4 Data Refresh Script

**Script:** `scripts/data_refresh.ps1`
PowerShell reference with all commands for refreshing SST, Chl, Niño, fishmeal, weekly SST.

---

## 4. CONDA ENVIRONMENT

**Name:** `geosentinel` (NOT `paews` — that env doesn't exist)

```powershell
conda activate geosentinel
```

---

## 5. NEW FILES CREATED THIS SESSION

| File | Location | Purpose |
|---|---|---|
| `godas_thermocline.py` | `scripts/` | GODAS Z20 pipeline (works, feature rejected) |
| `test_godas_feature.py` | `scripts/` | LOO test for Z20 (failed) |
| `test_sla_robustness.py` | `scripts/` | 5-test robustness battery for SLA |
| `build_juvenile_feature.py` | `scripts/` | Data entry + merge for juvenile % |
| `test_juvenile_feature.py` | `scripts/` | LOO test for juvenile % (ready when data collected) |
| `data_refresh.ps1` | `scripts/` | PowerShell commands for all data refreshes |
| `godas_z20_timeseries.csv` | `data/external/` | Cached Z20 monthly means 2003-2025 |
| `godas_z20_climatology.csv` | `data/processed/` | Z20 monthly climatology 2003-2022 |

**Feature matrix updated:** `z20_z` column added (32 values). Can be safely ignored by production model.

---

## 6. KEY TECHNICAL FINDINGS

### GODAS learnings:
- GODAS files are `pottmp.YYYY.nc` (not `potmp`)
- Temperature is in **Kelvin** (293.15K = 20°C)
- Peru coastal box (5-15°S, 85-76°W) has very shallow thermocline in S2 (5m floor)
- Z20 in this box correlates r=0.735 with Niño — mostly redundant
- The 2014 S1 Kelvin wave didn't show up in box-averaged Z20 for February
- Possible improvement: try equatorial slice Z20 at longer lead time (not attempted)

### SLA learnings:
- sla_z carries genuine information beyond SST/Niño (residualized test confirmed)
- But r=0.805 collinearity with sst_z makes it unstable with 30 samples
- The model learns to use SST-minus-SLA as a signal (warm surface + low SLA = danger)
- SLA hurts S1 predictions specifically — bad for our use case
- Revisit with more samples or as part of a 5+ feature model

### Model ceiling assessment:
- Remote sensing alone probably maxes out around ROC-AUC 0.65-0.70
- The three persistent misses (2014 S1, 2022 S2, 2011 S2) are biological, not oceanographic
- Juvenile % is the highest-value feature not yet tested
- The path to 0.70+ AUC goes through biological data

---

## 7. NEXT SESSION PRIORITIES

1. **Collect juvenile % data** — find pre-season values from PRODUCE opening resolutions or IMARPE cruise reports. Need at least 20 of 32 for meaningful LOO test. Priority: disrupted seasons first.

2. **Ship current model** — the 3-feature model works. ENFEN window is still open. README + GitHub push + outreach are higher ROI than more feature experiments.

3. **Fix health_check.py** — 3 stale assertions (30→32, 2024→2025). 5-minute job.

4. **Push to GitHub** — sessions 16-18 work isn't public.

---

## 8. THINGS NOT TO FORGET

1. **Conda env is `geosentinel`**, not `paews`.

2. **GODAS temperature is in Kelvin.** 293.15K = 20°C. This cost us one full failed run.

3. **SLA looks good on paper (+0.051 AUC) but fails robustness.** Don't be tempted to add it. The bootstrap says it's a coin flip.

4. **The resolutions that OPEN seasons have the pre-season juvenile %.** The ones that CLOSE seasons have during-season catch juvenile % (leakage).

5. **2025 S1 and S2 have no SLA data.** Would need to be computed before SLA could ever go to production.

6. **The feature matrix now has `z20_z` column.** This won't affect the 3-feature production model but the column exists.

7. **GODAS data only goes through 2025** (file updated Jan 15, 2026). For 2026 S1, Dec 2025 Z20 would be the proxy — but moot since we rejected the feature.

---

## 9. ALL PAEWS FILES (updated inventory)

### Scripts (all under `C:\Users\josep\Documents\paews\scripts\`):

| Script | Purpose | Status |
|---|---|---|
| `predict_2026_s1.py` | Production prediction (v2, 3-feature) | Works |
| `scenario_analysis.py` | Scenario sweep | Works |
| `external_data_puller.py` | Pull Niño, ONI, fishmeal | Works |
| `chl_migration.py` | Copernicus Chl processing | Works |
| `data_pipeline.py` | SST processing | Works |
| `health_check.py` | Validation (3 assertions need fixing) | Needs fix |
| `model_v2_audit.py` | 10-point data audit | Works |
| `composite_score.py` | Legacy composite score | Works |
| `godas_thermocline.py` | **NEW** GODAS Z20 pipeline | Works (feature rejected) |
| `test_godas_feature.py` | **NEW** LOO test for Z20 | Works (failed) |
| `test_sla_robustness.py` | **NEW** SLA robustness battery | Works (marginal) |
| `build_juvenile_feature.py` | **NEW** Juvenile % data entry | Needs data |
| `test_juvenile_feature.py` | **NEW** LOO test for juvenile % | Waiting for data |
| `test_prev_catch_feature.py` | Feature experiment (rejected) | Done |
| `test_prev_feature_variants.py` | 8-variant deep dive (rejected) | Done |
| `data_refresh.ps1` | **NEW** All refresh commands | Reference |

### Dashboard:
| File | Lines | Status |
|---|---|---|
| `paews_performance_dashboard.jsx` | 625 | Updated session 17 |

### GitHub: `github.com/monkeqi/paews` — needs push with sessions 16-18 work.

---

## 10. IMARPE PDFs — DOWNLOADED, READY FOR EXTRACTION

Josep downloaded **63 PDF files** (each ~1MB) from `repositorio.imarpe.gob.pe`, searching for "crucero evaluación hidroacústica anchoveta" type reports. These are the pre-season acoustic cruise reports that contain:
- **Juvenile percentage** (porcentaje de juveniles / fracción juvenil)
- **Biomass estimates** (in MMT)
- **Size/talla distribution** (range, moda)

**These should be saved to:** `C:\Users\josep\Documents\paews\data\external\imarpe_reports\`

**Next session task:** Upload a few of these PDFs (start with disrupted years: 2014, 2016, 2022, 2023) and extract juvenile % values to fill `build_juvenile_feature.py`. Also extract biomass values to fill `imarpe_biomass_verified.csv` gaps (21 of 32 empty).

**Data entry workflow:**
1. Read PDF, find "porcentaje de juveniles" in the survey results
2. Update the value in `build_juvenile_feature.py` 
3. Once 20+ values filled, run: `python scripts/build_juvenile_feature.py` then `python scripts/test_juvenile_feature.py`

**CRITICAL LEAKAGE RULE:** Use the PRE-SEASON survey juvenile % (from the IMARPE cruise BEFORE the season opens), NOT the during-season catch juvenile %. The opening resolutions cite the pre-season number. The closing resolutions cite during-season numbers (leakage).

---

## 11. MODEL BLIND SPOTS (unchanged from v8)

| Season | LOO Prob | Actual | Cause | What would fix it |
|---|---|---|---|---|
| **2014 S1** | 0.097 | DISRUPTED | Subsurface Kelvin wave. SST cool, Chl healthy, Niño negative. GODAS Z20 tested this session — didn't help. | Unknown. Possibly equatorial Z20 at longer lead, or just accept as unfixable by remote sensing |
| **2022 S2** | 0.163 | DISRUPTED | 70% juveniles in catch. SST cold, La Niña. IMARPE closed early. | **Juvenile % from IMARPE pre-season survey** |
| **2011 S2** | 0.095 | REDUCED | Stock depletion. Ocean fine. Biological. | **IMARPE biomass data** |

---

## 12. FEATURES TESTED AND REJECTED (complete list)

| Feature | Sessions | Result | Reason |
|---|---|---|---|
| is_summer | 12-13 | Dropped | r=0.963 with bio_thresh_pct |
| bio_thresh_pct | 12-13 | Dropped | Multicollinearity |
| Previous season catch % (8 variants) | 17 | All rejected | Anchoveta recover too fast |
| GODAS Z20 (z20_z) | **18** | Rejected | AUC −0.025, 2014 S1 didn't move |
| SLA (sla_z) | **18** | Not added | +0.051 AUC but fails bootstrap (53%), permutation (p=0.115), hurts S1 |

---

## 13. OUTREACH STATUS

Comprehensive outreach document exists: `PAEWS_OUTREACH_LIST.md`

Five tiers: Norwegian salmon companies, industry certification, fisheries science, space industry, academic/community. Includes LinkedIn templates and IMARPE email in Spanish.

**Timing:** Window is excellent. ENFEN confirmed El Niño Costero. Season opens ~6 weeks. Ship the README and push to GitHub before doing more feature work.

---

## 14. PRIOR SESSION SUMMARY (for context)

- **Sessions 1-6:** Built data pipeline (NOAA OISST, MODIS Chl, conda setup, GitHub repo)
- **Sessions 7-9:** IMARPE biomass data collection, ground truth verification
- **Session 10-11:** MODIS→Copernicus chlorophyll migration, coastal productivity mask
- **Session 12-13:** Model pruning (7→5→3 features), v2 model, React dashboard, scenario analysis
- **Session 14-15:** Data integrity audit, caught LLM-hallucinated biomass data (bio_thresh_pct dropped)
- **Session 16:** Fresh data pull (Feb 2026 Niño), prediction update 0.308→0.398, dashboard update
- **Session 17:** Full verification, scenario corrections, feature experiments (8 prev-season variants rejected), outreach list, NRT chlorophyll check
- **Session 18:** GODAS pipeline (rejected), SLA robustness (not robust), juvenile % framework built, 63 IMARPE PDFs downloaded

---

## 15. WHAT THE NEXT SESSION SHOULD DO

1. **Extract juvenile % from the IMARPE PDFs** — this is the #1 priority. Upload 3-4 PDFs from disrupted years, find "porcentaje de juveniles", fill build_juvenile_feature.py
2. **Also extract biomass estimates** from same PDFs to fill imarpe_biomass_verified.csv gaps
3. **Run test_juvenile_feature.py** once 20+ values are filled
4. **If juvenile % passes:** update production model to v3 (4-feature), rerun prediction, update dashboard
5. **If juvenile % fails:** ship the 3-feature model as-is — write README, push GitHub, start outreach
6. **Either way:** fix health_check.py (5 min), push to GitHub
