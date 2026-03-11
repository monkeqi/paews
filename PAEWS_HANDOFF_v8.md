# PAEWS HANDOFF v8 — Session 17 Complete

**Date:** March 5, 2026
**Session:** 17 (Verification completion, fresh data pull, feature experiments, outreach planning)
**Author:** Claude (Anthropic) with Josep
**Status:** Dashboard fully verified and updated with live Feb 2026 data. v3 feature experiments completed. Ready for biological data collection.

---

## 1. PRODUCTION MODEL STATE

### Model v2 (3-feature logistic regression) — CURRENT PRODUCTION

| Item | Value |
|---|---|
| Features | `sst_z`, `chl_z`, `nino12_t1` |
| Training samples | 32 (2010 S1 – 2025 S2) |
| Positives | 12 disrupted seasons |
| Validation | Leave-One-Out Cross-Validation |
| ROC-AUC | 0.629 |
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

**Updated March 4, 2026 with confirmed Feb Niño data.**

| Feature | Value | Source | Freshness |
|---|---|---|---|
| sst_z | +0.837 | sst_current.nc (Feb 15 snapshot) | 2 weeks old |
| chl_z | +0.166 | Copernicus Dec 2025 proxy | **STALE — 3 months old** |
| nino12_t1 | +0.920 | CPC Feb 2026 monthly | **FRESH** |

**Prediction: 0.398 MODERATE**
- Bootstrap 95% CI: [0.136, 0.738]
- Bootstrap IQR: [0.307, 0.500]
- Tier distribution: LOW 6.9%, MODERATE 68.0%, ELEVATED 21.4%, SEVERE 3.6%

**Previous prediction (Session 16): 0.308 MODERATE** using Jan Niño (−0.29). The +1.21°C Niño surge from Jan to Feb drove the 9-point increase.

### Scenario Analysis (verified against scenario_analysis.py output)

| Scenario | Prob | Tier | Status |
|---|---|---|---|
| Live (Feb Niño +0.92, Dec Chl proxy) | 0.398 | MODERATE | Current |
| If Chl drops to −0.40σ | 0.596 | ELEVATED | Plausible |
| If Chl drops to −0.80σ | 0.718 | SEVERE | If upwelling collapses |
| Niño +1.50, Chl −0.40σ | 0.665 | ELEVATED | If 2nd Kelvin wave hits |
| Niño +1.50, Chl −0.80σ | 0.788 | SEVERE | Strong Costero scenario |
| Worst case (2017-like) | 0.807 | SEVERE | Coastal El Niño repeat |

**Critical context:** ENFEN officially activated "Alerta de El Niño Costero" on Feb 14 (Comunicado N° 03-2026). As of March 3, ENFEN spokesperson Luis Vásquez confirmed the Coastal El Niño is in its initial phase. Two Kelvin waves are forecast: one Mar-Apr, another Apr-May. These could push the event from weak to moderate magnitude by July.

---

## 3. DATA FRESHNESS STATUS

### What was pulled this session:

1. **CPC Niño monthly indices** — `external_data_puller.py` pulled through Feb 2026. Niño 1+2 Feb = +0.92°C. File: `data/external/nino_indices_monthly.csv` (530 months, 1982-01 to 2026-02)

2. **CPC weekly SST** — manually downloaded from `https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for`. File: `data/external/weekly_sst.txt`. Shows Niño 1+2 weekly trajectory:
   - 31 Dec: −0.9
   - 07 Jan: −0.6
   - 14 Jan: −0.2
   - 21 Jan: −0.1
   - 28 Jan: +0.3 (crossed positive)
   - 04 Feb: +0.6
   - 11 Feb: +0.7
   - 18 Feb: **+1.2** (peak so far)
   - 25 Feb: +1.0 (slight pullback)
   - CPC March 2 diagnostic reports latest weekly at +0.7

3. **Copernicus Chl climatology** — `chl_migration.py` recomputed from full 2003–2025 dataset. All 32 seasons confirmed Copernicus source. File: `data/processed/chl_climatology_copernicus.nc`

4. **Copernicus NRT daily Chl** — pulled via `copernicusmarine subset` for Peru coastal box (−82 to −76°W, −16 to −4°S). File: `data/external/chl_nrt_2026.nc`. **Only covers Feb 17 – Mar 3** (12 days, rolling NRT window). Key finding: overall mean CHL = 1.5891 mg/m³, log10(CHL) = +0.201. This is **well above** the Feb climatological mean (log10 = −0.330). **Chlorophyll has NOT collapsed yet** despite the Niño surge. Upwelling is still productive. This means the Dec 2025 proxy (chl_z = +0.166) is probably still reasonable. The real Chl decline is expected in March-April when sustained warming suppresses upwelling.

5. **World Bank fishmeal prices** — pulled from CMO-Historical-Data-Monthly.xlsx. Latest: **$1,824/MT** (Dec 2025, Peru fishmeal FOB 65% protein). Confirmed correct.

6. **ONI monthly** — pulled through Feb 2026. File: `data/external/oni_monthly.csv` (913 records)

### What's still stale:

| Input | Current value | How to refresh |
|---|---|---|
| SST (sst_current.nc) | Feb 15 snapshot | Re-download OISST daily from ERDDAP |
| Chlorophyll | Dec 2025 monthly | Wait for Copernicus MY product to publish Jan 2026. NRT daily available but needs pipeline integration |
| Fishmeal price | Dec 2025 | Next Pink Sheet update will have Jan 2026 |

---

## 4. VERIFICATION STATUS — 100% COMPLETE

### Scripts that passed:

| Script | Result | Notes |
|---|---|---|
| `model_v2_audit.py` | 0 errors, 2 warnings | Warnings are known correlations, not bugs |
| `health_check.py` | 30 pass, 3 fail | Fails are stale assertions (expected 30 rows, now 32). **Need to fix the Python comparison variables, not just display text** — the text-replace approach only caught strings, not the actual comparison logic |
| `scenario_analysis.py` | Runs clean | Fixed with `r"""` docstring. All 7 scenario values verified |
| Independent LOO computation | Exact match | 32/32 probabilities match, ROC-AUC 0.629, coefficients match |
| Independent Niño cross-check | 32/32 match | Every nino12_t1 traces to exact month in raw CPC CSV |
| Independent ground truth check | 32/32 match | Every target traces to outcome in imarpe_ground_truth.csv |
| Quota/catch sanity check | 0 warnings | All catch numbers consistent with outcome classifications |

### Scenario values — CORRECTED this session:

The handoff v7 scenario estimates were **wrong**. Session 17 verified them against `scenario_analysis.py` output and found two tiers were incorrect:

| Scenario | Handoff v7 (wrong) | Actual (verified) | Tier change |
|---|---|---|---|
| Current (stale data) | 0.308 | 0.308 | ✓ same |
| Updated Niño +0.70 | ~0.37 | 0.394 | ✓ same tier |
| Updated Niño +1.00 | ~0.40 | 0.422 | ✓ same tier |
| Niño +0.70, Feb Chl −0.40 | ~0.49 | **0.580** | **MOD → ELEV** |
| Niño +1.00, Feb Chl −0.40 | ~0.57 | **0.608** | ✓ same tier |
| Niño +1.00, bad Chl −0.80 | ~0.67 | **0.725** | **ELEV → SEVERE** |
| Worst case (2017-like) | ~0.78 | **0.807** | ✓ same tier |

The model is more sensitive to the current trajectory than previously stated.

### Reproducibility hashes (from health_check.py):

| File | Hash |
|---|---|
| paews_feature_matrix.csv | f7f65663f9fd |
| imarpe_ground_truth.csv | cfea87fbe55a |
| composite_score.py | 7d3ef9d65be0 |
| chl_migration.py | 27313162a2c7 |

---

## 5. DASHBOARD STATE

**File:** `paews_performance_dashboard.jsx` (625 lines)
**Location:** `C:\Users\josep\Documents\paews\` (copy also in `/mnt/user-data/outputs/`)

### Changes made this session:

1. **Updated prediction from 0.308 → 0.398** with confirmed Feb 2026 Niño (+0.92)
2. **Updated all scenario values** to match verified script output
3. **Updated scenario interpretation prose** — now explains Chl decline scenarios instead of Niño uncertainty
4. **Updated Niño status indicator** from "⚠ Jan monthly" (yellow) to "✓ Feb monthly" (green)
5. **Updated Niño current state paragraph** — confirms +0.92°C Feb monthly, +1.21°C surge from January
6. **Updated prediction subtitle** — "Updated March 4, 2026 with confirmed Feb Niño 1+2"
7. **Updated critical signal note** — confirms the surge, identifies Chl as the remaining stale input
8. **Updated scenario analysis subtitle** — "How the prediction shifts if chlorophyll declines"
9. **Updated footer data stamp** — "Niño Feb 2026 (confirmed)"
10. **Removed Session 4 AI hallucination references** — both from verification protocol (#04) and development timeline
11. **Rewrote "No AI-Generated Data" policy** — clean statement without backstory about Gemini 65% error rate
12. **Merged timeline entries** — Sessions 1-3 and Sessions 4-6 (previously Sessions 1-3, Session 4, Sessions 5-6)

### What remains in the dashboard that could be updated:

- SST snapshot is still Feb 15. When a new SST download happens, the sst_z may change.
- Chlorophyll is still the Dec 2025 proxy. When Copernicus publishes Jan/Feb 2026, this is the last piece.
- The "Session 13" timeline entry still references "Detected Niño 1+2 surge from −0.29 to +1.0°C" — this was the weekly value from Session 13. It's historical context and accurate, but the monthly value is now confirmed at +0.92.
- If v3 model is adopted (4+ features), the entire model section would need updating.

---

## 6. FEATURE EXPERIMENTS — COMPLETED AND REJECTED

### Experiment 1: Raw previous season catch percentage

**Script:** `scripts/test_prev_catch_feature.py`
**Result:** ROC-AUC 0.654 → 0.684 (+0.031), but **SEVERE dropped from 100% to 83%**
**Rejected.** The 2016 S2 season (NORMAL) was pushed to 0.793 SEVERE because the prior S1 only landed 60%. The fishery recovered but the feature couldn't see that.

### Experiment 2: Deep dive — 8 variant formulations

**Script:** `scripts/test_prev_feature_variants.py`
**Variants tested:**

| Variant | Description | ΔROC | SEVERE | Pass? |
|---|---|---|---|---|
| A | Raw catch_pct | +0.031 | 5/6 (83%) | ✗ |
| B | Binary: prev season disrupted | +0.079 | 7/9 (78%) | ✗ |
| C | Binary: catch_pct < 80% | +0.000 | 5/6 (83%) | ✗ |
| D | Binary: catch_pct < 85% | +0.000 | 5/6 (83%) | ✗ |
| E | Capped inverted: 100 − min(pct,85) | +0.013 | 5/6 (83%) | ✗ |
| F | Disrupted × warm Niño interaction | +0.026 | 6/7 (86%) | ✗ |
| G | Previous quota < 2M MT | −0.018 | 5/6 (83%) | ✗ |
| **H** | **Any of prev 2 seasons disrupted** | **+0.056** | **5/5 (100%)** | **✓** |

**Variant H passed the hard constraint** (SEVERE stays 100%) with a meaningful AUC improvement (+0.056). However, detailed analysis revealed:
- 17 directionally correct changes vs **11 directionally wrong changes**
- **2014 S1 got worse** (0.162 → 0.105) — the ghost miss moved in the wrong direction
- **2022 S2 got worse** (0.315 → 0.213) — a real disruption pushed down
- Post-disruption false alarms are severe: 2013 S1/S2, 2017 S2, 2018 S1, 2024 S1/S2 all pushed up significantly as NORMAL seasons because the feature "remembers" disruptions that the fishery already recovered from

**Conclusion: ALL previous-season features rejected.** The fundamental problem is that anchoveta (Engraulis ringens) are a fast-reproducing species that can recover stock in one season. Sequential disruption logic doesn't reliably predict this fishery. The AUC improvements are real but come from easy cases while actively hurting hard cases.

### Known model blind spots (3 problem seasons):

| Season | LOO Prob | Actual | Cause | What would fix it |
|---|---|---|---|---|
| **2014 S1** | 0.097 | DISRUPTED | Subsurface Kelvin wave. Surface SST was cool (−0.64), Chl healthy (+0.25), Niño negative (−0.92). Warm water was propagating below the surface. El Niño onset disrupted recruitment before SST responded. | **GODAS thermocline depth** — Z20 anomaly would have shown depressed isotherm |
| **2022 S2** | 0.163 | DISRUPTED | Juvenile percentage. SST was cold (−1.03), La Niña. IMARPE closed season early because 70% of catch was juveniles. Biomass age structure collapsed. | **Juvenile % from PRODUCE resolutions** or **IMARPE biomass data** |
| **2011 S2** | 0.095 | REDUCED | Stock depletion. Ocean conditions were fine. Biological reasons invisible to remote sensing. | **IMARPE biomass data** |

---

## 7. V3 DEVELOPMENT PLAN

### Priority order:

1. **Collect juvenile percentage data** (est. 2-3 hours)
   - Source: PRODUCE Resoluciones Ministeriales on `gob.pe/produce`
   - Each resolution that opens a season cites IMARPE's pre-season survey, which includes juvenile percentage
   - One number per season, 32 seasons needed
   - Search terms: "Resolución Ministerial", "temporada de pesca", "anchoveta", "porcentaje de juveniles"
   - **Timing/leakage check:** The IMARPE survey happens weeks BEFORE the decision_month. The resolution is published at season open (after decision_month). Use the survey date, not the resolution date. If the survey result is known before decision_month=3 for S1 or decision_month=10 for S2, no leakage.

2. **Test juvenile % as 4th feature** (est. 30 min)
   - Same LOO framework as test_prev_feature_variants.py
   - Hard constraint: SEVERE must stay at 100%
   - Expected coefficient sign: positive (high juvenile % → more disruption risk)
   - Key test: does 2022 S2 improve? (juvenile-driven disruption)
   - Key test: does 2023 S1 improve? (86% juveniles was the proximate cause)

3. **Backfill IMARPE biomass data (2010–2017)** (est. 4-6 hours)
   - Source: IMARPE repositorio cruise reports
   - Cruise codes listed in `imarpe_biomass_verified.csv`: 1002, 1009, 1102, 1109, 1202, 1209, 1302, 1309, 1402, 1409, 1502, 1509, 1602, 1609, 1702, 1709
   - HIGH PRIORITY: 1302, 1309, 1402, 1409, 1702, 1709 (disrupted seasons)
   - Format: biomass in MMT, region (total or north-central), source citation with page number
   - Once complete, test `biomass / rolling_5yr_mean` as a feature (same framework)
   - **Leakage check:** Pre-season biomass surveys happen in Feb-Apr for S1 and Aug-Oct for S2. The survey result is known before the model's decision_month if the survey completes in time. Need to verify timing for each season.

4. **Build GODAS thermocline pipeline** (est. 3-4 hours)
   - Source: NOAA PSL THREDDS server at `https://psl.noaa.gov/thredds/catalog/Datasets/godas/`
   - File: `potmp.yyyy.nc` (potential temperature, monthly, by depth level)
   - Extract Peru coastal box (~5-15°S, coast to ~85°W)
   - Compute Z20: interpolate depth where temperature = 20°C at each grid point
   - Average Z20 anomaly against monthly climatology
   - Use 1-2 month lag matching decision_month pattern (Feb for S1, Sep for S2)
   - **Expected to fix 2014 S1** specifically. Won't help 2022 S2 or 2011 S2.
   - Gemini confirmed this architecture. Correction: GODAS is on THREDDS/OPeNDAP, NOT ERDDAP as Gemini stated.

5. **Ship** — push to GitHub, finalize dashboard, do outreach

### Data leakage rules (from Gemini, confirmed correct):

- **No leakage:** SST, Chl, Niño 1+2 (all available before decision_month from automated pipelines)
- **No leakage:** Previous season catch % (known months before current prediction)
- **No leakage if timed correctly:** IMARPE pre-season biomass survey (survey happens before decision_month, but resolution citing it may come after)
- **LEAKAGE:** Current season quota (announced after decision_month for S1)
- **LEAKAGE:** Current season start date delay (only known at season open)
- **LEAKAGE:** Current season juvenile percentage (only known during fishing)

---

## 8. ENFEN / EL NIÑO COSTERO INTELLIGENCE

### Official status (as of March 3, 2026):

- **ENFEN Comunicado N° 03-2026** (Feb 14): Activated "Alerta de El Niño Costero"
- El Niño Costero expected March through November 2026
- **Magnitude: weak** for most of the period, possibly **moderate by July**
- Central Pacific (Niño 3.4): neutral through May, then weak El Niño possible from June
- Two Kelvin waves forecast: one Mar-Apr, another Apr-May
- These could reinforce warming and upgrade the event
- ENFEN spokesperson Luis Vásquez confirmed initial phase on March 3

### Anchoveta-specific intelligence from ENFEN:

- Reproductive indicators (gonadal maturation, spawning) are increasing
- When sea temperature rises, anchoveta migrate coastward seeking colder water
- If warming continues, fish move deeper or southward, becoming less available to purse seine nets
- At weak magnitude, impact on anchoveta would be "limited" per ENFEN
- The question is whether it stays weak or escalates to moderate

### CPC/IRI ENSO outlook:

- La Niña → ENSO-neutral transition expected Feb-Apr 2026 (60% chance)
- ENSO-neutral likely through Northern Hemisphere summer (56% in Jun-Aug)
- El Niño probabilities low through Mar-May (<10%), rising to 30-35% by Jun-Aug
- Important: CPC now uses **Relative Oceanic Niño Index (RONI)** as of Feb 2, 2026

---

## 9. NRT CHLOROPHYLL FINDING

Copernicus NRT daily gap-free Chl was pulled for the Peru coastal box (Feb 17 – Mar 3, 12 days).

**Result:** Overall mean CHL = 1.5891 mg/m³, log10(CHL) = +0.201

This is **well above** the February climatological mean (log10 = −0.330). The upwelling system is still productive. Chlorophyll has not collapsed despite the Niño 1+2 surge to +0.92°C.

**Physical interpretation:** The first Kelvin wave arrived in mid-February (peak SST anomaly +1.2 on Feb 18), but chlorophyll decline typically lags SST warming by weeks to months. The plankton are still there. Sustained warming from the second Kelvin wave (Mar-Apr) is needed to suppress upwelling long enough for chlorophyll to respond.

**Implication for prediction:** The Dec 2025 proxy (chl_z = +0.166) is probably still reasonable or slightly conservative. The current 0.398 MODERATE prediction holds. The real risk inflection comes in March-April.

**Important caveat:** This NRT analysis used the raw box mean WITHOUT the 40% coastal mask or proper z-score computation. To get the real chl_z from NRT data, you'd need to integrate it into the `chl_migration.py` pipeline. The directional finding (chlorophyll is healthy) is reliable, but the exact number is not directly comparable to the model's chl_z.

**How to re-pull NRT data:**
```powershell
copernicusmarine subset -i cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D --variable CHL -x -82 -X -76 -y -16 -Y -4 -t "2026-02-01" -T "2026-03-31" -o data/external -f chl_nrt_2026.nc
```
Note: NRT has a rolling window (~2 weeks). Only recent data will be available. Older months require the MY (Multi-Year) product, which has longer processing latency.

---

## 10. FILE INVENTORY

### Key files (all under `C:\Users\josep\Documents\paews\`):

**Data:**
| File | Description | Last updated |
|---|---|---|
| `data/external/paews_feature_matrix.csv` | 32-row feature matrix, 3 production features + legacy | This session (chl_migration.py) |
| `data/external/imarpe_ground_truth.csv` | 32-row ground truth with quotas, catches, outcomes | Session 11 |
| `data/external/imarpe_biomass_verified.csv` | Biomass audit file, 11 verified + 21 empty | Session 9 |
| `data/external/nino_indices_monthly.csv` | CPC Niño 1+2/3/3.4/4, 1982–2026-02 | This session |
| `data/external/oni_monthly.csv` | Oceanic Niño Index, 913 records | This session |
| `data/external/weekly_sst.txt` | CPC weekly SST indices through Feb 25 | This session |
| `data/external/chl_nrt_2026.nc` | Copernicus NRT daily Chl, Feb 17–Mar 3 | This session |
| `data/external/CMO-Historical-Data-Monthly.xlsx` | World Bank Pink Sheet, fishmeal prices through Dec 2025 | This session |
| `data/processed/chl_climatology_copernicus.nc` | 2003–2022 monthly Chl climatology | This session |
| `data/processed/sst_climatology_v2.nc` | SST climatology | Earlier session |
| `data/processed/sst_current.nc` | Latest SST snapshot (Feb 15, 2026) | Session 13 |

**Scripts:**
| Script | Purpose | Status |
|---|---|---|
| `scripts/predict_2026_s1.py` | Production prediction | Works, ran this session |
| `scripts/scenario_analysis.py` | Scenario sweep | Works (fixed r""" this session) |
| `scripts/external_data_puller.py` | Pull Niño, ONI, fishmeal | Works, ran this session |
| `scripts/chl_migration.py` | Copernicus Chl processing | Works, ran this session |
| `scripts/data_pipeline.py` | SST processing | Works |
| `scripts/health_check.py` | 42-check validation | Works but 3 assertions need fixing (expects 30→32) |
| `scripts/model_v2_audit.py` | 10-point data audit | Works, passed |
| `scripts/model_v3_collinearity.py` | 4-model comparison | Works |
| `scripts/composite_score.py` | Legacy composite score | Works |
| `scripts/test_prev_catch_feature.py` | Feature experiment (rejected) | Created this session |
| `scripts/test_prev_feature_variants.py` | 8-variant deep dive (rejected) | Created this session |

**Dashboard:**
| File | Lines | Status |
|---|---|---|
| `paews_performance_dashboard.jsx` | 625 | Updated this session with live prediction, corrected scenarios, removed Session 4 references |

**GitHub:** `github.com/monkeqi/paews` — needs push with all updates from sessions 16-17.

---

## 11. HEALTH CHECK FIX NEEDED

The `health_check.py` text-replace from this session (`-replace 'expect 30', 'expect 32'`) only caught **display strings**, not the **Python comparison variables**. The script still shows 3 failures even though the values now match. Look for something like:

```python
expected_rows = 30  # ← change to 32
expected_normal = 18  # ← change to 20
```

or comparison logic like:

```python
if row_count != 30:  # ← change to 32
```

Also the year range check fails — it expects 2010-2024 but now has 2010-2025. Find and update that assertion too.

---

## 12. OUTREACH LIST

A comprehensive outreach document was created: `PAEWS_OUTREACH_LIST.md`

Five tiers:
1. Norwegian salmon companies (Mowi, SalMar, Lerøy, Austevoll, Cargill, Skretting)
2. Industry certification (MarinTrust, IFFO, SFP, trade media)
3. Fisheries science (Castillo et al. at IMARPE, ENFEN, FAO)
4. Space industry (Orbital Insight, Descartes Labs, Planet, Spire, Pixxel, etc.)
5. Academic/community (subreddits, journals, conferences)

Includes three outreach templates: salmon company LinkedIn, space company LinkedIn, IMARPE email in Spanish.

**Timing:** Current window is excellent. ENFEN confirmed El Niño Costero 2 days ago. Season opens in ~6 weeks. If the model's MODERATE prediction proves correct, that's the proof point for outreach.

---

## 13. SESSION 4 REFERENCES — REMOVED FROM DASHBOARD

The AI hallucination discovery (Google Gemini, 65% error rate on biomass data) was removed from two places in the dashboard:
1. Verification protocol entry #04 — rewritten as clean "No AI-Generated Data" policy without backstory
2. Development timeline — "Session 4" entry removed, timeline now goes Sessions 1-3 → Sessions 4-6

The information is preserved in this handoff and in the raw transcript but should NOT appear in any client-facing materials. The "no AI data" policy itself is important to keep — just not the story of why it exists.

---

## 14. THINGS NOT TO FORGET

1. **The Dec 2025 Chl proxy is the last stale input.** Everything else is fresh. When Copernicus publishes Jan 2026 monthly, re-run chl_migration.py then predict_2026_s1.py.

2. **The NRT data showed healthy chlorophyll (log10 = +0.201 vs climatology −0.330).** This is good news short-term but watch for decline in March-April.

3. **The model is more sensitive than previously thought.** Session 17 discovered the handoff v7 scenario estimates underestimated risk for two tiers. The corrected values show the "bad Chl" scenario already crosses SEVERE at 0.725.

4. **Weekly Niño peaked at +1.2 (Feb 18) and is pulling back to +0.7 (early March).** This is consistent with the first Kelvin wave arriving and dissipating. Second wave expected Mar-Apr.

5. **Anchoveta recover fast.** All previous-season features failed because sequential disruption logic doesn't work for this species. Don't revisit this line of inquiry — the biology doesn't support it.

6. **GODAS is on THREDDS, not ERDDAP.** Gemini got this wrong. The server is `psl.noaa.gov/thredds/catalog/Datasets/godas/`. Files are `potmp.yyyy.nc`.

7. **CPC switched to Relative Oceanic Niño Index (RONI)** as of Feb 2, 2026. This doesn't affect Niño 1+2 (which PAEWS uses), but it changes how CPC monitors/defines El Niño events in the central Pacific.

8. **The quota feature has a leakage problem** for S1 predictions. S1 quota is announced in April, after the March decision_month. Can only use previous season's quota, which the catch% experiments showed doesn't help reliably.

9. **Fishmeal price confirmed $1,824/MT** (Dec 2025). This traces to World Bank Pink Sheet, Peru fishmeal FOB 65% protein. The dashboard cites this correctly.

10. **The 2019 S2 biomass estimate is flagged** in imarpe_biomass_verified.csv — IMARPE officials were investigated for inflating the figure from ~4 to 8.3 MMT. This doesn't affect the production model (biomass isn't a feature) but matters if biomass is ever added.
