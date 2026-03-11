# PAEWS — Peru Anchovy Early Warning System

A satellite-based logistic regression model that predicts disruptions to Peru's anchovy (*Engraulis ringens*) fishing seasons using publicly available oceanographic data.

## Why This Matters

Peru's anchovy fishery is the world's largest single-species fishery, producing ~6 million tonnes annually and supplying ~20% of global fishmeal. Season disruptions — caused by El Niño events, stock depletion, or high juvenile incidence — can cascade through global fishmeal markets, impacting salmon farming, poultry feed, and food security.

PAEWS provides advance warning of season disruptions using three remotely-sensed oceanographic features, with the goal of giving fishmeal-dependent industries 4–8 weeks of lead time before official PRODUCE/IMARPE decisions.

## Model Summary

| Item | Value |
|---|---|
| **Type** | Logistic regression (binary: Normal vs. Disrupted) |
| **Features** | `sst_z`, `chl_z`, `nino12_t1` |
| **Training data** | 32 seasons (2010 S1 – 2025 S2) |
| **Disrupted seasons** | 12 of 32 (Reduced, Disrupted, or Cancelled) |
| **Validation** | Leave-One-Out Cross-Validation |
| **ROC-AUC** | 0.629 |
| **SEVERE tier accuracy** | 4/4 = 100% (zero false positives) |

### Features

- **`sst_z`** — Sea surface temperature z-score for Peru coastal box (0–16°S, 85–70°W), from NOAA OISST via ERDDAP (`ncdcOisst21Agg_LonPM180`)
- **`chl_z`** — Chlorophyll-a z-score from Copernicus Marine NRT, coastal productivity mask (top 50% most productive pixels)
- **`nino12_t1`** — NOAA CPC monthly Niño 1+2 index (°C anomaly), lagged by one month

### Tier System

| Tier | Probability | Historical disruption rate |
|---|---|---|
| SEVERE | ≥ 0.70 | 100% (4/4) |
| ELEVATED | 0.50–0.69 | 25% |
| MODERATE | 0.20–0.49 | ~30% |
| LOW | < 0.20 | 40% (few samples) |

## Current Prediction: 2026 S1

**Prediction: 0.398 MODERATE** (as of March 4, 2026)

Context: ENFEN declared El Niño Costero alert (Comunicados N°03-2026 and N°04-2026). Niño 1+2 at +1.28°C per SIOFEN Bulletin N°09-2026. Marine heatwave active (~130,000 km²). IMARPE cruise 2602-04 underway. Two Kelvin waves forecast for March–May 2026.

Scenario analysis shows the prediction is sensitive to chlorophyll changes: if coastal upwelling weakens (Chl drops to −0.80σ), the prediction crosses into SEVERE territory (0.718).

## Data Pipeline

All input data comes from free, public sources with automated download:

```
NOAA OISST (SST)          → ERDDAP API → sst_z
Copernicus Marine (Chl-a) → OPeNDAP   → chl_z  
NOAA CPC (Niño indices)   → CSV        → nino12_t1
```

No proprietary data. No IMARPE biomass data required for prediction (though biomass features are tracked for future model versions).

## Repository Structure

```
paews/
├── scripts/
│   ├── predict_2026_s1.py          # Production prediction
│   ├── scenario_analysis.py        # Scenario sweep
│   ├── data_pipeline.py            # SST processing
│   ├── chl_migration.py            # Copernicus Chl processing
│   ├── external_data_puller.py     # Pull Niño, ONI, fishmeal prices
│   ├── health_check.py             # Data validation
│   ├── model_v2_audit.py           # 10-point data audit
│   ├── godas_thermocline.py        # GODAS Z20 pipeline (rejected feature)
│   ├── build_juvenile_feature.py   # Juvenile % data collection template
│   └── data_refresh.ps1            # All refresh commands (PowerShell)
├── data/
│   └── external/
│       ├── paews_feature_matrix.csv
│       ├── imarpe_ground_truth.csv
│       ├── imarpe_biomass_verified.csv
│       └── sst_current.nc
├── paews_performance_dashboard.jsx  # Interactive React dashboard
└── README.md
```

## Known Limitations

- **Small sample size**: 32 seasons limits statistical power. Bootstrap 95% CI for the current prediction spans [0.136, 0.738].
- **Remote sensing ceiling**: The three persistent model misses (2014 S1, 2022 S2, 2011 S2) are biological, not oceanographic. Satellite-derived features alone likely max out around ROC-AUC 0.65–0.70.
- **Chlorophyll latency**: Copernicus NRT products have ~1 month delay. The current prediction uses a December 2025 proxy.
- **No biomass feature**: IMARPE biomass estimates would likely improve the model but are not publicly available in machine-readable form.

## Roadmap

- [ ] Integrate pre-season juvenile percentage from IMARPE cruise reports (highest-value untested feature)
- [ ] Automate Copernicus NRT chlorophyll refresh
- [ ] Add IMARPE biomass data as features once sufficient seasons are collected
- [ ] Explore GODAS equatorial thermocline depth at longer lead times
- [ ] Backtest against 1990–2009 seasons

## Data Integrity Policy

All ground truth and feature values are manually verified against primary sources. No AI-generated or AI-extracted numerical values are used — Claude (Anthropic) assists with pipeline code and analysis, but all data values are entered and verified by a human operator.

## Target Users

- Norwegian salmon companies (Austevoll Seafood, Mowi, SalMar) exposed to fishmeal price volatility
- Fishmeal/fish oil traders and commodity analysts
- Fisheries management bodies and marine conservation organizations
- Space industry portfolio reviewers (satellite remote sensing application)

## Author

Joseph Bell — GIS analyst building satellite-based data science projects at the intersection of remote sensing and environmental monitoring.

GitHub: [@monkeqi](https://github.com/monkeqi)

## License

MIT
