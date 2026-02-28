"""
predict_2026_s1.py - PAEWS 2026 S1 Live Prediction (Preliminary)

Usage:
    cd C:/Users/josep/Documents/paews
    python scripts/predict_2026_s1.py

Data already on disk (no downloads needed):
    - SST 2026: data/current/sst_current.nc (Jan 1 - Feb 8)
    - Chl 2026: data/current/chlorophyll_current.nc (Jan 1 - Feb 7, VIIRS)
    - SST climatology: data/processed/sst_climatology_v2.nc
    - Chl climatology: data/processed/chl_climatology_copernicus.nc
    - Nino indices: data/external/nino_indices_monthly.csv
    - Training data: data/external/paews_feature_matrix.csv

NOTE: S1 decision month is March, but we only have data through early Feb.
      This is a PRELIMINARY prediction. Rerun after March data is available.
"""

import numpy as np
import pandas as pd
import xarray as xr
import os
import sys
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# === PATHS ===
BASE = Path(__file__).resolve().parent.parent
DATA_CURRENT = BASE / "data" / "current"
DATA_BASELINE = BASE / "data" / "baseline"
DATA_PROCESSED = BASE / "data" / "processed"
DATA_EXTERNAL = BASE / "data" / "external"
FEATURE_CSV = DATA_EXTERNAL / "paews_feature_matrix.csv"

# === FEATURE COLUMNS (must match composite_score.py) ===
FEATURE_COLS = ['sst_z', 'chl_z', 'nino12_t1', 'is_summer', 'bio_thresh_pct']


def compute_sst_features(target_month=2):
    """
    Compute SST Z-score and bio threshold for Feb 2026.
    Matches composite_score.py: uses last available day in target month,
    computes pixel-wise Z-score against climatology, takes spatial mean.
    """
    print("  Loading SST climatology...", flush=True)
    clim = xr.open_dataset(DATA_PROCESSED / "sst_climatology_v2.nc")

    print("  Loading SST current (2026)...", flush=True)
    ds = xr.open_dataset(DATA_CURRENT / "sst_current.nc")

    # Filter to target month
    time_index = pd.DatetimeIndex(ds["time"].values)
    month_data = ds.sel(time=time_index.month == target_month)

    if len(month_data.time) == 0:
        print(f"  WARNING: No SST data for month {target_month}, using all available", flush=True)
        month_data = ds

    # Use last available timestep (matches composite_score.py approach)
    sst_snap = month_data["sst"].isel(time=-1).squeeze()

    # Drop zlev if present
    if 'zlev' in sst_snap.dims:
        sst_snap = sst_snap.isel(zlev=0)

    last_date = str(month_data.time.values[-1])[:10]
    print(f"  SST snapshot date: {last_date}", flush=True)
    print(f"  SST grid shape: {sst_snap.shape}", flush=True)

    # Climatology for target month
    clim_mean = clim["sst_mean"].sel(month=target_month)
    clim_std = clim["sst_std"].sel(month=target_month)
    std_safe = clim_std.where(clim_std > 0.01, 0.01)

    # Z-score map
    z_map = (sst_snap - clim_mean) / std_safe
    sst_z = float(z_map.mean(skipna=True))

    # MHW percentage (Z > 1.28)
    valid_z = z_map.where(z_map.notnull())
    mhw_count = int((valid_z > 1.28).sum())
    total_valid = int(valid_z.notnull().sum())
    mhw_pct = mhw_count / total_valid * 100 if total_valid > 0 else 0

    # Bio threshold: % pixels where absolute SST > 23C
    valid_sst = sst_snap.notnull()
    above_23 = (sst_snap > 23.0) & valid_sst
    total_valid_sst = int(valid_sst.sum())
    bio_thresh_pct = float(above_23.sum()) / total_valid_sst * 100 if total_valid_sst > 0 else 0

    ds.close()
    clim.close()

    print(f"  SST Z-score: {sst_z:+.3f}", flush=True)
    print(f"  MHW pixels:  {mhw_pct:.1f}%", flush=True)
    print(f"  Bio > 23C:   {bio_thresh_pct:.1f}%", flush=True)

    return sst_z, bio_thresh_pct, mhw_pct


def compute_chl_features(target_month=2):
    """
    Compute Chl Z-score for Feb 2026 from VIIRS daily data.
    Uses log-transform and compares against Copernicus climatology.
    """
    print("  Loading Chl climatology...", flush=True)

    # Try Copernicus climatology first, fall back to v2
    clim_path = DATA_PROCESSED / "chl_climatology_copernicus.nc"
    if not clim_path.exists():
        clim_path = DATA_PROCESSED / "chl_climatology_v2.nc"
    clim = xr.open_dataset(clim_path)
    print(f"  Using climatology: {clim_path.name}", flush=True)
    print(f"  Clim variables: {list(clim.data_vars)}", flush=True)

    print("  Loading Chl current (2026 VIIRS)...", flush=True)
    ds = xr.open_dataset(DATA_CURRENT / "chlorophyll_current.nc")

    # Filter to target month
    time_index = pd.DatetimeIndex(ds["time"].values)
    month_data = ds.sel(time=time_index.month == target_month)

    if len(month_data.time) == 0:
        print(f"  WARNING: No Chl data for month {target_month}, using all available", flush=True)
        month_data = ds

    # Get chlorophyll variable
    chl_var = 'chlor_a'
    if chl_var not in month_data:
        for candidate in ['CHL', 'chl', 'chlorophyll']:
            if candidate in month_data:
                chl_var = candidate
                break

    # Monthly mean of daily VIIRS data, then log-transform
    chl_monthly = month_data[chl_var].mean(dim='time').squeeze()

    # Drop altitude if present
    if 'altitude' in chl_monthly.dims:
        chl_monthly = chl_monthly.isel(altitude=0)

    chl_log = np.log10(chl_monthly.where(chl_monthly > 0))

    print(f"  Chl grid shape: {chl_monthly.shape}", flush=True)
    valid_pix = int(chl_monthly.notnull().sum())
    print(f"  Valid Chl pixels: {valid_pix}", flush=True)

    # Get climatological mean and std for target month
    # Variable names might differ between climatology files
    clim_mean_var = None
    clim_std_var = None
    for v in clim.data_vars:
        if 'mean' in v.lower() and 'log' in v.lower():
            clim_mean_var = v
        elif 'std' in v.lower() and 'log' in v.lower():
            clim_std_var = v

    # Fallback: try without 'log' qualifier
    if clim_mean_var is None:
        for v in clim.data_vars:
            if 'mean' in v.lower():
                clim_mean_var = v
            elif 'std' in v.lower():
                clim_std_var = v

    if clim_mean_var is None:
        print(f"  ERROR: Cannot find mean/std in climatology. Vars: {list(clim.data_vars)}")
        return np.nan

    print(f"  Clim mean var: {clim_mean_var}, std var: {clim_std_var}", flush=True)

    clim_mean = clim[clim_mean_var].sel(month=target_month)
    clim_std = clim[clim_std_var].sel(month=target_month)
    std_safe = clim_std.where(clim_std > 0.01, 0.01)

    # Interpolate VIIRS grid to climatology grid (they may differ)
    chl_lat = [d for d in chl_log.dims if 'lat' in d.lower()][0]
    chl_lon = [d for d in chl_log.dims if 'lon' in d.lower()][0]
    clim_lat = [d for d in clim_mean.dims if 'lat' in d.lower()][0]
    clim_lon = [d for d in clim_mean.dims if 'lon' in d.lower()][0]

    try:
        chl_log_interp = chl_log.interp({
            chl_lat: clim_mean[clim_lat],
            chl_lon: clim_mean[clim_lon],
        })
        z_map = (chl_log_interp - clim_mean) / std_safe
    except Exception as e:
        print(f"  Grid interpolation failed ({e}), computing on native grid", flush=True)
        z_map = (chl_log - clim_mean) / std_safe

    chl_z = float(z_map.mean(skipna=True))

    # Low-chl percentage
    lchl_count = int((z_map < -1.28).sum())
    total_valid_z = int(z_map.notnull().sum())
    lchl_pct = lchl_count / total_valid_z * 100 if total_valid_z > 0 else 0

    ds.close()
    clim.close()

    print(f"  Chl Z-score: {chl_z:+.3f}", flush=True)
    print(f"  Low-Chl pixels: {lchl_pct:.1f}%", flush=True)

    return chl_z


def get_nino12_t1():
    """
    Get Nino 1+2 anomaly for t-1 relative to decision month (March).
    t-1 = February. If Feb not available, use latest.
    """
    nino_path = DATA_EXTERNAL / "nino_indices_monthly.csv"
    df = pd.read_csv(nino_path)

    # Try to get Feb 2026
    feb = df[(df['year'] == 2026) & (df['month'] == 2)]
    if len(feb) > 0:
        val = float(feb['nino12_anom'].iloc[0])
        print(f"  Nino 1+2 (Feb 2026): {val:+.2f}", flush=True)
        return val

    # Fall back to latest available
    latest = df.dropna(subset=['nino12_anom']).iloc[-1]
    val = float(latest['nino12_anom'])
    print(f"  Nino 1+2 (latest: {int(latest['year'])}-{int(latest['month']):02d}): {val:+.2f}", flush=True)
    print(f"  NOTE: Feb 2026 not yet available, using {int(latest['year'])}-{int(latest['month']):02d}", flush=True)
    return val


def run_prediction(features_2026):
    """
    Train logistic regression on all 30 historical samples, predict 2026 S1.
    Matches composite_score.py feature order exactly.
    """
    train_df = pd.read_csv(FEATURE_CSV)

    # Drop rows with NaN in required features
    train_df = train_df.dropna(subset=FEATURE_COLS + ['target'])

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['target'].values

    print(f"  Training samples: {len(X_train)} ({int(y_train.sum())} positives)", flush=True)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train_s, y_train)

    # Build 2026 S1 feature vector (same column order as FEATURE_COLS)
    x_new = np.array([[
        features_2026['sst_z'],
        features_2026['chl_z'],
        features_2026['nino12_t1'],
        1.0,  # is_summer = True for S1
        features_2026['bio_thresh_pct'],
    ]])

    x_new_s = scaler.transform(x_new)
    prob = model.predict_proba(x_new_s)[0, 1]

    # Feature contributions (scaled coef * scaled value)
    coefs = model.coef_[0]
    contributions = coefs * x_new_s[0]

    return prob, coefs, x_new[0], x_new_s[0], contributions, scaler


def main():
    print("=" * 64)
    print("  PAEWS 2026 S1 LIVE PREDICTION (PRELIMINARY)")
    print(f"  Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Decision month: March (using Feb data - partial season)")
    print("=" * 64)
    print()

    # ---- Step 1: SST Features ----
    print("STEP 1: SST Features (Feb 2026)")
    print("-" * 40)
    sst_z, bio_thresh_pct, mhw_pct = compute_sst_features(target_month=2)
    print()

    # ---- Step 2: Chl Features ----
    print("STEP 2: Chlorophyll Features (Feb 2026)")
    print("-" * 40)
    chl_z = compute_chl_features(target_month=2)
    print()

    # ---- Step 3: Nino Index ----
    print("STEP 3: Nino 1+2 Index")
    print("-" * 40)
    nino12_t1 = get_nino12_t1()
    print()

    # ---- Step 4: Assemble Features ----
    features = {
        'sst_z': sst_z,
        'chl_z': chl_z if not np.isnan(chl_z) else 0.0,
        'nino12_t1': nino12_t1,
        'bio_thresh_pct': bio_thresh_pct,
    }

    print("STEP 4: Feature Summary")
    print("-" * 40)
    print(f"  {'Feature':<18} {'2026 S1':>10}  {'Historical Range':>20}")
    print(f"  {'-'*18} {'-'*10}  {'-'*20}")

    # Load historical for context
    hist = pd.read_csv(FEATURE_CSV)
    for feat in ['sst_z', 'chl_z', 'nino12_t1', 'bio_thresh_pct']:
        val = features[feat]
        col = hist[feat].dropna()
        rng = f"[{col.min():+.2f}, {col.max():+.2f}]"
        flag = " !!!" if val > col.max() or val < col.min() else ""
        print(f"  {feat:<18} {val:>+10.3f}  {rng:>20}{flag}")

    print(f"  {'is_summer':<18} {'1':>10}  {'[0, 1]':>20}")
    print()

    if np.isnan(chl_z):
        print("  WARNING: Chl Z-score is NaN. Using 0.0 (neutral) as fallback.")
        print()

    # ---- Step 5: Prediction ----
    print("STEP 5: Model Prediction")
    print("-" * 40)
    prob, coefs, x_raw, x_scaled, contributions, scaler = run_prediction(features)

    threshold = 0.38  # from existing model validation
    prediction = "AT RISK" if prob >= threshold else "NORMAL"

    print()
    print("=" * 64)
    if prediction == "AT RISK":
        print(f"  >>> 2026 S1 RISK PROBABILITY: {prob:.3f}  [{prediction}] <<<")
    else:
        print(f"      2026 S1 RISK PROBABILITY: {prob:.3f}  [{prediction}]")
    print(f"      Threshold: {threshold}")
    print("=" * 64)
    print()

    # Feature contribution breakdown
    print("  Feature Contributions (to log-odds):")
    print(f"  {'Feature':<18} {'Raw':>8} {'Scaled':>8} {'Coef':>8} {'Contrib':>10}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for i, feat in enumerate(FEATURE_COLS):
        direction = "+" if contributions[i] > 0 else ""
        print(f"  {feat:<18} {x_raw[i]:>8.3f} {x_scaled[i]:>8.3f} {coefs[i]:>8.3f} {direction}{contributions[i]:>9.3f}")

    intercept_approx = -np.log(1/prob - 1) - contributions.sum() if 0 < prob < 1 else 0
    print(f"  {'intercept':<18} {'':>8} {'':>8} {'':>8} {intercept_approx:>+10.3f}")
    print(f"  {'TOTAL log-odds':<18} {'':>8} {'':>8} {'':>8} {np.log(prob/(1-prob)):>+10.3f}")
    print()

    # Historical comparison
    print("  Historical Comparison (S1 seasons):")
    s1_hist = hist[hist['season'] == 1].dropna(subset=['sst_z', 'chl_z'])
    for _, row in s1_hist.iterrows():
        yr = int(row['year'])
        outcome = row['outcome'][:8] if isinstance(row['outcome'], str) else '?'
        target = int(row['target'])
        marker = " <-- DISRUPTED" if target == 1 else ""
        print(f"    {yr} S1: SST_Z={row['sst_z']:+.2f}  Chl_Z={row['chl_z']:+.2f}  "
              f"Bio={row['bio_thresh_pct']:5.1f}%  [{outcome}]{marker}")

    print(f"    2026 S1: SST_Z={sst_z:+.2f}  Chl_Z={features['chl_z']:+.2f}  "
          f"Bio={bio_thresh_pct:5.1f}%  [PREDICTION: {prediction}]")
    print()

    # Caveats
    print("  CAVEATS:")
    print("  - This uses Feb data; official S1 decision month is March")
    print("  - Rerun after March SST/Chl data becomes available")
    print("  - Feb Nino 1+2 not yet available; using Jan 2026 value")
    print(f"  - Model trained on {len(hist)} samples ({int(hist['target'].sum())} disruptions)")
    print()


if __name__ == "__main__":
    main()
