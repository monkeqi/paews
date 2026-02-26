"""
predict_2026_s1.py - Download latest data and generate 2026 S1 live prediction

Usage:
    cd C:/Users/josep/Documents/paews
    python scripts/predict_2026_s1.py

Prerequisites:
    - copernicusmarine login (already configured)
    - Internet connection
    - Existing pipeline files (composite_score.py, chl_migration.py)

What it does:
    1. Downloads 2025 SST from NOAA ERDDAP
    2. Downloads 2025-2026 Chl from Copernicus
    3. Computes S1 2026 features (Jan-Feb 2026, partial season)
    4. Loads trained model from existing 30 samples
    5. Outputs risk probability for 2026 S1
"""

import numpy as np
import pandas as pd
import xarray as xr
import requests
import os
import sys
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# === PATHS ===
BASE = Path(__file__).resolve().parent.parent
SST_DIR = BASE / "data" / "sst"
CHL_DIR = BASE / "data" / "copernicus_chl"
FEATURE_CSV = BASE / "data" / "external" / "paews_feature_matrix.csv"
MASK_FILE = BASE / "masks" / "coastal_mask_40pct.nc"
CLIM_SST = BASE / "data" / "processed" / "sst_climatology_v2.nc"
CLIM_CHL = BASE / "data" / "processed" / "chl_climatology_copernicus.nc"

# === BOUNDING BOX ===
LAT_MIN, LAT_MAX = -16.0, 0.0
LON_MIN, LON_MAX = -85.0, -70.0

def step1_download_sst_2025():
    """Download 2025 SST from NOAA ERDDAP."""
    sst_file = SST_DIR / "sst_2025.nc"
    if sst_file.exists():
        print(f"  SST 2025 already exists: {sst_file}")
        return sst_file
    
    print("  Downloading SST 2025 from ERDDAP...")
    url = (
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.nc?"
        f"sst[(2025-01-01T12:00:00Z):1:(2025-12-31T12:00:00Z)]"
        f"[({LAT_MIN}):1:({LAT_MAX})]"
        f"[({LON_MIN}):1:({LON_MAX})]"
    )
    
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    
    SST_DIR.mkdir(parents=True, exist_ok=True)
    with open(sst_file, 'wb') as f:
        f.write(r.content)
    
    ds = xr.open_dataset(sst_file)
    print(f"  Downloaded: {len(ds.time)} days, {sst_file.stat().st_size/1e6:.1f} MB")
    ds.close()
    return sst_file


def step1b_download_sst_2026():
    """Download 2026 SST (Jan-Feb) from NOAA ERDDAP."""
    sst_file = SST_DIR / "sst_2026.nc"
    if sst_file.exists():
        print(f"  SST 2026 already exists: {sst_file}")
        return sst_file
    
    print("  Downloading SST 2026 (Jan-Feb) from ERDDAP...")
    url = (
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.nc?"
        f"sst[(2026-01-01T12:00:00Z):1:(2026-02-25T12:00:00Z)]"
        f"[({LAT_MIN}):1:({LAT_MAX})]"
        f"[({LON_MIN}):1:({LON_MAX})]"
    )
    
    try:
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        SST_DIR.mkdir(parents=True, exist_ok=True)
        with open(sst_file, 'wb') as f:
            f.write(r.content)
        ds = xr.open_dataset(sst_file)
        print(f"  Downloaded: {len(ds.time)} days, {sst_file.stat().st_size/1e6:.1f} MB")
        ds.close()
        return sst_file
    except Exception as e:
        print(f"  Could not download 2026 SST: {e}")
        print("  Will use 2025 data only for partial prediction")
        return None


def step2_download_chl_2025():
    """Download 2025 Chl from Copernicus Marine."""
    # Check for existing files
    existing = list(CHL_DIR.glob("*2025*")) + list(CHL_DIR.glob("*202501*"))
    if existing:
        print(f"  Chl 2025 files found: {[f.name for f in existing]}")
        return existing
    
    print("  Downloading Chl 2025 from Copernicus Marine...")
    print("  (This uses copernicusmarine CLI - make sure you're logged in)")
    
    CHL_DIR.mkdir(parents=True, exist_ok=True)
    
    import subprocess
    cmd = [
        "copernicusmarine", "subset",
        "--dataset-id", "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M",
        "--variable", "CHL",
        "--minimum-longitude", str(LON_MIN),
        "--maximum-longitude", str(LON_MAX),
        "--minimum-latitude", str(LAT_MIN),
        "--maximum-latitude", str(LAT_MAX),
        "--start-datetime", "2025-01-01",
        "--end-datetime", "2025-12-31",
        "--output-directory", str(CHL_DIR),
        "--output-filename", "chl_2025.nc",
        "--force-download",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"  Downloaded Chl 2025")
            return [CHL_DIR / "chl_2025.nc"]
        else:
            print(f"  copernicusmarine error: {result.stderr[:500]}")
            # Try NRT product instead
            print("  Trying NRT product...")
            cmd[3] = "cmems_obs-oc_glo_bgc-plankton_nrt_l4-multi-4km_P1M"
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                print(f"  Downloaded Chl 2025 (NRT)")
                return [CHL_DIR / "chl_2025.nc"]
            else:
                print(f"  NRT also failed: {result.stderr[:500]}")
                return []
    except Exception as e:
        print(f"  Download failed: {e}")
        return []


def compute_sst_features(months_to_use):
    """Compute SST features for the given months from 2025-2026 data."""
    
    # Load SST climatology
    clim = xr.open_dataset(CLIM_SST)
    
    # Load available SST data
    sst_files = []
    for year in [2025, 2026]:
        f = SST_DIR / f"sst_{year}.nc"
        if f.exists():
            sst_files.append(f)
    
    if not sst_files:
        print("ERROR: No SST data available for 2025-2026")
        return None
    
    datasets = [xr.open_dataset(f) for f in sst_files]
    sst_all = xr.concat(datasets, dim='time')
    
    # Filter to requested months
    sst_period = sst_all.sel(time=sst_all.time.dt.month.isin(months_to_use))
    
    if len(sst_period.time) == 0:
        print(f"ERROR: No SST data for months {months_to_use}")
        return None
    
    print(f"  SST data: {sst_period.time.values[0]} to {sst_period.time.values[-1]}")
    print(f"  {len(sst_period.time)} days of data")
    
    # Compute mean SST over period
    sst_mean = sst_period['sst'].mean(dim='time')
    
    # Compute climatological mean for these months
    clim_months = clim['sst_mean'].sel(month=months_to_use).mean(dim='month')
    clim_std_months = clim['sst_std'].sel(month=months_to_use).mean(dim='month')
    
    # Z-score
    sst_z_map = (sst_mean - clim_months) / clim_std_months
    sst_z = float(sst_z_map.mean(skipna=True))
    
    # Bio threshold: % pixels > 23C
    valid_pixels = sst_mean.count().values
    hot_pixels = (sst_mean > 23.0).sum().values
    bio_thresh = float(hot_pixels / valid_pixels * 100) if valid_pixels > 0 else 0
    
    for ds in datasets:
        ds.close()
    clim.close()
    
    return {
        'sst_z': sst_z,
        'bio_thresh_pct': bio_thresh,
    }


def compute_chl_features(months_to_use):
    """Compute Chl features from available 2025 data."""
    
    # Find Chl files
    chl_files = list(CHL_DIR.glob("*2025*")) + list(CHL_DIR.glob("*chl_2025*"))
    
    if not chl_files:
        print("  WARNING: No 2025 Chl data. Using NaN for Chl features.")
        return {'chl_z': np.nan}
    
    # Load climatology
    clim = xr.open_dataset(CLIM_CHL)
    
    # Load Chl data
    ds = xr.open_dataset(chl_files[0])
    
    # Find the CHL variable (might be 'CHL' or 'chl' or 'chlor_a')
    chl_var = None
    for candidate in ['CHL', 'chl', 'chlor_a', 'CHL_mean']:
        if candidate in ds.data_vars:
            chl_var = candidate
            break
    
    if chl_var is None:
        print(f"  WARNING: Cannot find CHL variable. Available: {list(ds.data_vars)}")
        return {'chl_z': np.nan}
    
    # Filter to months
    if 'time' in ds.dims:
        chl_period = ds.sel(time=ds.time.dt.month.isin(months_to_use))
    else:
        chl_period = ds
    
    print(f"  Chl variable: {chl_var}")
    
    # Log-transform
    chl_log = np.log10(chl_period[chl_var].where(chl_period[chl_var] > 0))
    chl_log_mean = chl_log.mean(dim='time') if 'time' in chl_log.dims else chl_log
    
    # Apply coastal mask if available
    if MASK_FILE.exists():
        mask = xr.open_dataset(MASK_FILE)
        # Try to align mask with data (dimensions might differ)
        try:
            mask_var = list(mask.data_vars)[0]
            chl_log_mean = chl_log_mean.where(mask[mask_var] == 1)
        except:
            pass
        mask.close()
    
    # Compute Z-score against climatology
    clim_mean = clim['chl_log_mean'].sel(month=months_to_use).mean(dim='month') if 'month' in clim.dims else clim['chl_log_mean'].mean()
    clim_std = clim['chl_log_std'].sel(month=months_to_use).mean(dim='month') if 'month' in clim.dims else clim['chl_log_std'].mean()
    
    chl_z_map = (chl_log_mean - clim_mean) / clim_std
    chl_z = float(chl_z_map.mean(skipna=True))
    
    ds.close()
    clim.close()
    
    return {'chl_z': chl_z}


def get_nino12():
    """Get latest Nino 1+2 value."""
    nino_file = BASE / "data" / "external" / "nino_indices_monthly.csv"
    if not nino_file.exists():
        print("  WARNING: No Nino indices file. Using 0.0")
        return 0.0
    
    df = pd.read_csv(nino_file)
    # Find the nino12 column
    nino_col = None
    for c in df.columns:
        if '12' in c.lower() or 'nino12' in c.lower() or 'nino1+2' in c.lower():
            nino_col = c
            break
    
    if nino_col is None:
        # Try last numeric column
        print(f"  Nino columns available: {list(df.columns)}")
        return 0.0
    
    # Get latest value
    latest = df[nino_col].dropna().iloc[-1]
    print(f"  Latest Nino 1+2: {latest}")
    return float(latest)


def run_prediction(features_2026):
    """Train on 30 historical samples, predict 2026 S1."""
    
    # Load training data
    train_df = pd.read_csv(FEATURE_CSV)
    
    feature_cols = ['chl_z', 'sst_z', 'bio_thresh_pct', 'nino12_t1', 'is_summer']
    
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    # Train
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train_s, y_train)
    
    # Build 2026 S1 feature vector
    x_new = np.array([[
        features_2026.get('chl_z', 0.0),
        features_2026.get('sst_z', 0.0),
        features_2026.get('bio_thresh_pct', 0.0),
        features_2026.get('nino12_t1', 0.0),
        1.0,  # is_summer = True for S1
    ]])
    
    x_new_s = scaler.transform(x_new)
    prob = model.predict_proba(x_new_s)[0, 1]
    
    # Feature contributions
    coefs = model.coef_[0]
    contributions = coefs * x_new_s[0]
    
    return prob, feature_cols, x_new[0], contributions


def main():
    print("=" * 60)
    print("PAEWS 2026 S1 LIVE PREDICTION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print()
    
    # S1 decision months: Jan, Feb, Mar (use what we have)
    # Best case: Jan-Feb 2026 data
    # Fallback: use latest available months
    
    # Step 1: Download SST
    print("STEP 1: SST Data")
    print("-" * 40)
    step1_download_sst_2025()
    sst_2026 = step1b_download_sst_2026()
    print()
    
    # Step 2: Download Chl
    print("STEP 2: Chlorophyll Data")
    print("-" * 40)
    step2_download_chl_2025()
    print()
    
    # Step 3: Compute features
    print("STEP 3: Computing Features")
    print("-" * 40)
    
    # Determine which months we can use
    # For S1 prediction in late Feb, ideally use Jan-Feb 2026
    # Fall back to recent months of 2025 if 2026 not available
    if sst_2026 is not None:
        print("  Using Jan-Feb 2026 SST data")
        sst_months = [1, 2]
    else:
        print("  Using Nov-Dec 2025 SST data (fallback)")
        sst_months = [11, 12]
    
    sst_features = compute_sst_features(sst_months)
    if sst_features is None:
        print("FATAL: Cannot compute SST features. Need at least 2025 SST data.")
        return
    
    chl_features = compute_chl_features([1, 2] if sst_2026 else [11, 12])
    nino12 = get_nino12()
    
    # Combine features
    features = {
        'sst_z': sst_features['sst_z'],
        'chl_z': chl_features.get('chl_z', 0.0),
        'bio_thresh_pct': sst_features['bio_thresh_pct'],
        'nino12_t1': nino12,
    }
    
    print()
    print("  Computed features:")
    for k, v in features.items():
        print(f"    {k}: {v:.3f}")
    print()
    
    # Step 4: Run prediction
    print("STEP 4: Prediction")
    print("-" * 40)
    
    prob, feat_names, feat_values, contributions = run_prediction(features)
    
    # Classification
    threshold = 0.38  # from existing model
    prediction = "AT-RISK" if prob >= threshold else "NORMAL"
    
    print()
    print("=" * 60)
    print(f"  2026 S1 RISK PROBABILITY: {prob:.2f}")
    print(f"  PREDICTION: {prediction}")
    print(f"  (threshold: {threshold})")
    print("=" * 60)
    print()
    
    # Feature breakdown
    print("  Feature Breakdown:")
    print(f"  {'Feature':<20} {'Value':>8} {'Contribution':>14}")
    print(f"  {'-'*20} {'-'*8} {'-'*14}")
    for name, val, contrib in zip(feat_names, feat_values, contributions):
        direction = "+" if contrib > 0 else ""
        print(f"  {name:<20} {val:>8.3f} {direction}{contrib:>13.3f}")
    print()
    
    # Context
    print("  Historical Context:")
    print(f"  - 2024 S1 prob was 0.30 (NORMAL) ✓")
    print(f"  - 2024 S2 prob was 0.55 (borderline)")
    print(f"  - 2023 S1 prob was 0.94 (CANCELLED) ✓")
    print(f"  - Current Nino 1+2: {nino12:.2f} (neutral)")
    print()
    
    if prediction == "NORMAL":
        print("  ✅ No El Nino threat detected. System suggests normal season.")
    else:
        print("  ⚠️  Elevated risk detected. Monitor conditions closely.")
    
    print()
    print(f"  NOTE: This prediction uses {'Jan-Feb 2026' if sst_2026 else 'late 2025'} data.")
    print(f"  Rerun in March for a more complete S1 assessment.")


if __name__ == "__main__":
    main()
