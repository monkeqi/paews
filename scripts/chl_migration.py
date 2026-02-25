"""
PAEWS Copernicus Chl Migration
================================
Replaces the dying MODIS Aqua chlorophyll baseline with the unified
Copernicus GlobColour L4 multi-sensor product (MODIS+VIIRS+OLCI).

This script:
  1. Loads the full Copernicus Chl history (2003-2025, monthly)
  2. Computes a new log-space climatology (2003-2022 baseline)
  3. Recomputes Chl Z-scores for ALL seasons
  4. Updates the feature matrix CSV

CRITICAL: MODIS Aqua decommissions August 2026. This migration
must be complete before then to keep the system operational.

Usage:
    python chl_migration.py
"""

import sys
print("PAEWS Chl Migration starting...", flush=True)

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("Imports done", flush=True)

BASE_DIR = Path("c:/Users/josep/Documents/paews")
DATA_EXTERNAL = BASE_DIR / "data" / "external"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
FEAT_PATH = DATA_EXTERNAL / "paews_feature_matrix.csv"

CHL_FULL_PATH = DATA_EXTERNAL / "chl_copernicus_full.nc"

# Climatology baseline — match SST period
CLIM_START = 2003
CLIM_END = 2022

# Minimum observations for reliable climatology pixel
MIN_OBS = 5

# Coastal productivity mask: only average Z-scores over pixels where
# climatological mean Chl exceeds this percentile. This filters out
# open-ocean pixels that dilute the upwelling signal.
PRODUCTIVE_PERCENTILE = 50  # top 50% most productive pixels


def load_copernicus_chl():
    """Load the full Copernicus Chl dataset."""
    if not CHL_FULL_PATH.exists():
        print(f"  ERROR: {CHL_FULL_PATH} not found", flush=True)
        sys.exit(1)
    
    ds = xr.open_dataset(CHL_FULL_PATH)
    print(f"  Loaded: {ds.sizes['time']} months, "
          f"{ds.sizes['latitude']}x{ds.sizes['longitude']} grid", flush=True)
    print(f"  Range: {str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]}", flush=True)
    return ds


def compute_chl_climatology(ds):
    """
    Compute monthly Chl climatology in log10 space.
    
    Matches the approach used in the original chl_climatology.py:
    - Log10 transform (Chl is log-normally distributed)
    - Monthly mean and std per pixel
    - Observation count for masking sparse areas
    """
    print(f"\n  Computing Chl climatology ({CLIM_START}-{CLIM_END})...", flush=True)
    
    # Select baseline period
    baseline = ds.sel(time=slice(f"{CLIM_START}-01-01", f"{CLIM_END}-12-31"))
    print(f"  Baseline months: {baseline.sizes['time']}", flush=True)
    
    # Log10 transform (mask zeros and negatives)
    chl = baseline["CHL"]
    chl_log = np.log10(chl.where(chl > 0))
    
    # Monthly statistics
    chl_log_mean = chl_log.groupby('time.month').mean(dim='time')
    chl_log_std = chl_log.groupby('time.month').std(dim='time')
    chl_obs_count = chl_log.groupby('time.month').count(dim='time')
    
    # Build climatology dataset (matching original variable names)
    clim = xr.Dataset({
        'chl_log_mean': chl_log_mean,
        'chl_log_std': chl_log_std,
        'chl_obs_count': chl_obs_count,
    })
    
    # Save with versioned name
    clim_path = DATA_PROCESSED / "chl_climatology_copernicus.nc"
    clim.to_netcdf(clim_path)
    print(f"  Saved: {clim_path}", flush=True)
    
    # Print summary for decision months
    for m in [3, 10]:
        mean_val = float(chl_log_mean.sel(month=m).mean(skipna=True))
        std_val = float(chl_log_std.sel(month=m).mean(skipna=True))
        obs_val = float(chl_obs_count.sel(month=m).mean(skipna=True))
        print(f"    Month {m:2d}: log10(Chl) mean={mean_val:.3f}, "
              f"std={std_val:.3f}, avg obs={obs_val:.0f}", flush=True)
    
    # Static ocean pixel count (for percentage calculations)
    valid_pixels = int(chl_log_mean.sel(month=3).notnull().sum())
    print(f"  Static ocean pixels: {valid_pixels}", flush=True)
    
    # Build coastal productivity mask
    # Annual mean across all months — high values = upwelling zone
    annual_mean = chl_log_mean.mean(dim='month')
    threshold = float(np.nanpercentile(annual_mean.values[~np.isnan(annual_mean.values)],
                                        100 - PRODUCTIVE_PERCENTILE))
    productive_mask = annual_mean >= threshold
    coastal_pixels = int(productive_mask.sum())
    print(f"  Productive mask: top {PRODUCTIVE_PERCENTILE}% → {coastal_pixels} pixels "
          f"(threshold: log10(Chl) >= {threshold:.3f}, "
          f"Chl >= {10**threshold:.2f} mg/m³)", flush=True)
    
    return clim, valid_pixels, productive_mask


def compare_with_modis(cop_clim):
    """Compare new Copernicus climatology with old MODIS one."""
    modis_path = DATA_PROCESSED / "chl_climatology_v2.nc"
    if not modis_path.exists():
        print("\n  MODIS climatology not found — skipping comparison", flush=True)
        return
    
    modis = xr.open_dataset(modis_path)
    print("\n  COPERNICUS vs MODIS CLIMATOLOGY COMPARISON:", flush=True)
    print(f"  {'Month':>5} {'Cop_mean':>10} {'MODIS_mean':>11} {'Diff':>8}", flush=True)
    print(f"  {'-'*38}", flush=True)
    
    for m in range(1, 13):
        cop_val = float(cop_clim['chl_log_mean'].sel(month=m).mean(skipna=True))
        modis_val = float(modis['chl_log_mean'].sel(month=m).mean(skipna=True))
        diff = cop_val - modis_val
        flag = " ←" if abs(diff) > 0.05 else ""
        print(f"  {m:>5} {cop_val:>10.4f} {modis_val:>11.4f} {diff:>+8.4f}{flag}", flush=True)
    
    modis.close()


def extract_chl_z_for_season(ds, clim, year, month, static_pixels, productive_mask=None):
    """
    Compute mean Chl Z-score for a decision month using Copernicus data.
    
    If productive_mask is provided, only averages over coastal/upwelling pixels.
    
    Returns: (mean_z, lchl_pct)
      - mean_z: area-mean Chl Z-score (coastal pixels only if masked)
      - lchl_pct: % of pixels with Z < -1.28 (low chlorophyll)
    """
    try:
        month_data = ds.sel(time=f"{year}-{month:02d}")
        if 'time' in month_data["CHL"].dims:
            chl_snap = month_data["CHL"].isel(time=0).squeeze()
        else:
            chl_snap = month_data["CHL"].squeeze()
    except (KeyError, IndexError):
        return np.nan, np.nan
    
    # Log transform
    chl_log = np.log10(chl_snap.where(chl_snap > 0))
    
    # Climatology for this month
    clim_mean = clim['chl_log_mean'].sel(month=month)
    clim_std = clim['chl_log_std'].sel(month=month)
    obs_count = clim['chl_obs_count'].sel(month=month)
    
    # Safe std
    std_safe = clim_std.where(clim_std > 0.01)
    
    # Z-score
    z = (chl_log - clim_mean) / std_safe
    z = z.where(obs_count >= MIN_OBS)
    
    # Apply coastal mask if provided
    if productive_mask is not None:
        z_masked = z.where(productive_mask)
        mean_z = float(z_masked.mean(skipna=True))
    else:
        mean_z = float(z.mean(skipna=True))
    
    # Low Chl percentage (still computed on full domain for coverage metric)
    lchl_count = int((z < -1.28).sum(skipna=True))
    lchl_pct = lchl_count / static_pixels * 100 if static_pixels > 0 else 0
    
    return mean_z, lchl_pct


def update_feature_matrix(season_chl):
    """Merge Copernicus Chl Z-scores into feature matrix."""
    if not FEAT_PATH.exists():
        print(f"  ERROR: Feature matrix not found at {FEAT_PATH}", flush=True)
        return False
    
    df = pd.read_csv(FEAT_PATH)
    
    # Add tracking column
    if 'chl_source' not in df.columns:
        df['chl_source'] = 'MODIS'
    
    update_count = 0
    for entry in season_chl:
        mask = (df['year'] == entry['year']) & (df['season'] == entry['season'])
        if mask.any() and not np.isnan(entry['chl_z']):
            df.loc[mask, 'chl_z'] = entry['chl_z']
            df.loc[mask, 'lchl_pct'] = entry['lchl_pct']
            df.loc[mask, 'chl_source'] = 'Copernicus'
            
            # Recompute hardcoded composite
            row = df.loc[mask].iloc[0]
            sst_z = row.get('sst_z', np.nan)
            nino = row.get('nino12_t1', 0)
            if not pd.isna(sst_z):
                nino_val = nino if not pd.isna(nino) else 0
                composite = 0.4 * sst_z + 0.4 * (-entry['chl_z']) + 0.2 * nino_val
                df.loc[mask, 'composite_hard'] = composite
            
            update_count += 1
    
    df.to_csv(FEAT_PATH, index=False)
    print(f"\n  Updated {update_count} seasons in {FEAT_PATH.name}", flush=True)
    return True


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("PAEWS COPERNICUS CHL MIGRATION", flush=True)
    print("Replacing MODIS baseline with unified multi-sensor product", flush=True)
    print("=" * 60, flush=True)
    
    # ---- Step 1: Load data ----
    print("\nStep 1: Loading Copernicus Chl...", flush=True)
    ds = load_copernicus_chl()
    
    # ---- Step 2: Compute new climatology ----
    print("\nStep 2: Computing new climatology...", flush=True)
    clim, static_pixels, productive_mask = compute_chl_climatology(ds)
    
    # ---- Step 3: Compare with MODIS ----
    print("\nStep 3: Comparing with MODIS baseline...", flush=True)
    compare_with_modis(clim)
    
    # ---- Step 4: Recompute all Chl Z-scores ----
    print("\nStep 4: Recomputing Chl Z-scores for all seasons...", flush=True)
    
    gt_path = DATA_EXTERNAL / "imarpe_ground_truth.csv"
    gt = pd.read_csv(gt_path)
    
    season_results = []
    print(f"\n  {'Year':>4} {'S':>1} {'Outcome':>10} {'Chl_Z':>7} {'LChl%':>6} {'Src':>4}", flush=True)
    print(f"  {'-'*40}", flush=True)
    
    for _, row in gt.iterrows():
        year = int(row['year'])
        season = int(row['season'])
        outcome = row['outcome']
        decision_month = 3 if season == 1 else 10
        
        chl_z, lchl_pct = extract_chl_z_for_season(
            ds, clim, year, decision_month, static_pixels, productive_mask
        )
        
        season_results.append({
            'year': year,
            'season': season,
            'chl_z': chl_z,
            'lchl_pct': lchl_pct,
        })
        
        if not np.isnan(chl_z):
            print(f"  {year:>4} {season:>1} {outcome:>10} {chl_z:>+7.2f} {lchl_pct:>5.1f}%  COP",
                  flush=True)
        else:
            print(f"  {year:>4} {season:>1} {outcome:>10}     N/A    N/A    -", flush=True)
    
    # ---- Step 5: Update feature matrix ----
    print("\nStep 5: Updating feature matrix...", flush=True)
    update_feature_matrix(season_results)
    
    # ---- Summary ----
    print("\n" + "=" * 60, flush=True)
    print("CHL MIGRATION COMPLETE", flush=True)
    print("=" * 60, flush=True)
    
    filled = sum(1 for r in season_results if not np.isnan(r['chl_z']))
    print(f"  Seasons with Copernicus Chl: {filled}/{len(season_results)}", flush=True)
    print(f"  New climatology: {DATA_PROCESSED / 'chl_climatology_copernicus.nc'}", flush=True)
    print(f"  Static ocean pixels: {static_pixels}", flush=True)
    
    # Flag any major shifts from MODIS
    print(f"\n  IMPORTANT: The Chl Z-scores may shift slightly vs MODIS.", flush=True)
    print(f"  The multi-sensor product tends to be slightly higher in", flush=True)
    print(f"  coastal zones (better gap-fill under clouds).", flush=True)
    print(f"  Rerun composite_score.py to see impact on PR-AUC.", flush=True)
    
    print(f"\n  Now rerun: python composite_score.py", flush=True)
    print("=" * 60, flush=True)
    
    ds.close()
