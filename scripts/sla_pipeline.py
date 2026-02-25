"""
PAEWS SLA Pipeline: Sea Level Anomaly Feature Integration
===========================================================
Downloads are done externally via copernicusmarine CLI.
This script:
  1. Loads the SLA multi-year NetCDF (2010-2024)
  2. Computes a monthly SLA climatology (mean & std per pixel)
  3. Extracts SLA Z-scores for each season's decision month
  4. Updates the feature matrix CSV

SLA catches subsurface Kelvin waves that SST misses — the key to
fixing the 2014 S1 "ghost miss" where SST looked cool but a 
+10-15cm sea level bulge signaled incoming disruption.

Usage:
    python sla_pipeline.py
"""

import sys
print("PAEWS SLA Pipeline starting...", flush=True)

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

# SLA file from Copernicus (multi-year reprocessed, daily, 0.25°)
SLA_PATH = DATA_EXTERNAL / "sla_2010_2024.nc"

# Climatology baseline period — same as SST (avoids El Niño contamination)
CLIM_START = 2010
CLIM_END = 2022


def load_sla_data():
    """Load the SLA NetCDF. Variable may be 'sla', 'adt', or 'zos'."""
    if not SLA_PATH.exists():
        print(f"  ERROR: {SLA_PATH} not found", flush=True)
        print(f"  Run the download command first:", flush=True)
        print(f"  copernicusmarine subset --dataset-id cmems_obs-sl_glo_phy-ssh_my_all-sat-l4-duacs-0.25deg_P1D "
              f"--variable sla --minimum-longitude -85 --maximum-longitude -70 "
              f"--minimum-latitude -20 --maximum-latitude 0 "
              f"--start-datetime 2010-01-01T00:00:00 --end-datetime 2024-12-31T23:59:59 "
              f"--output-directory {DATA_EXTERNAL} --output-filename sla_2010_2024.nc", flush=True)
        sys.exit(1)
    
    ds = xr.open_dataset(SLA_PATH)
    
    # Find the SLA variable (different products use different names)
    sla_var = None
    for candidate in ['sla', 'adt', 'zos', 'ssh']:
        if candidate in ds.data_vars:
            sla_var = candidate
            break
    
    if sla_var is None:
        print(f"  Available variables: {list(ds.data_vars)}", flush=True)
        print(f"  ERROR: Could not find SLA variable. Check dataset.", flush=True)
        sys.exit(1)
    
    # Standardize coordinate names
    rename_map = {}
    for dim in ds.dims:
        if 'lat' in dim.lower():
            rename_map[dim] = 'latitude'
        elif 'lon' in dim.lower():
            rename_map[dim] = 'longitude'
    if rename_map:
        ds = ds.rename(rename_map)
    
    print(f"  SLA loaded: variable='{sla_var}', "
          f"{ds.sizes.get('time', 0)} timesteps, "
          f"lat {float(ds.latitude.min()):.1f} to {float(ds.latitude.max()):.1f}, "
          f"lon {float(ds.longitude.min()):.1f} to {float(ds.longitude.max()):.1f}", flush=True)
    
    return ds, sla_var


def compute_sla_climatology(ds, sla_var):
    """
    Compute monthly SLA climatology (mean and std per pixel).
    
    Uses 2010-2022 baseline to match SST climatology period.
    Returns xarray Dataset with sla_mean(month, lat, lon) and sla_std(month, lat, lon).
    """
    print(f"\n  Computing SLA climatology ({CLIM_START}-{CLIM_END})...", flush=True)
    
    # Select baseline period
    baseline = ds.sel(time=slice(f"{CLIM_START}-01-01", f"{CLIM_END}-12-31"))
    sla = baseline[sla_var]
    
    # Add month coordinate for grouping
    month_groups = sla.groupby('time.month')
    
    # Monthly mean and std
    sla_mean = month_groups.mean(dim='time')
    sla_std = month_groups.std(dim='time')
    
    # Count valid observations per month
    sla_count = sla.groupby('time.month').count(dim='time')
    
    # Build climatology dataset
    clim = xr.Dataset({
        'sla_mean': sla_mean,
        'sla_std': sla_std,
        'sla_count': sla_count,
    })
    
    # Save
    clim_path = DATA_PROCESSED / "sla_climatology.nc"
    clim.to_netcdf(clim_path)
    print(f"  Saved: {clim_path}", flush=True)
    
    # Print summary
    for m in [3, 10]:  # Decision months
        mean_val = float(sla_mean.sel(month=m).mean(skipna=True))
        std_val = float(sla_std.sel(month=m).mean(skipna=True))
        print(f"    Month {m:2d}: mean SLA = {mean_val*100:+.1f} cm, "
              f"avg std = {std_val*100:.1f} cm", flush=True)
    
    return clim


def extract_sla_z_for_season(ds, sla_var, clim, year, month):
    """
    Compute mean SLA Z-score for a decision month.
    
    With monthly data, we select the single timestep directly.
    
    Returns: (mean_z, mean_sla_cm, pct_positive)
      - mean_z: area-mean Z-score
      - mean_sla_cm: raw mean SLA in cm (for interpretability)
      - pct_positive: % of pixels with SLA > 0 (Kelvin wave footprint)
    """
    # Select the monthly timestep
    try:
        month_data = ds.sel(time=f"{year}-{month:02d}")
        if 'time' in month_data[sla_var].dims:
            sla_monthly = month_data[sla_var].isel(time=0).squeeze()
        else:
            sla_monthly = month_data[sla_var].squeeze()
    except (KeyError, IndexError):
        return np.nan, np.nan, np.nan
    
    # Climatology for this month
    clim_mean = clim['sla_mean'].sel(month=month)
    clim_std = clim['sla_std'].sel(month=month)
    
    # Interpolate climatology to data grid if needed
    if not np.array_equal(clim_mean.latitude.values, sla_monthly.latitude.values):
        clim_mean = clim_mean.interp(
            latitude=sla_monthly.latitude, longitude=sla_monthly.longitude,
            method="nearest"
        )
        clim_std = clim_std.interp(
            latitude=sla_monthly.latitude, longitude=sla_monthly.longitude,
            method="nearest"
        )
    
    # Safe std (avoid division by zero)
    std_safe = clim_std.where(clim_std > 0.001)
    
    # Z-score grid
    z = (sla_monthly - clim_mean) / std_safe
    
    # Metrics
    mean_z = float(z.mean(skipna=True))
    mean_sla_cm = float(sla_monthly.mean(skipna=True)) * 100  # meters to cm
    
    # % of pixels with positive SLA (Kelvin wave signature)
    valid = sla_monthly.notnull()
    total_valid = int(valid.sum())
    positive_count = int((sla_monthly > 0).sum(skipna=True))
    pct_positive = positive_count / total_valid * 100 if total_valid > 0 else 0
    
    return mean_z, mean_sla_cm, pct_positive


def update_feature_matrix(season_sla):
    """
    Merge SLA Z-scores into the existing feature matrix CSV.
    """
    if not FEAT_PATH.exists():
        print(f"  ERROR: Feature matrix not found at {FEAT_PATH}", flush=True)
        print(f"  Run composite_score.py first.", flush=True)
        return False
    
    df = pd.read_csv(FEAT_PATH)
    
    # Add sla_z column if missing
    if 'sla_z' not in df.columns:
        df['sla_z'] = np.nan
    if 'sla_cm' not in df.columns:
        df['sla_cm'] = np.nan
    if 'sla_pct_pos' not in df.columns:
        df['sla_pct_pos'] = np.nan
    
    update_count = 0
    for entry in season_sla:
        mask = (df['year'] == entry['year']) & (df['season'] == entry['season'])
        if mask.any() and not np.isnan(entry['sla_z']):
            df.loc[mask, 'sla_z'] = entry['sla_z']
            df.loc[mask, 'sla_cm'] = entry['sla_cm']
            df.loc[mask, 'sla_pct_pos'] = entry['sla_pct_pos']
            update_count += 1
    
    df.to_csv(FEAT_PATH, index=False)
    print(f"\n  Updated {update_count} seasons in {FEAT_PATH.name}", flush=True)
    return True


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("PAEWS SLA PIPELINE", flush=True)
    print("=" * 60, flush=True)
    
    # ---- Step 1: Load SLA data ----
    print("\nStep 1: Loading SLA data...", flush=True)
    ds, sla_var = load_sla_data()
    
    # ---- Step 2: Compute climatology ----
    print("\nStep 2: Computing SLA climatology...", flush=True)
    clim = compute_sla_climatology(ds, sla_var)
    
    # ---- Step 3: Extract SLA Z-scores for all seasons ----
    print("\nStep 3: Extracting SLA features for each season...", flush=True)
    
    # Load ground truth to get all season/year pairs
    gt_path = DATA_EXTERNAL / "imarpe_ground_truth.csv"
    gt = pd.read_csv(gt_path)
    
    season_results = []
    print(f"\n  {'Year':>4} {'S':>1} {'Outcome':>10} {'SLA_Z':>7} {'SLA_cm':>7} {'%Pos':>5}", flush=True)
    print(f"  {'-'*45}", flush=True)
    
    for _, row in gt.iterrows():
        year = int(row['year'])
        season = int(row['season'])
        outcome = row['outcome']
        
        # Decision month
        decision_month = 3 if season == 1 else 10
        
        sla_z, sla_cm, pct_pos = extract_sla_z_for_season(
            ds, sla_var, clim, year, decision_month
        )
        
        season_results.append({
            'year': year,
            'season': season,
            'sla_z': sla_z,
            'sla_cm': sla_cm,
            'sla_pct_pos': pct_pos,
        })
        
        if not np.isnan(sla_z):
            # Flag if this looks like a Kelvin wave
            flag = " ← KELVIN?" if sla_z > 1.0 else ""
            print(f"  {year:>4} {season:>1} {outcome:>10} {sla_z:>+7.2f} {sla_cm:>+7.1f} {pct_pos:>5.0f}%{flag}",
                  flush=True)
        else:
            print(f"  {year:>4} {season:>1} {outcome:>10}    N/A     N/A   N/A", flush=True)
    
    # ---- Step 4: Update feature matrix ----
    print("\nStep 4: Updating feature matrix...", flush=True)
    update_feature_matrix(season_results)
    
    # ---- Summary ----
    print("\n" + "=" * 60, flush=True)
    print("SLA PIPELINE COMPLETE", flush=True)
    print("=" * 60, flush=True)
    
    # Highlight the key seasons
    print("\n  KEY SEASONS TO WATCH:", flush=True)
    for r in season_results:
        if not np.isnan(r['sla_z']):
            if abs(r['sla_z']) > 1.0:
                print(f"  ** {r['year']} S{r['season']}: SLA_Z={r['sla_z']:+.2f} "
                      f"({r['sla_cm']:+.1f} cm, {r['sla_pct_pos']:.0f}% positive)", flush=True)
    
    # 2014 S1 specifically
    s2014 = [r for r in season_results if r['year'] == 2014 and r['season'] == 1]
    if s2014 and not np.isnan(s2014[0]['sla_z']):
        z = s2014[0]['sla_z']
        cm = s2014[0]['sla_cm']
        print(f"\n  2014 S1 (GHOST MISS): SLA_Z={z:+.2f} ({cm:+.1f} cm)", flush=True)
        if z > 0.5:
            print(f"  → Kelvin wave CONFIRMED. This fixes the 0.11 probability miss.", flush=True)
        else:
            print(f"  → SLA signal weak. May need finer temporal resolution (weekly).", flush=True)
    
    print(f"\n  Now rerun: python composite_score.py", flush=True)
    print("=" * 60, flush=True)
    
    ds.close()
