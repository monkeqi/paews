"""
PAEWS Gap Filler: Copernicus Chlorophyll → Missing Composite Scores
====================================================================
Uses the Copernicus GlobColour merged chlorophyll (2022-2023) to fill
the missing seasons in our feature matrix, then reruns validation.

The Copernicus data is monthly, multi-sensor (MODIS+VIIRS+OLCI), 
gap-filled L4 at 4km. Our climatology is from MODIS 8-day composites.
We compute Z-scores by interpolating the climatology to the Copernicus
grid and using log10-space statistics.
"""

import sys
print("PAEWS Gap Filler starting...", flush=True)

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
DATA_SST = BASE_DIR / "data" / "baseline_v2"

MIN_CHL_OBS = 10


def load_copernicus_chl():
    """Load the Copernicus merged chlorophyll."""
    path = DATA_EXTERNAL / "chl_copernicus_2022_2023.nc"
    ds = xr.open_dataset(path)
    print(f"  Copernicus Chl: {ds.sizes['time']} months, "
          f"lat {float(ds.latitude.min()):.1f} to {float(ds.latitude.max()):.1f}, "
          f"lon {float(ds.longitude.min()):.1f} to {float(ds.longitude.max()):.1f}", flush=True)
    return ds


def load_chl_climatology():
    """Load our MODIS-based chlorophyll climatology."""
    path = DATA_PROCESSED / "chl_climatology_v2.nc"
    ds = xr.open_dataset(path)
    print(f"  Chl climatology loaded: {path.name}", flush=True)
    return ds


def load_sst_climatology():
    """Load SST climatology."""
    path = DATA_PROCESSED / "sst_climatology_v2.nc"
    if not path.exists():
        path = DATA_PROCESSED / "sst_climatology.nc"
    ds = xr.open_dataset(path)
    print(f"  SST climatology loaded: {path.name}", flush=True)
    return ds


def compute_chl_z_from_copernicus(cop_ds, clim_ds, year, month):
    """
    Compute mean Chl Z-score from Copernicus monthly data.
    
    Handles grid mismatch by interpolating climatology to Copernicus grid.
    """
    # Get Copernicus snapshot for target month
    time_sel = cop_ds.sel(time=f"{year}-{month:02d}")
    if len(time_sel.time) == 0:
        return np.nan, np.nan
    
    chl_snap = time_sel["CHL"].isel(time=0).squeeze()
    
    # Log transform
    chl_log = np.log10(chl_snap.where(chl_snap > 0))
    
    # Get climatology for this month
    clim_mean = clim_ds["chl_log_mean"].sel(month=month)
    clim_std = clim_ds["chl_log_std"].sel(month=month)
    obs_count = clim_ds["chl_obs_count"].sel(month=month)
    
    # Interpolate climatology to Copernicus grid
    # Copernicus uses 'latitude'/'longitude', climatology uses 'latitude'/'longitude' or 'lat'/'lon'
    clim_lat_name = [d for d in clim_mean.dims if 'lat' in d.lower()][0]
    clim_lon_name = [d for d in clim_mean.dims if 'lon' in d.lower()][0]
    
    clim_mean_interp = clim_mean.interp(
        **{clim_lat_name: chl_log.latitude, clim_lon_name: chl_log.longitude},
        method="nearest"
    )
    clim_std_interp = clim_std.interp(
        **{clim_lat_name: chl_log.latitude, clim_lon_name: chl_log.longitude},
        method="nearest"
    )
    obs_count_interp = obs_count.interp(
        **{clim_lat_name: chl_log.latitude, clim_lon_name: chl_log.longitude},
        method="nearest"
    )
    
    # Compute Z-scores
    std_safe = clim_std_interp.where(clim_std_interp > 0.01)
    z = (chl_log - clim_mean_interp) / std_safe
    z = z.where(obs_count_interp >= MIN_CHL_OBS)
    
    mean_z = float(z.mean(skipna=True))
    
    # Low chl percentage
    valid_count = int(z.notnull().sum())
    lchl_count = int((z < -1.28).sum(skipna=True))
    lchl_pct = lchl_count / valid_count * 100 if valid_count > 0 else 0
    
    return mean_z, lchl_pct


def compute_sst_z(year, month, sst_clim):
    """Compute SST Z-score for a year/month."""
    sst_path = DATA_SST / f"sst_{year}.nc"
    if not sst_path.exists():
        return np.nan, np.nan, np.nan
    
    ds = xr.open_dataset(sst_path)
    time_idx = pd.DatetimeIndex(ds["time"].values)
    month_data = ds.sel(time=time_idx.month == month)
    
    if len(month_data.time) == 0:
        ds.close()
        return np.nan, np.nan, np.nan
    
    sst_snap = month_data["sst"].isel(time=-1).squeeze()
    if 'zlev' in sst_snap.dims:
        sst_snap = sst_snap.isel(zlev=0)
    
    clim_mean = sst_clim["sst_mean"].sel(month=month)
    clim_std = sst_clim["sst_std"].sel(month=month)
    std_safe = clim_std.where(clim_std > 0.01)
    
    z = (sst_snap - clim_mean) / std_safe
    mean_z = float(z.mean(skipna=True))
    
    # MHW pct
    valid = int(z.notnull().sum())
    mhw_count = int((z > 1.28).sum(skipna=True))
    mhw_pct = mhw_count / valid * 100 if valid > 0 else 0
    
    # Bio threshold (>23°C)
    valid_sst = sst_snap.notnull()
    above_23 = (sst_snap > 23.0) & valid_sst
    total_valid = int(valid_sst.sum())
    bio_pct = float(above_23.sum()) / total_valid * 100 if total_valid > 0 else 0
    
    ds.close()
    return mean_z, mhw_pct, bio_pct


def get_nino12(year, month, lag=1):
    """Get Niño 1+2 with lag."""
    nino_path = DATA_EXTERNAL / "nino_indices_monthly.csv"
    if not nino_path.exists():
        return np.nan
    
    df = pd.read_csv(nino_path)
    target_year = year
    target_month = month - lag
    if target_month <= 0:
        target_month += 12
        target_year -= 1
    
    row = df[(df['year'] == target_year) & (df['month'] == target_month)]
    if len(row) == 0:
        return np.nan
    return float(row['nino12_anom'].iloc[0])


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("PAEWS GAP FILLER: 2022-2023 Missing Seasons", flush=True)
    print("=" * 60, flush=True)
    
    # Load data
    print("\nStep 1: Loading data...", flush=True)
    cop_chl = load_copernicus_chl()
    chl_clim = load_chl_climatology()
    sst_clim = load_sst_climatology()
    
    # Define missing seasons
    # Season 1 decision month = March, Season 2 decision month = October
    missing = [
        {'year': 2022, 'season': 2, 'decision_month': 10, 'outcome': 'DISRUPTED', 'target': 1},
        {'year': 2023, 'season': 1, 'decision_month': 3,  'outcome': 'CANCELLED', 'target': 1},
        {'year': 2023, 'season': 2, 'decision_month': 10, 'outcome': 'DISRUPTED', 'target': 1},
    ]
    
    # Compute features
    print("\nStep 2: Computing features for missing seasons...", flush=True)
    results = []
    
    for m in missing:
        year = m['year']
        month = m['decision_month']
        
        print(f"\n  {year} S{m['season']} ({m['outcome']}) — decision month: {month}", flush=True)
        
        # Chl Z from Copernicus
        chl_z, lchl_pct = compute_chl_z_from_copernicus(cop_chl, chl_clim, year, month)
        print(f"    Chl Z (Copernicus): {chl_z:+.2f}  LChl: {lchl_pct:.1f}%" 
              if not np.isnan(chl_z) else "    Chl Z: NO DATA", flush=True)
        
        # SST Z from baseline
        sst_z, mhw_pct, bio_pct = compute_sst_z(year, month, sst_clim)
        print(f"    SST Z: {sst_z:+.2f}  MHW: {mhw_pct:.1f}%  Bio>23: {bio_pct:.0f}%"
              if not np.isnan(sst_z) else "    SST Z: NO DATA", flush=True)
        
        # Niño 1+2
        nino12 = get_nino12(year, month, lag=1)
        print(f"    Niño 1+2 (t-1): {nino12:+.2f}" 
              if not np.isnan(nino12) else "    Niño: NO DATA", flush=True)
        
        # Season flag
        is_summer = 1 if month in [1, 2, 3, 12] else 0
        
        # Hardcoded composite
        if not np.isnan(sst_z) and not np.isnan(chl_z):
            composite = 0.4 * sst_z + 0.4 * (-chl_z) + 0.2 * (nino12 if not np.isnan(nino12) else 0)
        else:
            composite = np.nan
        
        print(f"    Composite (hardcoded): {composite:+.2f}" 
              if not np.isnan(composite) else "    Composite: INCOMPLETE", flush=True)
        
        results.append({
            'year': year,
            'season': m['season'],
            'outcome': m['outcome'],
            'target': m['target'],
            'decision_month': month,
            'is_summer': is_summer,
            'sst_z': sst_z,
            'chl_z': chl_z,
            'mhw_pct': mhw_pct,
            'lchl_pct': lchl_pct,
            'bio_thresh_pct': bio_pct if not np.isnan(sst_z) else np.nan,
            'nino12_t1': nino12,
            'nino12_t2': get_nino12(year, month, lag=2),
            'composite_hard': composite,
        })
    
    # Load existing feature matrix and merge
    print("\nStep 3: Merging with existing feature matrix...", flush=True)
    feat_path = DATA_EXTERNAL / "paews_feature_matrix.csv"
    existing = pd.read_csv(feat_path)
    
    new_rows = pd.DataFrame(results)
    
    # Replace the incomplete rows for these seasons
    for _, new_row in new_rows.iterrows():
        mask = (existing['year'] == new_row['year']) & (existing['season'] == new_row['season'])
        if mask.any():
            for col in new_row.index:
                if col in existing.columns:
                    existing.loc[mask, col] = new_row[col]
            print(f"  Updated {int(new_row['year'])} S{int(new_row['season'])}", flush=True)
    
    # Save updated matrix
    existing.to_csv(feat_path, index=False)
    print(f"\n  Updated feature matrix saved: {feat_path}", flush=True)
    
    # Summary
    print("\n" + "=" * 60, flush=True)
    print("GAP FILL COMPLETE", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        print(f"  {r['year']} S{r['season']} ({r['outcome']}): "
              f"SST_Z={r['sst_z']:+.2f}  Chl_Z={r['chl_z']:+.2f}  "
              f"Niño={r['nino12_t1']:+.2f}  Comp={r['composite_hard']:+.2f}"
              if not np.isnan(r['sst_z']) and not np.isnan(r['chl_z']) and not np.isnan(r['nino12_t1'])
              else f"  {r['year']} S{r['season']} ({r['outcome']}): INCOMPLETE", flush=True)
    
    print(f"\nNow rerun: python composite_score.py", flush=True)
    print("=" * 60, flush=True)
