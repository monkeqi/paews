"""
PAEWS GODAS Thermocline Pipeline
==================================
Adds subsurface ocean intelligence to PAEWS by computing the depth of the
20°C isotherm (Z20) from NOAA's GODAS reanalysis. A depressed thermocline
(deeper Z20) indicates subsurface warming — the signal that caught the
2014 S1 Kelvin wave the surface model missed.

Data source:
    NOAA PSL THREDDS (OPeNDAP):
    https://psl.noaa.gov/thredds/dodsC/Datasets/godas/potmp.YYYY.nc

    NOT ERDDAP. GODAS is served via THREDDS/OPeNDAP only.

What it does:
    1. Downloads GODAS potential temperature via OPeNDAP (1980-2025)
    2. Extracts Peru coastal box (5-15°S, 85-76°W)
    3. Computes Z20 (depth where T=20°C / 293.15K) at each grid point per month
       NOTE: GODAS stores temperature in Kelvin, not Celsius
    4. Builds monthly Z20 climatology (2003-2022 baseline, matching Chl)
    5. Computes Z20 anomaly Z-scores for all 32 PAEWS seasons
    6. Saves z20_z column to feature matrix

Timing / leakage:
    Decision month = 3 (S1) or 10 (S2)
    GODAS lag = 1 month before decision (Feb for S1, Sep for S2)
    GODAS is a monthly reanalysis product — Feb data available by mid-March.
    No leakage: the subsurface state is known before the season decision.

Physical justification:
    - Depressed thermocline → warm water propagating coastward (Kelvin wave)
    - Suppresses upwelling nutrients even before SST responds
    - 2014 S1: Z20 was anomalously deep (Kelvin wave) but SST was still cool
    - Expected coefficient: POSITIVE (deeper Z20 anomaly → more disruption risk)

Usage:
    cd C:\\Users\\josep\\Documents\\paews
    conda activate paews
    python scripts/godas_thermocline.py

Requirements:
    - xarray with netCDF4/OPeNDAP support (already in paews env)
    - Internet connection for OPeNDAP download
    - ~10 min for first run (downloads ~23 years of data)
"""

import sys
print("PAEWS GODAS Thermocline Pipeline starting...", flush=True)

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

print("Imports done", flush=True)

# === PATHS ===
BASE_DIR = Path("c:/Users/josep/Documents/paews")
DATA_EXTERNAL = BASE_DIR / "data" / "external"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
FEAT_PATH = DATA_EXTERNAL / "paews_feature_matrix.csv"
GT_PATH = DATA_EXTERNAL / "imarpe_ground_truth.csv"

# Output files
GODAS_CACHE = DATA_EXTERNAL / "godas_z20_timeseries.csv"
GODAS_CLIM = DATA_PROCESSED / "godas_z20_climatology.csv"

# === GODAS CONFIG ===
# OPeNDAP base URL (THREDDS, NOT ERDDAP)
GODAS_BASE_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/godas/pottmp.{year}.nc"

# Peru coastal box — matches SST/Chl domain
# Slightly wider longitude to capture offshore thermocline variability
LAT_SOUTH = -15.0
LAT_NORTH = -5.0
LON_WEST = -85.0  # further offshore to see Kelvin wave approach
LON_EAST = -76.0  # coast

# Z20 = depth where temperature = 20°C
# This isotherm tracks the thermocline in the tropical/subtropical Pacific
TARGET_TEMP = 293.15  # 20°C in Kelvin (GODAS stores temperature in K)

# Climatology baseline (match Chl: 2003-2022)
CLIM_START = 2003
CLIM_END = 2022

# Years to process (need 1 year before first season for lagged features)
PROCESS_START = 2003  # climatology needs this
PROCESS_END = 2025    # latest available (GODAS updated annually in January)

# GODAS lag: use month BEFORE decision month
# S1 decision_month=3 → use Feb (month 2)
# S2 decision_month=10 → use Sep (month 9)
GODAS_LAG = {1: 2, 2: 9}  # season → GODAS month


def download_godas_year(year):
    """
    Download one year of GODAS potential temperature via OPeNDAP.
    Subsets to Peru coastal box and relevant depth levels.

    Returns xarray Dataset or None if year not available.
    """
    url = GODAS_BASE_URL.format(year=year)
    print(f"  Fetching {year}... ", end="", flush=True)

    try:
        ds = xr.open_dataset(url)

        # On first successful load, print file structure for debugging
        if year == PROCESS_START:
            print(f"\n    Variables: {list(ds.data_vars)}", flush=True)
            print(f"    Coords: {list(ds.coords)}", flush=True)
            print(f"    Dims: {dict(ds.dims)}", flush=True)
            if 'level' in ds.coords:
                levels = ds.level.values
                print(f"    Levels (first 10): {levels[:10]}", flush=True)

        # GODAS coordinates:
        #   level: depth in meters (40 levels: 5, 15, 25, ... ~4478)
        #   lat: latitude (-74.5 to 64.5)
        #   lon: longitude (0.5 to 359.5, 0-360 convention)
        #   time: monthly

        # Detect variable name (pottmp or potmp or pottmp4)
        temp_var = None
        for candidate in ['pottmp', 'potmp', 'pottmp4', 'temperature']:
            if candidate in ds.data_vars:
                temp_var = candidate
                break
        if temp_var is None:
            print(f"FAILED — no temperature variable found in {list(ds.data_vars)}", flush=True)
            ds.close()
            return None

        # Convert our lon range to 0-360
        lon_west_360 = LON_WEST + 360  # -85 → 275
        lon_east_360 = LON_EAST + 360  # -76 → 284

        # Subset spatially — only depths relevant for Z20 (5-300m)
        ds_sub = ds.sel(
            lat=slice(LAT_SOUTH, LAT_NORTH),
            lon=slice(lon_west_360, lon_east_360),
            level=slice(5, 300)
        )

        # Store the variable name for downstream use
        ds_sub.attrs['_temp_var'] = temp_var

        potmp = ds_sub[temp_var]
        print(f"OK — {potmp.sizes['time']} months, "
              f"{potmp.sizes['level']} levels, "
              f"{potmp.sizes['lat']}x{potmp.sizes['lon']} grid", flush=True)

        return ds_sub

    except Exception as e:
        print(f"FAILED — {e}", flush=True)
        return None


def compute_z20_single(temp_profile, depths):
    """
    Compute depth of 20°C isotherm from a single temperature profile.
    Note: GODAS stores temperature in Kelvin. TARGET_TEMP = 293.15 K = 20°C.

    Uses linear interpolation between depth levels.
    Returns NaN if:
        - Profile is all NaN
        - 20°C is above the shallowest level (surface too cold)
        - 20°C is below the deepest level (very warm column)

    Args:
        temp_profile: 1D array of temperatures at each depth level
        depths: 1D array of depth values (meters)

    Returns:
        z20: depth in meters where T = 20°C, or NaN
    """
    # Remove NaN levels
    valid = ~np.isnan(temp_profile)
    if valid.sum() < 2:
        return np.nan

    t_valid = temp_profile[valid]
    d_valid = depths[valid]

    # Check if 20°C is within the profile range
    t_max = t_valid.max()
    t_min = t_valid.min()

    if TARGET_TEMP > t_max:
        # Entire column is colder than 20°C — Z20 is above the surface
        # This means thermocline is very shallow (strong upwelling)
        # Return the shallowest depth as a floor
        return float(d_valid[0])

    if TARGET_TEMP < t_min:
        # Entire column is warmer than 20°C — Z20 is below our range
        # This means thermocline is very deep (strong El Niño)
        # Return the deepest depth as a ceiling
        return float(d_valid[-1])

    # Temperature decreases with depth in the upper ocean
    # Interpolate: find depth where T = 20
    try:
        # interp1d expects monotonically increasing x
        # Temperature decreases with depth, so we need to handle that
        # Use depth as x, temperature as y, then invert
        f_interp = interp1d(t_valid[::-1], d_valid[::-1],
                           kind='linear', bounds_error=False)
        z20 = float(f_interp(TARGET_TEMP))
        return z20
    except Exception:
        return np.nan


def compute_z20_field(ds_month):
    """
    Compute Z20 at every grid point for a single month.

    Args:
        ds_month: xarray Dataset for one month, with potmp(level, lat, lon)

    Returns:
        z20_field: 2D array (lat, lon) of Z20 depths in meters
    """
    # Detect variable name
    temp_var = ds_month.attrs.get('_temp_var', 'pottmp')
    for candidate in ['pottmp', 'potmp', 'pottmp4', 'temperature']:
        if candidate in ds_month.data_vars:
            temp_var = candidate
            break

    potmp = ds_month[temp_var].squeeze()

    # Handle case where potmp might have extra dims
    if 'time' in potmp.dims:
        potmp = potmp.isel(time=0)

    depths = potmp.level.values
    lats = potmp.lat.values
    lons = potmp.lon.values

    z20_field = np.full((len(lats), len(lons)), np.nan)

    for i in range(len(lats)):
        for j in range(len(lons)):
            profile = potmp.values[:, i, j]
            z20_field[i, j] = compute_z20_single(profile, depths)

    return z20_field, lats, lons


def process_all_years():
    """
    Download GODAS data and compute Z20 for all years.
    Saves monthly Z20 spatial means to a CSV cache.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"STEP 1: Download GODAS and compute Z20 ({PROCESS_START}-{PROCESS_END})", flush=True)
    print(f"{'='*60}", flush=True)

    records = []

    for year in range(PROCESS_START, PROCESS_END + 1):
        ds = download_godas_year(year)
        if ds is None:
            continue

        times = pd.DatetimeIndex(ds.time.values)

        for t_idx, t in enumerate(times):
            month = t.month

            # Extract this month
            ds_month = ds.isel(time=t_idx)
            z20_field, lats, lons = compute_z20_field(ds_month)

            # Spatial statistics
            z20_mean = float(np.nanmean(z20_field))
            z20_std = float(np.nanstd(z20_field))
            z20_min = float(np.nanmin(z20_field)) if not np.all(np.isnan(z20_field)) else np.nan
            z20_max = float(np.nanmax(z20_field)) if not np.all(np.isnan(z20_field)) else np.nan
            valid_pct = float(np.sum(~np.isnan(z20_field))) / z20_field.size * 100

            records.append({
                'year': year,
                'month': month,
                'z20_mean': z20_mean,
                'z20_std': z20_std,
                'z20_min': z20_min,
                'z20_max': z20_max,
                'valid_pct': valid_pct,
            })

        ds.close()

    df = pd.DataFrame(records)
    df.to_csv(GODAS_CACHE, index=False)
    print(f"\n  Saved Z20 timeseries: {GODAS_CACHE}", flush=True)
    print(f"  Total months: {len(df)}", flush=True)

    if len(df) == 0:
        print("  ERROR: No data downloaded. Check internet connection and URL.", flush=True)
        print(f"  URL pattern: {GODAS_BASE_URL}", flush=True)
        sys.exit(1)

    print(f"  Year range: {df['year'].min()}-{df['year'].max()}", flush=True)

    return df


def build_climatology(df_z20):
    """
    Build monthly Z20 climatology from baseline period.
    Matches the 2003-2022 baseline used for SST and Chl.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"STEP 2: Build Z20 climatology ({CLIM_START}-{CLIM_END})", flush=True)
    print(f"{'='*60}", flush=True)

    baseline = df_z20[(df_z20['year'] >= CLIM_START) & (df_z20['year'] <= CLIM_END)]

    clim = baseline.groupby('month')['z20_mean'].agg(['mean', 'std', 'count']).reset_index()
    clim.columns = ['month', 'z20_clim_mean', 'z20_clim_std', 'z20_clim_count']

    # Enforce minimum std to avoid division by near-zero
    clim['z20_clim_std'] = clim['z20_clim_std'].clip(lower=1.0)

    clim.to_csv(GODAS_CLIM, index=False)
    print(f"  Saved: {GODAS_CLIM}", flush=True)

    print(f"\n  {'Month':>5} {'Mean Z20':>10} {'Std':>8} {'N':>4}", flush=True)
    print(f"  {'-'*30}", flush=True)
    for _, row in clim.iterrows():
        print(f"  {int(row['month']):>5} {row['z20_clim_mean']:>10.1f}m "
              f"{row['z20_clim_std']:>7.1f}m {int(row['z20_clim_count']):>4}", flush=True)

    # Flag decision months
    for m in [2, 9]:
        row = clim[clim['month'] == m].iloc[0]
        label = "S1 (Feb)" if m == 2 else "S2 (Sep)"
        print(f"\n  {label}: mean Z20 = {row['z20_clim_mean']:.1f}m ± "
              f"{row['z20_clim_std']:.1f}m", flush=True)

    return clim


def compute_season_z20(df_z20, clim):
    """
    Compute Z20 Z-scores for all 32 PAEWS seasons.
    Uses 1-month lag: Feb for S1 (decision month 3), Sep for S2 (decision month 10).

    Positive Z-score = thermocline deeper than normal = subsurface warming = risk.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"STEP 3: Compute Z20 Z-scores for all seasons", flush=True)
    print(f"{'='*60}", flush=True)

    gt = pd.read_csv(GT_PATH)

    clim_lookup = {}
    for _, row in clim.iterrows():
        clim_lookup[int(row['month'])] = (row['z20_clim_mean'], row['z20_clim_std'])

    z20_lookup = {}
    for _, row in df_z20.iterrows():
        z20_lookup[(int(row['year']), int(row['month']))] = row['z20_mean']

    results = []
    print(f"\n  {'Year':>4} {'S':>1} {'Outcome':>10} {'GODAS_mo':>8} {'Z20':>8} "
          f"{'Z20_z':>7} {'Nino12':>7}", flush=True)
    print(f"  {'-'*55}", flush=True)

    fm = pd.read_csv(FEAT_PATH)

    for _, row in gt.iterrows():
        year = int(row['year'])
        season = int(row['season'])
        outcome = row['outcome']

        godas_month = GODAS_LAG[season]
        godas_year = year  # same year for both S1 (Feb) and S2 (Sep)

        z20_val = z20_lookup.get((godas_year, godas_month), np.nan)
        clim_mean, clim_std = clim_lookup.get(godas_month, (np.nan, np.nan))

        if not np.isnan(z20_val) and not np.isnan(clim_mean):
            z20_z = (z20_val - clim_mean) / clim_std
        else:
            z20_z = np.nan

        # Get nino12_t1 for context
        fm_row = fm[(fm['year'] == year) & (fm['season'] == season)]
        nino = float(fm_row['nino12_t1'].iloc[0]) if len(fm_row) > 0 else np.nan

        results.append({
            'year': year,
            'season': season,
            'godas_month': godas_month,
            'z20_raw': z20_val,
            'z20_z': z20_z,
        })

        z20_str = f"{z20_val:.1f}m" if not np.isnan(z20_val) else "N/A"
        z20z_str = f"{z20_z:+.3f}" if not np.isnan(z20_z) else "N/A"
        nino_str = f"{nino:+.2f}" if not np.isnan(nino) else "N/A"
        flag = " ← DISRUPTED" if outcome in ['DISRUPTED', 'CANCELLED'] else ""

        print(f"  {year:>4} {season:>1} {outcome:>10} "
              f"{'%d-%02d' % (godas_year, godas_month):>8} "
              f"{z20_str:>8} {z20z_str:>7} {nino_str:>7}{flag}", flush=True)

    return results


def update_feature_matrix(season_z20):
    """Add z20_z column to the feature matrix."""
    print(f"\n{'='*60}", flush=True)
    print(f"STEP 4: Update feature matrix", flush=True)
    print(f"{'='*60}", flush=True)

    df = pd.read_csv(FEAT_PATH)

    # Add or update z20_z column
    if 'z20_z' not in df.columns:
        df['z20_z'] = np.nan

    update_count = 0
    for entry in season_z20:
        mask = (df['year'] == entry['year']) & (df['season'] == entry['season'])
        if mask.any() and not np.isnan(entry['z20_z']):
            df.loc[mask, 'z20_z'] = entry['z20_z']
            update_count += 1

    df.to_csv(FEAT_PATH, index=False)
    print(f"  Updated {update_count}/{len(season_z20)} seasons with z20_z", flush=True)
    print(f"  Saved: {FEAT_PATH}", flush=True)

    # Sanity check: show correlation with existing features
    df_valid = df.dropna(subset=['z20_z', 'sst_z', 'chl_z', 'nino12_t1'])
    if len(df_valid) > 5:
        print(f"\n  Correlation check (n={len(df_valid)}):", flush=True)
        for feat in ['sst_z', 'chl_z', 'nino12_t1']:
            r = df_valid['z20_z'].corr(df_valid[feat])
            flag = " ⚠ HIGH" if abs(r) > 0.8 else ""
            print(f"    z20_z vs {feat}: r = {r:+.3f}{flag}", flush=True)

        # Check target correlation
        r_target = df_valid['z20_z'].corr(df_valid['target'])
        print(f"    z20_z vs target:   r = {r_target:+.3f}", flush=True)

    return df


def validate_2014_s1(season_z20):
    """
    Specifically check whether the 2014 S1 Z20 shows the expected
    subsurface signal that the surface model missed.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"VALIDATION: 2014 S1 — The Ghost Miss", flush=True)
    print(f"{'='*60}", flush=True)

    entry_2014 = [e for e in season_z20 if e['year'] == 2014 and e['season'] == 1]
    if not entry_2014:
        print("  No data for 2014 S1", flush=True)
        return

    e = entry_2014[0]
    print(f"  Z20 anomaly (Feb 2014): {e['z20_z']:+.3f}" if not np.isnan(e['z20_z'])
          else "  Z20 anomaly: N/A", flush=True)
    print(f"  Z20 depth (Feb 2014):   {e['z20_raw']:.1f}m" if not np.isnan(e['z20_raw'])
          else "  Z20 depth: N/A", flush=True)
    print(flush=True)

    if not np.isnan(e['z20_z']):
        if e['z20_z'] > 0.5:
            print("  ✓ CONFIRMED: Thermocline was deeper than normal.", flush=True)
            print("    Subsurface Kelvin wave signal detected.", flush=True)
            print("    This is what the surface model (SST/Chl) couldn't see.", flush=True)
        elif e['z20_z'] > 0:
            print("  ~ WEAK SIGNAL: Thermocline was slightly deeper than normal.", flush=True)
            print("    Some subsurface warming but not dramatic.", flush=True)
        else:
            print("  ✗ UNEXPECTED: Thermocline was NOT deeper than normal.", flush=True)
            print("    The subsurface hypothesis may not explain 2014 S1.", flush=True)
            print("    Review the spatial pattern — signal may be regional.", flush=True)

    # Also check a few normal seasons for contrast
    print(f"\n  Context — normal S1 seasons:", flush=True)
    for yr in [2013, 2018, 2021]:
        entry = [e for e in season_z20 if e['year'] == yr and e['season'] == 1]
        if entry:
            e = entry[0]
            z_str = f"{e['z20_z']:+.3f}" if not np.isnan(e['z20_z']) else "N/A"
            print(f"    {yr} S1 (NORMAL): z20_z = {z_str}", flush=True)


def print_2026_s1_value(season_z20, df_z20):
    """Print the Z20 value for 2026 S1 prediction input."""
    print(f"\n{'='*60}", flush=True)
    print(f"2026 S1: GODAS Z20 Feature Value", flush=True)
    print(f"{'='*60}", flush=True)

    entry = [e for e in season_z20 if e['year'] == 2026 and e['season'] == 1]
    if entry and not np.isnan(entry[0]['z20_z']):
        e = entry[0]
        print(f"  z20_z = {e['z20_z']:+.3f}", flush=True)
        print(f"  Z20 depth = {e['z20_raw']:.1f}m (Feb 2026)", flush=True)
        print(f"\n  To use in prediction:", flush=True)
        print(f"    features['z20_z'] = {e['z20_z']:.6f}", flush=True)
    else:
        print("  ⚠ Feb 2026 GODAS data not yet available.", flush=True)
        print("  GODAS files are updated annually (~January for prior year).", flush=True)
        print("  The latest file (pottmp.2025.nc) covers through Dec 2025.", flush=True)

        # Show latest available months as potential proxies
        if len(df_z20) > 0:
            latest = df_z20.tail(6)
            print(f"\n  Latest available Z20 values (potential proxies):", flush=True)
            # Need to compute z-scores for these
            clim_df = pd.read_csv(GODAS_CLIM) if GODAS_CLIM.exists() else None
            if clim_df is not None:
                clim_lookup = {}
                for _, row in clim_df.iterrows():
                    clim_lookup[int(row['month'])] = (row['z20_clim_mean'], row['z20_clim_std'])

                for _, row in latest.iterrows():
                    m = int(row['month'])
                    if m in clim_lookup:
                        cm, cs = clim_lookup[m]
                        z = (row['z20_mean'] - cm) / cs
                        print(f"    {int(row['year'])}-{m:02d}: Z20={row['z20_mean']:.1f}m, "
                              f"z20_z={z:+.3f}", flush=True)

            # Recommend Dec 2025 or Jan 2025 (same calendar month as Feb proxy)
            print(f"\n  Recommendation: Use Dec 2025 Z20 as proxy for 2026 S1.", flush=True)
            print(f"  The subsurface state changes slowly — a 2-month proxy is", flush=True)
            print(f"  more defensible than the 3-month stale Chl proxy you", flush=True)
            print(f"  already use.", flush=True)


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("PAEWS GODAS THERMOCLINE PIPELINE", flush=True)
    print("Adding subsurface ocean intelligence (Z20 isotherm depth)", flush=True)
    print("=" * 60, flush=True)

    # ── Step 1: Download and compute Z20 ──
    if GODAS_CACHE.exists():
        df_check = pd.read_csv(GODAS_CACHE)
        if len(df_check) > 0:
            print(f"\n  Found cached Z20 timeseries: {GODAS_CACHE}", flush=True)
            print(f"  Delete this file to re-download from GODAS.", flush=True)
            df_z20 = df_check
            print(f"  Loaded {len(df_z20)} months ({df_z20['year'].min()}-{df_z20['year'].max()})",
                  flush=True)
        else:
            print(f"\n  Cache file is empty (previous failed run). Re-downloading.", flush=True)
            GODAS_CACHE.unlink()
            df_z20 = process_all_years()
    else:
        df_z20 = process_all_years()

    # ── Step 2: Build climatology ──
    clim = build_climatology(df_z20)

    # ── Step 3: Compute season Z20 Z-scores ──
    season_z20 = compute_season_z20(df_z20, clim)

    # ── Step 4: Update feature matrix ──
    update_feature_matrix(season_z20)

    # ── Step 5: Validate 2014 S1 ──
    validate_2014_s1(season_z20)

    # ── Step 6: 2026 S1 value ──
    print_2026_s1_value(season_z20, df_z20)

    # ── Summary ──
    print(f"\n{'='*60}", flush=True)
    print(f"GODAS PIPELINE COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Z20 timeseries:  {GODAS_CACHE}", flush=True)
    print(f"  Z20 climatology: {GODAS_CLIM}", flush=True)
    print(f"  Feature matrix:  {FEAT_PATH} (z20_z column added)", flush=True)
    print(f"\n  Next: python scripts/test_godas_feature.py", flush=True)
    print(f"  Then: python scripts/predict_2026_s1.py (add z20_z)", flush=True)
    print(f"{'='*60}", flush=True)
