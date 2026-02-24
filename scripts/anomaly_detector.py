"""
PAEWS Anomaly Detector v2
=========================
Compares current or historical SST against the 2003-2022 climatology
to produce Z-score maps, gradient anomalies, and alert classifications.

Extended bounding box: 0S-16S, 85W-70W (captures coastal El Nino zone)
Baseline: baseline_v2/ (2003-2022, clean — no El Nino contamination)

Usage:
    # Current conditions
    python anomaly_detector.py

    # Backtest a specific year (uses Dec 31 snapshot by default)
    python anomaly_detector.py --backtest 2023

    # Backtest a specific year AND month (e.g., March 2017 coastal El Nino peak)
    python anomaly_detector.py --backtest 2017 --month 3

    # Backtest with a specific day
    python anomaly_detector.py --backtest 2017 --month 3 --day 15
"""

import sys
print("PAEWS Anomaly Detector starting...", flush=True)

import argparse
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from datetime import datetime, timedelta
import requests
import calendar

print("Imports done", flush=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("c:/Users/josep/Documents/paews")
DATA_BASELINE_V2 = BASE_DIR / "data" / "baseline_v2"
DATA_CURRENT = BASE_DIR / "data" / "current"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUTS = BASE_DIR / "outputs"

# Extended bounding box — captures 0S-16S (includes coastal El Nino hotspot)
LAT_MIN, LAT_MAX = -16, 0
LON_MIN, LON_MAX = -85, -70

# Baseline period (clean — excludes El Nino years)
BASELINE_START_YEAR = 2003
BASELINE_END_YEAR = 2022

# Nearshore/offshore split for gradient analysis
# Nearshore: within ~200km of Peru coast (roughly -78 to -70)
# Offshore: further out (-85 to -80)
NEARSHORE_LON = slice(-78, -70)
OFFSHORE_LON = slice(-85, -80)

# =============================================================================
# SEASONAL ALERT CALENDAR
# =============================================================================
# Different thresholds for different parts of the anchovy season cycle.
# The first season (Apr-Jul) is the most important — lower thresholds to
# catch early signals. Secondary season (Nov-Jan) also monitored.
#
# Z-score thresholds:
#   NORMAL:  Z < watch_threshold
#   WATCH:   watch_threshold <= Z < alert_threshold
#   WARNING: alert_threshold <= Z < max_alert
#   ALERT:   Z >= max_alert

SEASON_CALENDAR = {
    # month: (season_name, watch_threshold, alert_threshold, max_alert)
    1:  ("PRE-SEASON WATCH",     1.0, 1.5, 2.0),
    2:  ("PRE-SEASON CRITICAL",  1.0, 1.5, 2.0),
    3:  ("PRE-SEASON CRITICAL",  1.0, 1.5, 2.0),
    4:  ("FIRST SEASON",         0.8, 1.2, 1.8),
    5:  ("FIRST SEASON",         0.8, 1.2, 1.8),
    6:  ("FIRST SEASON",         0.8, 1.2, 1.8),
    7:  ("FIRST SEASON END",     1.0, 1.5, 2.0),
    8:  ("INTER-SEASON",         1.5, 2.0, 2.5),
    9:  ("INTER-SEASON",         1.5, 2.0, 2.5),
    10: ("SECONDARY WATCH",      1.2, 1.8, 2.3),
    11: ("SECONDARY SEASON",     1.0, 1.5, 2.0),
    12: ("SECONDARY WATCH",      1.2, 1.8, 2.3),
}


def get_season_info(month):
    """Return season name and thresholds for a given month."""
    name, watch, alert, max_alert = SEASON_CALENDAR[month]
    return name, watch, alert, max_alert


# =============================================================================
# DATA LOADING
# =============================================================================

def load_climatology():
    """Load the v2 climatology (2003-2022, extended box)."""
    clim_path = DATA_PROCESSED / "sst_climatology_v2.nc"
    if not clim_path.exists():
        # Fall back to v1 if v2 not found
        clim_path = DATA_PROCESSED / "sst_climatology.nc"
        if not clim_path.exists():
            print("  ERROR: No climatology file found. Run compute_climatology.py first.", flush=True)
            return None
        print("  WARNING: Using v1 climatology (old bounding box)", flush=True)
    
    ds = xr.open_dataset(clim_path)
    print(f"  Loaded climatology: {clim_path.name}", flush=True)
    return ds


def load_current_sst():
    """Load the most recent SST download."""
    sst_path = DATA_CURRENT / "sst_current.nc"
    if not sst_path.exists():
        print("  ERROR: No current SST data. Run data_pipeline.py first.", flush=True)
        return None
    ds = xr.open_dataset(sst_path)
    print(f"  Loaded current SST: {ds.dims}", flush=True)
    return ds


def download_year_sst(year):
    """
    Download a full year of SST data for backtesting.
    Saves to baseline_v2/ so it can be reused.
    This handles test years (like 2023) that aren't in the baseline.
    """
    outfile = DATA_BASELINE_V2 / f"sst_{year}.nc"
    if outfile.exists():
        print(f"  Year {year} already downloaded, loading...", flush=True)
        return xr.open_dataset(outfile)
    
    print(f"  Downloading SST for {year} (not in baseline_v2/, fetching from ERDDAP)...", flush=True)
    DATA_BASELINE_V2.mkdir(parents=True, exist_ok=True)
    
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    url = (
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.nc"
        f"?sst[({start_date}T12:00:00Z):1:({end_date}T12:00:00Z)]"
        f"[0:1:0]"
        f"[({LAT_MAX}):1:({LAT_MIN})]"
        f"[({LON_MIN}):1:({LON_MAX})]"
    )
    
    try:
        r = requests.get(url, timeout=300)
        if r.status_code == 200:
            outfile.write_bytes(r.content)
            print(f"  Downloaded {len(r.content)/1024:.0f} KB → {outfile.name}", flush=True)
            return xr.open_dataset(outfile)
        else:
            print(f"  ERROR {r.status_code}: {r.text[:300]}", flush=True)
            return None
    except requests.exceptions.Timeout:
        print(f"  ERROR: Download timed out for {year}", flush=True)
        return None


def load_backtest_sst(year):
    """
    Load SST data for a backtest year.
    First checks baseline_v2/, then tries to download if not found.
    """
    # Check baseline_v2 first
    sst_path = DATA_BASELINE_V2 / f"sst_{year}.nc"
    if sst_path.exists():
        ds = xr.open_dataset(sst_path)
        n_times = ds.sizes.get('time', 0)
        print(f"  Loaded backtest SST for {year}: {n_times} days", flush=True)
        return ds
    
    # Not found — download it (handles test years like 2023, 2024, 2025)
    print(f"  Year {year} not in baseline_v2/, downloading...", flush=True)
    return download_year_sst(year)


# =============================================================================
# ANOMALY DETECTION
# =============================================================================

def compute_zscore(sst_data, clim_ds, target_month):
    """
    Compute Z-scores: (observed - climatological_mean) / climatological_std
    
    Parameters:
        sst_data: xarray Dataset with SST observations
        clim_ds: xarray Dataset with monthly_mean and monthly_std
        target_month: integer 1-12, which month's climatology to compare against
    
    Returns:
        z_scores: xarray DataArray of Z-scores
        sst_snapshot: the SST slice used
    """
    # Get climatological mean and std for the target month
    clim_mean = clim_ds["sst_mean"].sel(month=target_month)
    clim_std = clim_ds["sst_std"].sel(month=target_month)
    
    # Get the SST snapshot — use the last available timestep in the target month
    # Filter to just the target month
    times = sst_data["time"].values
    month_mask = [int(str(t)[:7].split("-")[1]) == target_month 
                  for t in times] if hasattr(times[0], 'astype') else [
                  t.month == target_month for t in 
                  [np.datetime64(t, 'ns').astype('datetime64[ns]').item() if not hasattr(t, 'month') 
                   else t for t in times]]
    
    # Simpler approach: convert times and filter
    import pandas as pd
    time_index = pd.DatetimeIndex(sst_data["time"].values)
    month_data = sst_data.sel(time=time_index.month == target_month)
    
    if len(month_data.time) == 0:
        print(f"  WARNING: No data found for month {target_month}", flush=True)
        # Fall back to nearest available month
        month_data = sst_data.isel(time=-1)
        sst_snapshot = month_data["sst"].squeeze()
    else:
        # Use the last day in the target month (most recent conditions)
        sst_snapshot = month_data["sst"].isel(time=-1).squeeze()
    
    # Align grids — interpolate climatology to match SST resolution if needed
    # OISST is 0.25° and our climatology should match, but let's be safe
    try:
        sst_vals = sst_snapshot.values
        mean_vals = clim_mean.values
        std_vals = clim_std.values
        
        # If shapes don't match, interpolate climatology to SST grid
        if sst_vals.shape != mean_vals.shape:
            print(f"  Grid mismatch: SST {sst_vals.shape} vs clim {mean_vals.shape}, interpolating...", flush=True)
            clim_mean = clim_mean.interp(
                latitude=sst_snapshot.latitude if 'latitude' in sst_snapshot.dims else sst_snapshot.lat,
                longitude=sst_snapshot.longitude if 'longitude' in sst_snapshot.dims else sst_snapshot.lon,
                method="nearest"
            )
            clim_std = clim_std.interp(
                latitude=sst_snapshot.latitude if 'latitude' in sst_snapshot.dims else sst_snapshot.lat,
                longitude=sst_snapshot.longitude if 'longitude' in sst_snapshot.dims else sst_snapshot.lon,
                method="nearest"
            )
    except Exception as e:
        print(f"  Grid alignment note: {e}", flush=True)
    
    # Compute Z-scores
    # Mask where std is 0 or NaN to avoid division errors
    std_safe = clim_std.where(clim_std > 0.01)
    z_scores = (sst_snapshot - clim_mean) / std_safe
    
    return z_scores, sst_snapshot


def classify_alert(z_mean, z_max, pct_exceedance, gradient_anomaly, month):
    """
    Classify the overall alert level based on multiple indicators.
    
    Returns: (status_string, color_code)
    """
    season_name, watch_thresh, alert_thresh, max_alert = get_season_info(month)
    
    # Primary classification based on mean Z-score
    if z_mean >= max_alert:
        status = "ALERT"
    elif z_mean >= alert_thresh:
        status = "WARNING"
    elif z_mean >= watch_thresh:
        status = "WATCH"
    else:
        status = "NORMAL"
    
    # Upgrade if spatial exceedance is very high (>25% of pixels above alert)
    if pct_exceedance > 25 and status == "WARNING":
        status = "ALERT"
    
    # Upgrade if gradient is strongly weakened (upwelling failing)
    if gradient_anomaly < -1.0 and status in ("WATCH", "WARNING"):
        status = "WARNING" if status == "WATCH" else "ALERT"
    
    colors = {
        "NORMAL": "green",
        "WATCH": "gold", 
        "WARNING": "orange",
        "ALERT": "red"
    }
    
    return status, colors[status]


def compute_gradient(sst_snapshot, clim_ds, target_month):
    """
    Compute the nearshore-offshore SST gradient and compare to climatology.
    
    A weakening gradient means upwelling is failing — warm water is reaching
    the coast, which is bad for anchovies.
    """
    # Current gradient
    try:
        nearshore_sst = float(sst_snapshot.sel(
            **{get_lon_name(sst_snapshot): NEARSHORE_LON}).mean(skipna=True))
        offshore_sst = float(sst_snapshot.sel(
            **{get_lon_name(sst_snapshot): OFFSHORE_LON}).mean(skipna=True))
    except Exception:
        # Try alternate dimension names
        lon_name = [d for d in sst_snapshot.dims if 'lon' in d.lower()][0]
        nearshore_sst = float(sst_snapshot.sel(**{lon_name: NEARSHORE_LON}).mean(skipna=True))
        offshore_sst = float(sst_snapshot.sel(**{lon_name: OFFSHORE_LON}).mean(skipna=True))
    
    current_gradient = offshore_sst - nearshore_sst
    
    # Climatological gradient for this month
    clim_mean = clim_ds["sst_mean"].sel(month=target_month)
    try:
        clim_nearshore = float(clim_mean.sel(
            **{get_lon_name(clim_mean): NEARSHORE_LON}).mean(skipna=True))
        clim_offshore = float(clim_mean.sel(
            **{get_lon_name(clim_mean): OFFSHORE_LON}).mean(skipna=True))
    except Exception:
        lon_name = [d for d in clim_mean.dims if 'lon' in d.lower()][0]
        clim_nearshore = float(clim_mean.sel(**{lon_name: NEARSHORE_LON}).mean(skipna=True))
        clim_offshore = float(clim_mean.sel(**{lon_name: OFFSHORE_LON}).mean(skipna=True))
    
    normal_gradient = clim_offshore - clim_nearshore
    gradient_anomaly = current_gradient - normal_gradient
    
    return current_gradient, normal_gradient, gradient_anomaly


def get_lon_name(da):
    """Find the longitude dimension name in a DataArray."""
    for name in da.dims:
        if 'lon' in name.lower():
            return name
    # Check coordinates
    for name in da.coords:
        if 'lon' in name.lower():
            return name
    return 'longitude'


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_anomaly_dashboard(z_scores, sst_snapshot, status, status_color, 
                           z_mean, z_max, pct_exceed, gradient_anomaly,
                           current_gradient, normal_gradient,
                           target_date, month, year_label="current"):
    """Generate the 4-panel anomaly dashboard."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    
    season_name, watch, alert, max_alert = get_season_info(month)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        f"PAEWS SST Anomaly Dashboard — {target_date}\n"
        f"Season: {season_name} | Status: {status}",
        fontsize=14, fontweight='bold',
        color=status_color
    )
    
    # Panel 1: Z-score map
    ax1 = axes[0, 0]
    z_plot = z_scores.plot(ax=ax1, vmin=-3, vmax=3, cmap="RdBu_r",
                           cbar_kwargs={"label": "Z-score (σ)", "shrink": 0.8})
    ax1.set_title("SST Z-Score Anomaly")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    
    # Panel 2: Raw SST
    ax2 = axes[0, 1]
    sst_snapshot.plot(ax=ax2, vmin=14, vmax=30, cmap="RdYlBu_r",
                      cbar_kwargs={"label": "SST (°C)", "shrink": 0.8})
    ax2.set_title("Observed SST")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    
    # Panel 3: Z-score histogram
    ax3 = axes[1, 0]
    z_flat = z_scores.values.flatten()
    z_flat = z_flat[~np.isnan(z_flat)]
    ax3.hist(z_flat, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axvline(watch, color='gold', linestyle='--', linewidth=2, label=f'Watch ({watch}σ)')
    ax3.axvline(alert, color='orange', linestyle='--', linewidth=2, label=f'Alert ({alert}σ)')
    ax3.axvline(max_alert, color='red', linestyle='--', linewidth=2, label=f'Max Alert ({max_alert}σ)')
    ax3.axvline(z_mean, color='black', linestyle='-', linewidth=2, label=f'Mean ({z_mean:.2f}σ)')
    ax3.set_xlabel("Z-score")
    ax3.set_ylabel("Pixel count")
    ax3.set_title("Z-Score Distribution")
    ax3.legend(fontsize=8)
    
    # Panel 4: Summary stats text
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = (
        f"PAEWS ANOMALY DETECTION v2\n"
        f"{'='*40}\n\n"
        f"Date: {target_date}\n"
        f"Season: {season_name}\n"
        f"Bounding Box: 0°S–16°S, 85°W–70°W\n"
        f"Baseline: {BASELINE_START_YEAR}–{BASELINE_END_YEAR}\n\n"
        f"STATUS: {status}\n\n"
        f"Mean Z-score: {z_mean:+.2f}σ\n"
        f"Max Z-score:  {z_max:+.2f}σ\n"
        f"Min Z-score:  {z_scores.min().values:+.2f}σ\n"
        f"Pixels > alert: {pct_exceed:.1f}%\n\n"
        f"Gradient (offshore - nearshore):\n"
        f"  Current:  {current_gradient:+.2f}°C\n"
        f"  Normal:   {normal_gradient:+.2f}°C\n"
        f"  Anomaly:  {gradient_anomaly:+.2f}°C\n"
    )
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    outpath = OUTPUTS / f"sst_anomaly_dashboard_{year_label}.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dashboard saved: {outpath}", flush=True)


def plot_backtest_timeseries(sst_ds, clim_ds, year):
    """
    Generate a full-year time series showing daily Z-scores for a backtest year.
    This is the 'money chart' — shows exactly when the system would have flagged.
    """
    import pandas as pd
    
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    
    times = pd.DatetimeIndex(sst_ds["time"].values)
    daily_z = []
    daily_sst = []
    
    for i, t in enumerate(times):
        month = t.month
        try:
            clim_mean = clim_ds["sst_mean"].sel(month=month)
            clim_std = clim_ds["sst_std"].sel(month=month)
            std_safe = clim_std.where(clim_std > 0.01)
            
            sst_slice = sst_ds["sst"].isel(time=i).squeeze()
            z = ((sst_slice - clim_mean) / std_safe).mean(skipna=True).values
            sst_mean = sst_slice.mean(skipna=True).values
            
            daily_z.append(float(z))
            daily_sst.append(float(sst_mean))
        except Exception:
            daily_z.append(np.nan)
            daily_sst.append(np.nan)
    
    daily_z = np.array(daily_z)
    daily_sst = np.array(daily_sst)
    
    # Build threshold lines (vary by month per seasonal calendar)
    watch_line = np.array([SEASON_CALENDAR[t.month][1] for t in times])
    alert_line = np.array([SEASON_CALENDAR[t.month][2] for t in times])
    max_alert_line = np.array([SEASON_CALENDAR[t.month][3] for t in times])
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"PAEWS Backtest: {year} SST Anomaly Time Series\n"
                 f"Baseline: {BASELINE_START_YEAR}–{BASELINE_END_YEAR} | Box: 0°S–16°S, 85°W–70°W",
                 fontsize=13, fontweight='bold')
    
    # Top panel: Z-scores
    ax1.plot(times, daily_z, color='black', linewidth=0.8, label='Daily mean Z-score')
    ax1.fill_between(times, alert_line, max_alert_line, alpha=0.2, color='orange', label='Warning zone')
    ax1.fill_between(times, max_alert_line, 4.0, alpha=0.3, color='red', label='Alert zone')
    ax1.fill_between(times, watch_line, alert_line, alpha=0.15, color='gold', label='Watch zone')
    ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax1.set_ylabel("Z-score (σ)")
    ax1.set_ylim(-3, 4)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_title("Spatial Mean Z-Score")
    
    # Bottom panel: Raw SST
    ax2.plot(times, daily_sst, color='darkred', linewidth=0.8)
    ax2.set_ylabel("SST (°C)")
    ax2.set_xlabel("Date")
    ax2.set_title("Spatial Mean SST")
    
    plt.tight_layout()
    outpath = OUTPUTS / f"sst_backtest_{year}.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Backtest time series saved: {outpath}", flush=True)


def save_zscore_netcdf(z_scores, year_label):
    """Save Z-score data as NetCDF for later analysis."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    z_ds = z_scores.to_dataset(name="z_score")
    z_ds.attrs["description"] = f"SST anomalies as Z-scores relative to {BASELINE_START_YEAR}-{BASELINE_END_YEAR} climatology v2"
    z_ds.attrs["bounding_box"] = "0S-16S, 85W-70W"
    z_ds.attrs["created"] = datetime.now().isoformat()
    
    outpath = DATA_PROCESSED / f"sst_anomaly_{year_label}.nc"
    z_ds.to_netcdf(outpath)
    size_kb = outpath.stat().st_size / 1024
    print(f"  Z-score data saved: {outpath} ({size_kb:.0f} KB)", flush=True)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAEWS SST Anomaly Detector v2")
    parser.add_argument("--backtest", type=int, default=None,
                        help="Year to backtest (e.g., 2023)")
    parser.add_argument("--month", type=int, default=None,
                        help="Target month 1-12 (default: last month in data)")
    parser.add_argument("--day", type=int, default=None,
                        help="Target day 1-31 (default: last day available)")
    args = parser.parse_args()
    
    is_backtest = args.backtest is not None
    
    if is_backtest:
        year = args.backtest
        month_label = f" month={args.month}" if args.month else ""
        print("=" * 60, flush=True)
        print(f"PAEWS Anomaly Detector v2 - BACKTEST MODE ({year}{month_label})", flush=True)
        print("=" * 60, flush=True)
    else:
        print("=" * 60, flush=True)
        print("PAEWS Anomaly Detector v2 - CURRENT CONDITIONS", flush=True)
        print("=" * 60, flush=True)
    
    # ---- Step 1: Load climatology ----
    print("\nStep 1: Loading climatology...", flush=True)
    clim_ds = load_climatology()
    if clim_ds is None:
        sys.exit(1)
    
    # ---- Step 2: Load SST data ----
    print("\nStep 2: Loading SST data...", flush=True)
    if is_backtest:
        sst_ds = load_backtest_sst(year)
    else:
        sst_ds = load_current_sst()
    
    if sst_ds is None:
        print("  FATAL: Could not load SST data. Exiting.", flush=True)
        sys.exit(1)
    
    # ---- Determine target month ----
    import pandas as pd
    time_index = pd.DatetimeIndex(sst_ds["time"].values)
    
    if args.month:
        target_month = args.month
        # Filter data to target month to find the right snapshot
        month_times = time_index[time_index.month == target_month]
        if len(month_times) == 0:
            print(f"  ERROR: No data for month {target_month} in this dataset.", flush=True)
            sys.exit(1)
        if args.day:
            # Find closest date to requested day
            target_date = pd.Timestamp(year=args.backtest or time_index[-1].year, 
                                        month=target_month, day=args.day)
            closest_idx = np.argmin(np.abs(time_index - target_date))
            latest_date = time_index[closest_idx]
        else:
            latest_date = month_times[-1]
    else:
        latest_date = time_index[-1]
        target_month = latest_date.month
    
    # ---- Step 3: Compute Z-scores ----
    print("\nStep 3: Computing Z-scores...", flush=True)
    z_scores, sst_snapshot = compute_zscore(sst_ds, clim_ds, target_month)
    
    # Statistics
    z_mean = float(z_scores.mean(skipna=True))
    z_max = float(z_scores.max(skipna=True))
    z_min = float(z_scores.min(skipna=True))
    
    season_name, watch, alert_thresh, max_alert = get_season_info(target_month)
    n_exceed = int((z_scores > alert_thresh).sum(skipna=True))
    n_total = int((~np.isnan(z_scores)).sum())
    pct_exceed = (n_exceed / n_total * 100) if n_total > 0 else 0
    
    print(f"  Target date: {latest_date.strftime('%Y-%m-%d')}", flush=True)
    print(f"  Target month: {target_month} ({calendar.month_name[target_month]})", flush=True)
    print(f"  Season mode: {season_name}", flush=True)
    print(f"  Alert threshold: Z >= +{alert_thresh}", flush=True)
    print(f"  Spatial mean Z-score: {z_mean:+.2f}", flush=True)
    print(f"  Max Z-score: {z_max:+.2f}", flush=True)
    print(f"  Min Z-score: {z_min:+.2f}", flush=True)
    print(f"  Pixels exceeding threshold: {n_exceed}/{n_total} ({pct_exceed:.1f}%)", flush=True)
    
    # ---- Step 4: Gradient analysis ----
    print("\nStep 4: Computing gradient anomaly...", flush=True)
    current_gradient, normal_gradient, gradient_anomaly = compute_gradient(
        sst_snapshot, clim_ds, target_month)
    
    print(f"  Month: {calendar.month_abbr[target_month]}", flush=True)
    print(f"  Current gradient:  {current_gradient:+.2f}C (offshore - nearshore)", flush=True)
    print(f"  Normal gradient:   {normal_gradient:+.2f}C", flush=True)
    print(f"  Gradient anomaly:  {gradient_anomaly:+.2f}C", flush=True)
    
    if gradient_anomaly < -1.0:
        print(f"  >>> GRADIENT WARNING: Upwelling significantly weakened <<<", flush=True)
    else:
        print(f"  Upwelling gradient within normal range", flush=True)
    
    # ---- Step 5: Classify alert ----
    status, status_color = classify_alert(z_mean, z_max, pct_exceed, gradient_anomaly, target_month)
    print(f"\n  >>> PAEWS STATUS: {status} <<<", flush=True)
    
    # ---- Step 6: Generate dashboard ----
    print("\nStep 5: Generating anomaly dashboard...", flush=True)
    year_label = str(year) if is_backtest else "current"
    if args.month and is_backtest:
        year_label = f"{year}_m{target_month:02d}"
    
    target_date_str = latest_date.strftime('%Y-%m-%d')
    plot_anomaly_dashboard(
        z_scores, sst_snapshot, status, status_color,
        z_mean, z_max, pct_exceed, gradient_anomaly,
        current_gradient, normal_gradient,
        target_date_str, target_month, year_label
    )
    
    # ---- Step 7: Save Z-score data ----
    save_zscore_netcdf(z_scores, year_label)
    
    # ---- Step 8: Backtest time series ----
    if is_backtest:
        print(f"\nStep 7: Generating {year} backtest time series...", flush=True)
        plot_backtest_timeseries(sst_ds, clim_ds, year)
    
    # ---- Summary ----
    print("\n" + "=" * 60, flush=True)
    print(f"ANOMALY DETECTION v2 COMPLETE", flush=True)
    print(f"  Status: {status}", flush=True)
    print(f"  Mean Z-score: {z_mean:+.2f}", flush=True)
    print(f"  Gradient anomaly: {gradient_anomaly:+.2f}C", flush=True)
    print(f"  Dashboard: outputs/sst_anomaly_dashboard_{year_label}.png", flush=True)
    if is_backtest:
        print(f"  Time series: outputs/sst_backtest_{year}.png", flush=True)
    print("=" * 60, flush=True)
