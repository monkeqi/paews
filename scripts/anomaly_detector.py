"""
PAEWS Phase 2 - Step 3: SST Anomaly Detector
==============================================
Compares current SST data against the monthly climatology to
detect anomalies using Z-scores.

WHAT IS A Z-SCORE?
A Z-score tells you how many standard deviations a value is
from the mean. In plain English:
  Z = 0    -> perfectly normal
  Z = +1   -> warmer than 84% of historical values for this month
  Z = +2   -> warmer than 97.7% (unusual)
  Z = +3   -> warmer than 99.9% (extreme)
  Z = -2   -> colder than 97.7% of historical values

The formula: Z = (current_value - climatological_mean) / climatological_std

For PAEWS, POSITIVE Z-scores mean WARMER than normal, which is
BAD for anchovies (suppressed upwelling, reduced nutrients).
NEGATIVE Z-scores mean COLDER than normal (enhanced upwelling,
usually good for productivity).

ALERT THRESHOLDS (from our seasonal calendar):
  Feb-Mar (MAXIMUM ALERT):   flag at Z >= +1.5
  Apr-Jul (ACTIVE SEASON):   flag at Z >= +2.0
  Aug-Oct (QUIET PERIOD):    flag at Z >= +2.5
  Nov-Jan (SECONDARY WATCH): flag at Z >= +2.0

This script can run on:
  1. Current data (from data/current/sst_current.nc)
  2. Any historical year (for backtesting - e.g., 2023)

OUTPUT:
  outputs/sst_anomaly_map.png      - spatial Z-score map
  outputs/sst_anomaly_summary.png  - dashboard with map + gradient + stats
  data/processed/sst_anomaly.nc    - Z-score data as NetCDF

Usage (current data):
  & C:/Users/josep/miniconda3/Scripts/conda.exe run -n geosentinel python c:/Users/josep/Documents/paews/scripts/anomaly_detector.py

Usage (backtest 2023):
  & C:/Users/josep/miniconda3/Scripts/conda.exe run -n geosentinel python c:/Users/josep/Documents/paews/scripts/anomaly_detector.py --backtest 2023
"""

import sys
print("PAEWS Anomaly Detector starting...", flush=True)

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from datetime import datetime

print("Imports done", flush=True)

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path("c:/Users/josep/Documents/paews")
DATA_CURRENT = BASE_DIR / "data" / "current"
DATA_BASELINE = BASE_DIR / "data" / "baseline"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUTS = BASE_DIR / "outputs"

LAT_MIN, LAT_MAX = -16, -4
LON_MIN, LON_MAX = -82, -70

# Alert thresholds by season
# Keys are month numbers, values are Z-score thresholds
ALERT_THRESHOLDS = {
    1:  2.0,   # Jan - Secondary watch
    2:  1.5,   # Feb - MAXIMUM ALERT (prediction window)
    3:  1.5,   # Mar - MAXIMUM ALERT (prediction window)
    4:  2.0,   # Apr - Active season
    5:  2.0,   # May - Active season
    6:  2.0,   # Jun - Active season
    7:  2.0,   # Jul - Active season
    8:  2.5,   # Aug - Quiet period
    9:  2.5,   # Sep - Quiet period
    10: 2.5,   # Oct - Quiet period
    11: 2.0,   # Nov - Secondary watch
    12: 2.0,   # Dec - Secondary watch
}

SEASON_NAMES = {
    1:  "SECONDARY WATCH",
    2:  "MAXIMUM ALERT",
    3:  "MAXIMUM ALERT",
    4:  "ACTIVE SEASON",
    5:  "ACTIVE SEASON",
    6:  "ACTIVE SEASON",
    7:  "ACTIVE SEASON",
    8:  "QUIET PERIOD",
    9:  "QUIET PERIOD",
    10: "QUIET PERIOD",
    11: "SECONDARY WATCH",
    12: "SECONDARY WATCH",
}


# ============================================================
# STEP 1: LOAD CLIMATOLOGY
# ============================================================
def load_climatology():
    """Load the pre-computed monthly climatology."""
    clim_path = DATA_PROCESSED / "sst_climatology.nc"
    if not clim_path.exists():
        print("ERROR: Climatology file not found!", flush=True)
        print("  Run compute_climatology.py first.", flush=True)
        return None
    clim = xr.open_dataset(clim_path)
    print(f"  Loaded climatology: {clim_path.name}", flush=True)
    return clim


# ============================================================
# STEP 2: LOAD SST DATA (current or backtest)
# ============================================================
def load_current_sst():
    """Load the most recent SST data from the pipeline."""
    sst_path = DATA_CURRENT / "sst_current.nc"
    if not sst_path.exists():
        print("ERROR: Current SST file not found!", flush=True)
        print("  Run data_pipeline.py first.", flush=True)
        return None
    ds = xr.open_dataset(sst_path)
    # Drop altitude/zlev if present
    if "zlev" in ds.dims:
        ds = ds.squeeze("zlev", drop=True)
    if "altitude" in ds.dims:
        ds = ds.squeeze("altitude", drop=True)
    print(f"  Loaded current SST: {ds.sizes['time']} days", flush=True)
    return ds


def load_backtest_sst(year):
    """Load a historical year for backtesting."""
    sst_path = DATA_BASELINE / f"sst_{year}.nc"
    if not sst_path.exists():
        print(f"ERROR: Baseline file for {year} not found!", flush=True)
        return None
    ds = xr.open_dataset(sst_path)
    if "zlev" in ds.dims:
        ds = ds.squeeze("zlev", drop=True)
    if "altitude" in ds.dims:
        ds = ds.squeeze("altitude", drop=True)
    print(f"  Loaded backtest SST for {year}: {ds.sizes['time']} days", flush=True)
    return ds


# ============================================================
# STEP 3: COMPUTE Z-SCORES
# ============================================================
# This is the core computation. For each day of SST data:
#   1. Determine what month it falls in
#   2. Look up that month's climatological mean and std
#   3. Compute Z = (observed - mean) / std
#
# We compute Z-scores for EVERY pixel on EVERY day.
# Then we can average across time to get a summary,
# or look at the most recent day for current conditions.

def compute_zscore(sst_ds, clim_ds):
    """Compute Z-scores for SST data against climatology."""
    print("\nStep 3: Computing Z-scores...", flush=True)

    sst = sst_ds["sst"]

    # Get the month for each time step
    months = sst.time.dt.month

    # Look up the climatological mean and std for each time step's month
    # xarray's .sel() with the month coordinate does this elegantly
    clim_mean = clim_ds["sst_mean"].sel(month=months)
    clim_std = clim_ds["sst_std"].sel(month=months)

    # Compute Z-scores
    # Where std is 0 or NaN (land pixels), Z will be NaN
    zscore = (sst - clim_mean) / clim_std

    # Report summary statistics
    # Use the most recent time step for the "current" assessment
    latest_z = zscore.isel(time=-1)
    latest_date = str(sst.time.values[-1])[:10]
    latest_month = int(sst.time.dt.month.values[-1])

    # Compute spatial statistics (ocean pixels only)
    z_mean = float(latest_z.mean(skipna=True).values)
    z_max = float(latest_z.max(skipna=True).values)
    z_min = float(latest_z.min(skipna=True).values)
    threshold = ALERT_THRESHOLDS[latest_month]
    season = SEASON_NAMES[latest_month]

    # Count pixels exceeding threshold
    n_ocean = int(latest_z.count().values)
    n_alert = int((latest_z >= threshold).sum().values)
    alert_pct = (n_alert / n_ocean * 100) if n_ocean > 0 else 0

    print(f"\n  Latest date: {latest_date}", flush=True)
    print(f"  Season mode: {season}", flush=True)
    print(f"  Alert threshold: Z >= +{threshold}", flush=True)
    print(f"  Spatial mean Z-score: {z_mean:+.2f}", flush=True)
    print(f"  Max Z-score: {z_max:+.2f}", flush=True)
    print(f"  Min Z-score: {z_min:+.2f}", flush=True)
    print(f"  Pixels exceeding threshold: {n_alert}/{n_ocean} ({alert_pct:.1f}%)", flush=True)

    # Determine alert level
    if z_mean >= threshold:
        alert_level = "ALERT"
    elif z_mean >= threshold * 0.75:
        alert_level = "WARNING"
    elif z_mean >= threshold * 0.5:
        alert_level = "WATCH"
    else:
        alert_level = "NORMAL"

    print(f"\n  >>> PAEWS STATUS: {alert_level} <<<", flush=True)

    return zscore, {
        "date": latest_date,
        "month": latest_month,
        "season": season,
        "threshold": threshold,
        "z_mean": z_mean,
        "z_max": z_max,
        "z_min": z_min,
        "n_alert": n_alert,
        "n_ocean": n_ocean,
        "alert_pct": alert_pct,
        "alert_level": alert_level,
    }


# ============================================================
# STEP 4: COMPUTE GRADIENT ANOMALY
# ============================================================
# Compare the current coastal-offshore gradient against the
# climatological gradient for this month. If the gradient has
# weakened, upwelling may be failing.

def compute_gradient_anomaly(sst_ds, clim_ds):
    """Compute how the current upwelling gradient compares to normal."""
    print("\nStep 4: Computing gradient anomaly...", flush=True)

    sst = sst_ds["sst"].isel(time=-1)
    latest_month = int(sst_ds.time.dt.month.values[-1])

    # Current gradient
    nearshore_current = float(sst.sel(longitude=slice(-78, -70)).mean(skipna=True).values)
    offshore_current = float(sst.sel(longitude=slice(-82, -80)).mean(skipna=True).values)
    gradient_current = offshore_current - nearshore_current

    # Climatological gradient for this month
    clim_mean = clim_ds["sst_mean"].sel(month=latest_month)
    nearshore_clim = float(clim_mean.sel(longitude=slice(-78, -70)).mean(skipna=True).values)
    offshore_clim = float(clim_mean.sel(longitude=slice(-82, -80)).mean(skipna=True).values)
    gradient_clim = offshore_clim - nearshore_clim

    # Gradient anomaly (negative means weaker upwelling than normal)
    gradient_anomaly = gradient_current - gradient_clim

    month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    print(f"  Month: {month_names[latest_month]}", flush=True)
    print(f"  Current gradient:  {gradient_current:+.2f}C (offshore - nearshore)", flush=True)
    print(f"  Normal gradient:   {gradient_clim:+.2f}C", flush=True)
    print(f"  Gradient anomaly:  {gradient_anomaly:+.2f}C", flush=True)

    if gradient_anomaly < -1.0:
        print(f"  >>> UPWELLING WEAKENING DETECTED <<<", flush=True)
    elif gradient_anomaly < -0.5:
        print(f"  Upwelling slightly weaker than normal", flush=True)
    else:
        print(f"  Upwelling gradient within normal range", flush=True)

    return {
        "nearshore_current": nearshore_current,
        "offshore_current": offshore_current,
        "gradient_current": gradient_current,
        "nearshore_clim": nearshore_clim,
        "offshore_clim": offshore_clim,
        "gradient_clim": gradient_clim,
        "gradient_anomaly": gradient_anomaly,
    }


# ============================================================
# STEP 5: GENERATE ANOMALY DASHBOARD
# ============================================================
# A single figure with multiple panels showing:
#   1. The Z-score map (where are the anomalies?)
#   2. The actual SST map (what does the ocean look like?)
#   3. Key statistics and alert status

def plot_anomaly_dashboard(zscore_ds, sst_ds, clim_ds, stats, gradient_info, output_suffix=""):
    """Create a multi-panel anomaly dashboard."""
    print("\nStep 5: Generating anomaly dashboard...", flush=True)

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    latest_z = zscore_ds.isel(time=-1)
    latest_sst = sst_ds["sst"].isel(time=-1)
    latest_month = stats["month"]
    clim_mean = clim_ds["sst_mean"].sel(month=latest_month)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"PAEWS SST Anomaly Dashboard - {stats['date']}\n"
        f"Season: {stats['season']} | Status: {stats['alert_level']}",
        fontsize=16, fontweight="bold"
    )

    # Panel 1: Z-score map (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    im1 = latest_z.plot(
        ax=ax1,
        vmin=-3, vmax=3,
        cmap="RdBu_r",
        cbar_kwargs={"label": "Z-score (std deviations from normal)"}
    )
    ax1.set_title(f"SST Z-Score Map\nThreshold: {stats['threshold']:+.1f} SD")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    # Panel 2: Current SST (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    latest_sst.plot(
        ax=ax2,
        vmin=14, vmax=28,
        cmap="RdYlBu_r",
        cbar_kwargs={"label": "SST (C)"}
    )
    ax2.set_title(f"Current SST - {stats['date']}")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    # Panel 3: Climatological mean for this month (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    clim_mean.plot(
        ax=ax3,
        vmin=14, vmax=28,
        cmap="RdYlBu_r",
        cbar_kwargs={"label": "SST (C)"}
    )
    month_names = ["", "January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    ax3.set_title(f"Normal SST - {month_names[latest_month]} Climatology")
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")

    # Panel 4: Text summary (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")

    # Build summary text
    summary_lines = [
        f"PAEWS ANOMALY REPORT",
        f"{'='*35}",
        f"Date: {stats['date']}",
        f"Season: {stats['season']}",
        f"Alert Level: {stats['alert_level']}",
        f"",
        f"SST Z-SCORE STATISTICS",
        f"{'-'*35}",
        f"Spatial Mean:  {stats['z_mean']:+.2f} SD",
        f"Maximum:       {stats['z_max']:+.2f} SD",
        f"Minimum:       {stats['z_min']:+.2f} SD",
        f"Alert Threshold: +{stats['threshold']:.1f} SD",
        f"Pixels in Alert: {stats['n_alert']}/{stats['n_ocean']} ({stats['alert_pct']:.1f}%)",
        f"",
        f"UPWELLING GRADIENT",
        f"{'-'*35}",
        f"Current:  {gradient_info['gradient_current']:+.2f}C",
        f"Normal:   {gradient_info['gradient_clim']:+.2f}C",
        f"Anomaly:  {gradient_info['gradient_anomaly']:+.2f}C",
    ]

    if gradient_info['gradient_anomaly'] < -1.0:
        summary_lines.append(f">>> UPWELLING WEAKENING <<<")
    elif gradient_info['gradient_anomaly'] < -0.5:
        summary_lines.append(f"Upwelling slightly weak")
    else:
        summary_lines.append(f"Upwelling normal")

    summary_text = "\n".join(summary_lines)
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()

    outpath = OUTPUTS / f"sst_anomaly_dashboard{output_suffix}.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dashboard saved: {outpath}", flush=True)


# ============================================================
# STEP 6: SAVE Z-SCORE DATA
# ============================================================
def save_zscore(zscore_ds, stats, output_suffix=""):
    """Save Z-score data as NetCDF for later analysis."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    out_ds = xr.Dataset(
        {"sst_zscore": zscore_ds},
        attrs={
            "title": "PAEWS SST Z-Scores",
            "description": "SST anomalies as Z-scores relative to 2003-2025 climatology",
            "alert_level": stats["alert_level"],
            "analysis_date": stats["date"],
            "created_by": "PAEWS anomaly_detector.py",
        }
    )
    outpath = DATA_PROCESSED / f"sst_anomaly{output_suffix}.nc"
    out_ds.to_netcdf(outpath)
    size_kb = outpath.stat().st_size / 1024
    print(f"  Z-score data saved: {outpath} ({size_kb:.0f} KB)", flush=True)


# ============================================================
# STEP 7: BACKTEST TIME SERIES (for historical years)
# ============================================================
# When backtesting, we want to see how the Z-score evolved
# over time through the year. This generates a time series
# plot showing the spatial mean Z-score day by day, with
# the alert threshold overlaid.

def plot_backtest_timeseries(zscore_ds, sst_ds, year):
    """Plot daily mean Z-score for a full backtest year."""
    print(f"\nStep 7: Generating {year} backtest time series...", flush=True)

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # Compute daily spatial mean Z-score (average across all ocean pixels)
    daily_mean_z = zscore_ds.mean(dim=["latitude", "longitude"], skipna=True)

    # Get dates and months for threshold coloring
    times = sst_ds.time.values
    dates = [str(t)[:10] for t in times]
    months = [int(str(t)[5:7]) for t in times]
    z_values = daily_mean_z.values

    # Get threshold for each day based on its month
    thresholds = [ALERT_THRESHOLDS[m] for m in months]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    fig.suptitle(
        f"PAEWS Backtest: {year} SST Anomaly Time Series\n"
        f"Daily Spatial Mean Z-Score vs Alert Threshold",
        fontsize=14, fontweight="bold"
    )

    # Top panel: Z-score time series
    ax1.plot(range(len(z_values)), z_values, 'b-', linewidth=0.8, alpha=0.7, label="Daily mean Z-score")
    ax1.plot(range(len(thresholds)), thresholds, 'r--', linewidth=1.5, label="Alert threshold")
    ax1.axhline(y=0, color='black', linewidth=0.5)

    # Shade areas where Z exceeds threshold
    z_arr = np.array(z_values)
    t_arr = np.array(thresholds)
    alert_mask = z_arr >= t_arr
    for i in range(len(z_arr)):
        if alert_mask[i]:
            ax1.axvspan(i - 0.5, i + 0.5, alpha=0.3, color='red')

    ax1.set_ylabel("Z-Score (SD from normal)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-4, 6)

    # Add month labels
    month_starts = []
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    current_month = 0
    for i, m in enumerate(months):
        if m != current_month:
            month_starts.append(i)
            current_month = m
    ax1.set_xticks(month_starts)
    ax1.set_xticklabels([month_labels[months[i]-1] for i in month_starts])

    # Bottom panel: Actual SST spatial mean
    sst_daily_mean = sst_ds["sst"].mean(dim=["latitude", "longitude"], skipna=True)
    ax2.plot(range(len(sst_daily_mean)), sst_daily_mean.values, 'darkorange', linewidth=1)
    ax2.set_ylabel("Mean SST (C)")
    ax2.set_xlabel("Month")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(month_starts)
    ax2.set_xticklabels([month_labels[months[i]-1] for i in month_starts])

    plt.tight_layout()
    outpath = OUTPUTS / f"sst_backtest_{year}.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Backtest time series saved: {outpath}", flush=True)


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    # Check if backtest mode
    backtest_year = None
    if len(sys.argv) > 2 and sys.argv[1] == "--backtest":
        try:
            backtest_year = int(sys.argv[2])
        except ValueError:
            print(f"ERROR: Invalid year: {sys.argv[2]}", flush=True)
            sys.exit(1)

    print("=" * 55, flush=True)
    if backtest_year:
        print(f"PAEWS Anomaly Detector - BACKTEST MODE ({backtest_year})", flush=True)
    else:
        print("PAEWS Anomaly Detector - CURRENT CONDITIONS", flush=True)
    print("=" * 55, flush=True)

    # Step 1: Load climatology
    print("\nStep 1: Loading climatology...", flush=True)
    clim_ds = load_climatology()
    if clim_ds is None:
        sys.exit(1)

    # Step 2: Load SST data
    print("\nStep 2: Loading SST data...", flush=True)
    if backtest_year:
        sst_ds = load_backtest_sst(backtest_year)
        output_suffix = f"_{backtest_year}"
    else:
        sst_ds = load_current_sst()
        output_suffix = ""
    if sst_ds is None:
        sys.exit(1)

    # Step 3: Compute Z-scores
    zscore, stats = compute_zscore(sst_ds, clim_ds)

    # Step 4: Gradient anomaly
    gradient_info = compute_gradient_anomaly(sst_ds, clim_ds)

    # Step 5: Dashboard
    plot_anomaly_dashboard(zscore, sst_ds, clim_ds, stats, gradient_info, output_suffix)

    # Step 6: Save data
    save_zscore(zscore, stats, output_suffix)

    # Step 7: Backtest time series (only in backtest mode)
    if backtest_year:
        plot_backtest_timeseries(zscore, sst_ds, backtest_year)

    # Final summary
    print("\n" + "=" * 55, flush=True)
    print("ANOMALY DETECTION COMPLETE", flush=True)
    print(f"  Status: {stats['alert_level']}", flush=True)
    print(f"  Mean Z-score: {stats['z_mean']:+.2f}", flush=True)
    print(f"  Gradient anomaly: {gradient_info['gradient_anomaly']:+.2f}C", flush=True)
    if backtest_year:
        print(f"  Dashboard: outputs/sst_anomaly_dashboard_{backtest_year}.png", flush=True)
        print(f"  Time series: outputs/sst_backtest_{backtest_year}.png", flush=True)
    else:
        print(f"  Dashboard: outputs/sst_anomaly_dashboard.png", flush=True)
    print("=" * 55, flush=True)
