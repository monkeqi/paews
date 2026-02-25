"""
PAEWS Chlorophyll Anomaly Detector + Compound Event Detection
==============================================================
Detects chlorophyll anomalies and combines with SST anomalies to identify
compound marine heatwave + low-chlorophyll events (Le Grix et al., 2021).

This is the second pillar of PAEWS. SST tells you the ocean is warm.
Chlorophyll tells you the food chain is collapsing. Both together = 
compound event = high confidence that anchovy season is at risk.

METHOD (following Le Grix et al., 2021):
    - MHW (marine heatwave): SST anomaly > 90th percentile (Z > +1.28)
    - LChl (low chlorophyll): Chl anomaly < 10th percentile (Z < -1.28)
    - Compound event: MHW AND LChl simultaneously in the same pixel
    
    Z-scores are computed in log10 space for chlorophyll because
    chlorophyll is log-normally distributed.

NEARSHORE/OFFSHORE SPLIT (Espinoza-Morriberon, 2025):
    During El Nino, chlorophyll DECREASES offshore but INCREASES within
    25km of shore. If we average the whole box, these cancel out and we
    miss the signal. So we split:
    - Nearshore: lon >= -78 (roughly within 200km of coast)
    - Offshore: lon < -78

RATE OF CHANGE:
    A sudden week-over-week chlorophyll drop is more predictive than
    absolute low values. We compute this as the difference between
    consecutive 8-day composites in Z-score space.

Usage:
    # Current conditions
    python chl_anomaly_detector.py

    # Backtest a year (default: last month in data)
    python chl_anomaly_detector.py --backtest 2023

    # Backtest specific month
    python chl_anomaly_detector.py --backtest 2023 --month 3
"""

import sys
print("PAEWS Chlorophyll Anomaly Detector starting...", flush=True)

import argparse
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from datetime import datetime
import calendar
import warnings

print("Imports done", flush=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("c:/Users/josep/Documents/paews")
DATA_CHL_BASELINE = BASE_DIR / "data" / "baseline_v2_chl"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_CURRENT = BASE_DIR / "data" / "current"
OUTPUTS = BASE_DIR / "outputs"

# Spatial zones
NEARSHORE_LON = slice(-78, -70)   # within ~200km of Peru coast
OFFSHORE_LON = slice(-85, -78)     # further offshore

# Compound event thresholds (Le Grix et al., 2021)
# 90th percentile SST = Z > +1.28
# 10th percentile Chl = Z < -1.28
MHW_THRESHOLD = 1.28       # SST Z-score above this = marine heatwave
LCHL_THRESHOLD = -1.28     # Chl Z-score below this = low chlorophyll

# Coverage threshold — mask pixels with too few observations
MIN_OBS_COUNT = 10  # minimum valid composites per pixel per month in baseline


# =============================================================================
# DATA LOADING
# =============================================================================

def load_chl_climatology():
    """Load the chlorophyll climatology (log10 space)."""
    path = DATA_PROCESSED / "chl_climatology_v2.nc"
    if not path.exists():
        print("  ERROR: No chlorophyll climatology. Run chl_compute_climatology.py first.", flush=True)
        return None
    ds = xr.open_dataset(path)
    print(f"  Loaded chlorophyll climatology: {path.name}", flush=True)
    return ds


def load_sst_climatology():
    """Load the SST climatology."""
    path = DATA_PROCESSED / "sst_climatology_v2.nc"
    if not path.exists():
        path = DATA_PROCESSED / "sst_climatology.nc"
        if not path.exists():
            print("  WARNING: No SST climatology found. Compound detection disabled.", flush=True)
            return None
    ds = xr.open_dataset(path)
    print(f"  Loaded SST climatology: {path.name}", flush=True)
    return ds


def load_chl_data(year=None):
    """Load chlorophyll data — either from baseline (backtest) or current."""
    if year:
        path = DATA_CHL_BASELINE / f"chl_{year}.nc"
        if not path.exists():
            print(f"  Chlorophyll for {year} not found, attempting download...", flush=True)
            # Try downloading
            from chl_baseline_builder import download_year
            result = download_year(year)
            if not result:
                return None
        ds = xr.open_dataset(path)
        n = ds.sizes.get('time', 0)
        print(f"  Loaded chlorophyll {year}: {n} composites", flush=True)
        return ds
    else:
        path = DATA_CURRENT / "chlorophyll_current.nc"
        if not path.exists():
            print("  ERROR: No current chlorophyll data. Run data_pipeline.py first.", flush=True)
            return None
        ds = xr.open_dataset(path)
        print(f"  Loaded current chlorophyll data", flush=True)
        return ds


def load_sst_data(year=None):
    """Load SST data for compound detection."""
    if year:
        # Check baseline_v2 first
        path = BASE_DIR / "data" / "baseline_v2" / f"sst_{year}.nc"
        if not path.exists():
            print(f"  SST for {year} not in baseline_v2. Compound detection limited.", flush=True)
            return None
        ds = xr.open_dataset(path)
        print(f"  Loaded SST {year} for compound detection", flush=True)
        return ds
    else:
        path = DATA_CURRENT / "sst_current.nc"
        if not path.exists():
            return None
        ds = xr.open_dataset(path)
        print(f"  Loaded current SST for compound detection", flush=True)
        return ds


# =============================================================================
# ANOMALY DETECTION
# =============================================================================

def compute_chl_zscore(chl_ds, clim_ds, target_month):
    """
    Compute chlorophyll Z-scores in log10 space.
    
    Z = (log10(observed_chl) - log10_mean) / log10_std
    
    Returns z_scores DataArray and the log10 chlorophyll snapshot.
    """
    # Get climatology for target month
    clim_mean = clim_ds["chl_log_mean"].sel(month=target_month)
    clim_std = clim_ds["chl_log_std"].sel(month=target_month)
    obs_count = clim_ds["chl_obs_count"].sel(month=target_month)
    
    # Filter to target month
    time_index = pd.DatetimeIndex(chl_ds["time"].values)
    month_data = chl_ds.sel(time=time_index.month == target_month)
    
    if len(month_data.time) == 0:
        print(f"  WARNING: No chlorophyll data for month {target_month}", flush=True)
        return None, None
    
    # Get the last composite in the target month
    chl_snapshot = month_data["chlorophyll"].isel(time=-1).squeeze()
    
    # Drop altitude dimension if it exists
    if 'altitude' in chl_snapshot.dims:
        chl_snapshot = chl_snapshot.isel(altitude=0)
    if 'altitude' in chl_snapshot.coords:
        chl_snapshot = chl_snapshot.drop_vars('altitude')
    
    # Log-transform (mask zeros/negatives)
    chl_log = np.log10(chl_snapshot.where(chl_snapshot > 0))
    
    # Align grids if needed
    if chl_log.shape != clim_mean.shape:
        print(f"  Grid mismatch: chl {chl_log.shape} vs clim {clim_mean.shape}, interpolating...", flush=True)
        lat_name = [d for d in chl_log.dims if 'lat' in d.lower()][0]
        lon_name = [d for d in chl_log.dims if 'lon' in d.lower()][0]
        clim_mean = clim_mean.interp(
            **{lat_name: chl_log[lat_name], lon_name: chl_log[lon_name]},
            method="nearest"
        )
        clim_std = clim_std.interp(
            **{lat_name: chl_log[lat_name], lon_name: chl_log[lon_name]},
            method="nearest"
        )
        obs_count = obs_count.interp(
            **{lat_name: chl_log[lat_name], lon_name: chl_log[lon_name]},
            method="nearest"
        )
    
    # Compute Z-scores
    std_safe = clim_std.where(clim_std > 0.01)
    z_scores = (chl_log - clim_mean) / std_safe
    
    # Mask low-coverage pixels
    z_scores = z_scores.where(obs_count >= MIN_OBS_COUNT)
    
    return z_scores, chl_log


def compute_sst_zscore_for_compound(sst_ds, sst_clim, target_month):
    """Compute SST Z-scores for the target month (for compound detection)."""
    if sst_ds is None or sst_clim is None:
        return None
    
    clim_mean = sst_clim["sst_mean"].sel(month=target_month)
    clim_std = sst_clim["sst_std"].sel(month=target_month)
    
    time_index = pd.DatetimeIndex(sst_ds["time"].values)
    month_data = sst_ds.sel(time=time_index.month == target_month)
    
    if len(month_data.time) == 0:
        return None
    
    sst_snapshot = month_data["sst"].isel(time=-1).squeeze()
    if 'zlev' in sst_snapshot.dims:
        sst_snapshot = sst_snapshot.isel(zlev=0)
    
    std_safe = clim_std.where(clim_std > 0.01)
    sst_z = (sst_snapshot - clim_mean) / std_safe
    
    return sst_z


def compute_rate_of_change(chl_ds, clim_ds, target_month):
    """
    Compute week-over-week rate of change in chlorophyll Z-scores.
    
    A sudden drop (large negative ROC) is an early warning signal.
    Uses the last two composites in the target month.
    """
    time_index = pd.DatetimeIndex(chl_ds["time"].values)
    month_data = chl_ds.sel(time=time_index.month == target_month)
    
    if len(month_data.time) < 2:
        return None, None
    
    clim_mean = clim_ds["chl_log_mean"].sel(month=target_month)
    clim_std = clim_ds["chl_log_std"].sel(month=target_month)
    std_safe = clim_std.where(clim_std > 0.01)
    
    # Z-score for last two composites
    chl_now = month_data["chlorophyll"].isel(time=-1).squeeze()
    chl_prev = month_data["chlorophyll"].isel(time=-2).squeeze()
    
    # Drop altitude if present
    for arr in [chl_now, chl_prev]:
        if 'altitude' in arr.dims:
            arr = arr.isel(altitude=0)
    
    log_now = np.log10(chl_now.where(chl_now > 0))
    log_prev = np.log10(chl_prev.where(chl_prev > 0))
    
    z_now = (log_now - clim_mean) / std_safe
    z_prev = (log_prev - clim_mean) / std_safe
    
    roc = z_now - z_prev  # negative = chlorophyll dropping
    roc_mean = float(roc.mean(skipna=True))
    
    return roc, roc_mean


def detect_compound_events(chl_z, sst_z):
    """
    Detect compound MHW + LChl events (Le Grix et al., 2021).
    
    Compound event = pixel where SST Z > +1.28 AND Chl Z < -1.28
    
    Returns compound flag array and statistics.
    """
    if sst_z is None:
        return None, {}
    
    # Interpolate SST Z-scores to chlorophyll grid (SST is 0.25°, Chl is ~4km)
    try:
        lat_name_chl = [d for d in chl_z.dims if 'lat' in d.lower()][0]
        lon_name_chl = [d for d in chl_z.dims if 'lon' in d.lower()][0]
        
        sst_z_interp = sst_z.interp(
            **{
                [d for d in sst_z.dims if 'lat' in d.lower()][0]: chl_z[lat_name_chl],
                [d for d in sst_z.dims if 'lon' in d.lower()][0]: chl_z[lon_name_chl],
            },
            method="nearest"
        )
    except Exception as e:
        print(f"  Could not align SST/Chl grids: {e}", flush=True)
        return None, {}
    
    # Detect events
    mhw_flag = sst_z_interp > MHW_THRESHOLD
    lchl_flag = chl_z < LCHL_THRESHOLD
    compound_flag = mhw_flag & lchl_flag
    
    # Statistics
    valid_pixels = int((~np.isnan(chl_z) & ~np.isnan(sst_z_interp)).sum())
    
    stats = {
        "mhw_pixels": int(mhw_flag.sum(skipna=True)),
        "lchl_pixels": int(lchl_flag.sum(skipna=True)),
        "compound_pixels": int(compound_flag.sum(skipna=True)),
        "valid_pixels": valid_pixels,
        "mhw_pct": float(mhw_flag.sum() / valid_pixels * 100) if valid_pixels > 0 else 0,
        "lchl_pct": float(lchl_flag.sum() / valid_pixels * 100) if valid_pixels > 0 else 0,
        "compound_pct": float(compound_flag.sum() / valid_pixels * 100) if valid_pixels > 0 else 0,
    }
    
    return compound_flag, stats


def compute_zone_stats(z_scores):
    """Compute separate statistics for nearshore and offshore zones."""
    lon_name = [d for d in z_scores.dims if 'lon' in d.lower()][0]
    
    try:
        nearshore = z_scores.sel(**{lon_name: NEARSHORE_LON})
        offshore = z_scores.sel(**{lon_name: OFFSHORE_LON})
    except Exception:
        return {}, {}
    
    near_stats = {
        "mean_z": float(nearshore.mean(skipna=True)),
        "min_z": float(nearshore.min(skipna=True)),
        "max_z": float(nearshore.max(skipna=True)),
        "pct_below_thresh": float((nearshore < LCHL_THRESHOLD).sum(skipna=True) / 
                                   nearshore.count() * 100) if nearshore.count() > 0 else 0,
    }
    
    off_stats = {
        "mean_z": float(offshore.mean(skipna=True)),
        "min_z": float(offshore.min(skipna=True)),
        "max_z": float(offshore.max(skipna=True)),
        "pct_below_thresh": float((offshore < LCHL_THRESHOLD).sum(skipna=True) / 
                                   offshore.count() * 100) if offshore.count() > 0 else 0,
    }
    
    return near_stats, off_stats


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_chl_dashboard(chl_z, compound_flag, compound_stats,
                       near_stats, off_stats, roc_mean,
                       target_date, month, year_label):
    """Generate chlorophyll + compound event dashboard."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Determine compound status
    cpct = compound_stats.get("compound_pct", 0)
    if cpct > 15:
        status = "COMPOUND ALERT"
        status_color = "red"
    elif cpct > 5:
        status = "COMPOUND WARNING"
        status_color = "orange"
    elif compound_stats.get("lchl_pct", 0) > 20:
        status = "LOW CHL WARNING"
        status_color = "gold"
    else:
        status = "NORMAL"
        status_color = "green"
    
    fig.suptitle(
        f"PAEWS Chlorophyll + Compound Event Dashboard — {target_date}\n"
        f"Status: {status} | Compound pixels: {cpct:.1f}%",
        fontsize=14, fontweight='bold', color=status_color
    )
    
    # Panel 1: Chlorophyll Z-score map
    ax1 = axes[0, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chl_z.plot(ax=ax1, vmin=-3, vmax=3, cmap="RdYlGn",
                   cbar_kwargs={"label": "Chl Z-score (log space)", "shrink": 0.8})
    ax1.set_title("Chlorophyll Z-Score (negative = low productivity)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    
    # Panel 2: Compound event map
    ax2 = axes[0, 1]
    if compound_flag is not None:
        compound_float = compound_flag.astype(float).where(~np.isnan(chl_z))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compound_float.plot(ax=ax2, vmin=0, vmax=1, cmap="RdYlGn_r",
                               cbar_kwargs={"label": "Compound event (1=yes)", "shrink": 0.8})
        ax2.set_title("Compound Events (warm SST + low Chl)")
    else:
        ax2.text(0.5, 0.5, "SST data not available\nfor compound detection",
                 transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        ax2.set_title("Compound Events (unavailable)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    
    # Panel 3: Z-score histogram with nearshore/offshore split
    ax3 = axes[1, 0]
    z_flat = chl_z.values.flatten()
    z_flat = z_flat[~np.isnan(z_flat)]
    ax3.hist(z_flat, bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
    ax3.axvline(LCHL_THRESHOLD, color='red', linestyle='--', linewidth=2, 
                label=f'Low Chl threshold ({LCHL_THRESHOLD}σ)')
    ax3.axvline(0, color='gray', linestyle='-', linewidth=0.5)
    chl_mean_z = float(chl_z.mean(skipna=True))
    ax3.axvline(chl_mean_z, color='black', linestyle='-', linewidth=2, 
                label=f'Mean ({chl_mean_z:.2f}σ)')
    ax3.set_xlabel("Chlorophyll Z-score (log space)")
    ax3.set_ylabel("Pixel count")
    ax3.set_title("Z-Score Distribution")
    ax3.legend(fontsize=8)
    
    # Panel 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    roc_str = f"{roc_mean:+.3f}" if roc_mean is not None else "N/A"
    
    summary = (
        f"PAEWS CHLOROPHYLL ANOMALY DETECTION\n"
        f"{'='*42}\n\n"
        f"Date: {target_date}\n"
        f"Month: {calendar.month_name[month]}\n"
        f"Status: {status}\n\n"
        f"CHLOROPHYLL Z-SCORES (log10 space):\n"
        f"  Overall mean:  {chl_mean_z:+.2f}σ\n"
        f"  Rate of change: {roc_str}/composite\n\n"
        f"NEARSHORE (lon >= -78°):\n"
        f"  Mean Z: {near_stats.get('mean_z', 0):+.2f}σ\n"
        f"  Below threshold: {near_stats.get('pct_below_thresh', 0):.1f}%\n\n"
        f"OFFSHORE (lon < -78°):\n"
        f"  Mean Z: {off_stats.get('mean_z', 0):+.2f}σ\n"
        f"  Below threshold: {off_stats.get('pct_below_thresh', 0):.1f}%\n\n"
        f"COMPOUND EVENTS (MHW + LChl):\n"
        f"  MHW pixels: {compound_stats.get('mhw_pct', 0):.1f}%\n"
        f"  LChl pixels: {compound_stats.get('lchl_pct', 0):.1f}%\n"
        f"  Compound: {cpct:.1f}%\n"
    )
    
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    outpath = OUTPUTS / f"chl_compound_dashboard_{year_label}.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dashboard saved: {outpath}", flush=True)


def plot_backtest_timeseries(chl_ds, clim_ds, sst_ds, sst_clim, year):
    """
    Generate full-year time series showing chlorophyll Z-scores,
    compound event frequency, and rate of change.
    """
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    
    times = pd.DatetimeIndex(chl_ds["time"].values)
    daily_chl_z = []
    daily_compound_pct = []
    daily_near_z = []
    daily_off_z = []
    
    for i, t in enumerate(times):
        month = t.month
        try:
            clim_mean = clim_ds["chl_log_mean"].sel(month=month)
            clim_std = clim_ds["chl_log_std"].sel(month=month)
            obs_count = clim_ds["chl_obs_count"].sel(month=month)
            std_safe = clim_std.where(clim_std > 0.01)
            
            chl_slice = chl_ds["chlorophyll"].isel(time=i).squeeze()
            if 'altitude' in chl_slice.dims:
                chl_slice = chl_slice.isel(altitude=0)
            
            chl_log = np.log10(chl_slice.where(chl_slice > 0))
            z = (chl_log - clim_mean) / std_safe
            z = z.where(obs_count >= MIN_OBS_COUNT)
            
            z_mean = float(z.mean(skipna=True))
            daily_chl_z.append(z_mean)
            
            # Zone stats
            lon_name = [d for d in z.dims if 'lon' in d.lower()][0]
            try:
                near_z = float(z.sel(**{lon_name: NEARSHORE_LON}).mean(skipna=True))
                off_z = float(z.sel(**{lon_name: OFFSHORE_LON}).mean(skipna=True))
            except Exception:
                near_z = z_mean
                off_z = z_mean
            daily_near_z.append(near_z)
            daily_off_z.append(off_z)
            
            # Compound detection (if SST available)
            if sst_ds is not None and sst_clim is not None:
                sst_z = compute_sst_zscore_for_compound(sst_ds, sst_clim, month)
                if sst_z is not None:
                    _, stats = detect_compound_events(z, sst_z)
                    daily_compound_pct.append(stats.get("compound_pct", 0))
                else:
                    daily_compound_pct.append(0)
            else:
                daily_compound_pct.append(0)
                
        except Exception as e:
            daily_chl_z.append(np.nan)
            daily_compound_pct.append(0)
            daily_near_z.append(np.nan)
            daily_off_z.append(np.nan)
    
    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f"PAEWS Chlorophyll Backtest: {year}\n"
        f"Baseline: 2004-2022 | Box: 0°S-16°S, 85°W-70°W",
        fontsize=13, fontweight='bold'
    )
    
    # Panel 1: Chlorophyll Z-scores (nearshore vs offshore)
    ax1.plot(times, daily_chl_z, color='black', linewidth=1, label='Overall mean')
    ax1.plot(times, daily_near_z, color='blue', linewidth=0.8, alpha=0.7, label='Nearshore')
    ax1.plot(times, daily_off_z, color='red', linewidth=0.8, alpha=0.7, label='Offshore')
    ax1.axhline(LCHL_THRESHOLD, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax1.fill_between(times, LCHL_THRESHOLD, -4, alpha=0.1, color='red', label='Low Chl zone')
    ax1.set_ylabel("Chl Z-score (log space)")
    ax1.set_ylim(-4, 3)
    ax1.legend(loc='lower left', fontsize=8)
    ax1.set_title("Chlorophyll Anomaly (nearshore vs offshore)")
    
    # Panel 2: Compound event percentage
    ax2.fill_between(times, 0, daily_compound_pct, color='darkred', alpha=0.6)
    ax2.axhline(15, color='red', linestyle='--', linewidth=1, label='Alert (15%)')
    ax2.axhline(5, color='orange', linestyle='--', linewidth=1, label='Warning (5%)')
    ax2.set_ylabel("Compound event (%)")
    ax2.set_ylim(0, max(50, max(daily_compound_pct) * 1.1) if daily_compound_pct else 50)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_title("Compound MHW + Low Chlorophyll Events")
    
    # Panel 3: Rate of change (computed as diff between consecutive Z-scores)
    roc = np.diff(daily_chl_z, prepend=np.nan)
    ax3.bar(times, roc, width=8, color=['red' if r < -0.3 else 'gray' for r in roc], alpha=0.7)
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.axhline(-0.3, color='red', linestyle='--', linewidth=1, label='Rapid drop threshold')
    ax3.set_ylabel("Z-score change / composite")
    ax3.set_xlabel("Date")
    ax3.legend(loc='lower left', fontsize=8)
    ax3.set_title("Chlorophyll Rate of Change (red = rapid drop)")
    
    plt.tight_layout()
    outpath = OUTPUTS / f"chl_backtest_{year}.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Backtest time series saved: {outpath}", flush=True)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAEWS Chlorophyll Anomaly Detector")
    parser.add_argument("--backtest", type=int, default=None, help="Year to backtest")
    parser.add_argument("--month", type=int, default=None, help="Target month 1-12")
    args = parser.parse_args()
    
    is_backtest = args.backtest is not None
    year = args.backtest
    
    header = f"BACKTEST MODE ({year})" if is_backtest else "CURRENT CONDITIONS"
    if args.month:
        header += f" month={args.month}"
    
    print("=" * 60, flush=True)
    print(f"PAEWS Chlorophyll Anomaly Detector - {header}", flush=True)
    print("=" * 60, flush=True)
    
    # ---- Load climatologies ----
    print("\nStep 1: Loading climatologies...", flush=True)
    chl_clim = load_chl_climatology()
    sst_clim = load_sst_climatology()
    
    if chl_clim is None:
        sys.exit(1)
    
    # ---- Load data ----
    print("\nStep 2: Loading data...", flush=True)
    chl_ds = load_chl_data(year)
    sst_ds = load_sst_data(year)
    
    if chl_ds is None:
        print("  FATAL: No chlorophyll data available.", flush=True)
        sys.exit(1)
    
    # ---- Determine target month ----
    time_index = pd.DatetimeIndex(chl_ds["time"].values)
    if args.month:
        target_month = args.month
        month_times = time_index[time_index.month == target_month]
        if len(month_times) == 0:
            print(f"  ERROR: No data for month {target_month}", flush=True)
            sys.exit(1)
        latest_date = month_times[-1]
    else:
        latest_date = time_index[-1]
        target_month = latest_date.month
    
    target_date_str = latest_date.strftime('%Y-%m-%d')
    
    # ---- Compute chlorophyll Z-scores ----
    print(f"\nStep 3: Computing chlorophyll Z-scores...", flush=True)
    chl_z, chl_log = compute_chl_zscore(chl_ds, chl_clim, target_month)
    
    if chl_z is None:
        print("  FATAL: Could not compute Z-scores.", flush=True)
        sys.exit(1)
    
    chl_z_mean = float(chl_z.mean(skipna=True))
    chl_z_min = float(chl_z.min(skipna=True))
    chl_z_max = float(chl_z.max(skipna=True))
    
    print(f"  Target date: {target_date_str}", flush=True)
    print(f"  Target month: {target_month} ({calendar.month_name[target_month]})", flush=True)
    print(f"  Mean Chl Z-score: {chl_z_mean:+.2f}", flush=True)
    print(f"  Min/Max Z: {chl_z_min:+.2f} / {chl_z_max:+.2f}", flush=True)
    
    # ---- Zone statistics ----
    print(f"\nStep 4: Nearshore/offshore split...", flush=True)
    near_stats, off_stats = compute_zone_stats(chl_z)
    print(f"  Nearshore mean Z: {near_stats.get('mean_z', 0):+.2f}", flush=True)
    print(f"  Offshore mean Z:  {off_stats.get('mean_z', 0):+.2f}", flush=True)
    print(f"  Nearshore below threshold: {near_stats.get('pct_below_thresh', 0):.1f}%", flush=True)
    print(f"  Offshore below threshold:  {off_stats.get('pct_below_thresh', 0):.1f}%", flush=True)
    
    # ---- Rate of change ----
    print(f"\nStep 5: Rate of change...", flush=True)
    roc, roc_mean = compute_rate_of_change(chl_ds, chl_clim, target_month)
    if roc_mean is not None:
        print(f"  ROC (last two composites): {roc_mean:+.3f} Z/composite", flush=True)
        if roc_mean < -0.3:
            print(f"  >>> RAPID CHLOROPHYLL DROP DETECTED <<<", flush=True)
    else:
        print(f"  Insufficient data for ROC", flush=True)
    
    # ---- Compound event detection ----
    print(f"\nStep 6: Compound event detection...", flush=True)
    sst_z = compute_sst_zscore_for_compound(sst_ds, sst_clim, target_month)
    compound_flag, compound_stats = detect_compound_events(chl_z, sst_z)
    
    if compound_stats:
        print(f"  MHW pixels (SST > +{MHW_THRESHOLD}σ): {compound_stats['mhw_pct']:.1f}%", flush=True)
        print(f"  LChl pixels (Chl < {LCHL_THRESHOLD}σ): {compound_stats['lchl_pct']:.1f}%", flush=True)
        print(f"  COMPOUND pixels: {compound_stats['compound_pct']:.1f}%", flush=True)
        
        if compound_stats['compound_pct'] > 15:
            print(f"\n  >>> COMPOUND ALERT: Widespread warm SST + low chlorophyll <<<", flush=True)
        elif compound_stats['compound_pct'] > 5:
            print(f"\n  >>> COMPOUND WARNING <<<", flush=True)
    else:
        print(f"  SST data not available — compound detection skipped", flush=True)
    
    # ---- Dashboard ----
    print(f"\nStep 7: Generating dashboard...", flush=True)
    year_label = f"{year}_m{target_month:02d}" if is_backtest and args.month else (str(year) if is_backtest else "current")
    
    plot_chl_dashboard(
        chl_z, compound_flag, compound_stats,
        near_stats, off_stats, roc_mean,
        target_date_str, target_month, year_label
    )
    
    # ---- Backtest time series ----
    if is_backtest:
        print(f"\nStep 8: Generating backtest time series...", flush=True)
        plot_backtest_timeseries(chl_ds, chl_clim, sst_ds, sst_clim, year)
    
    # ---- Summary ----
    print("\n" + "=" * 60, flush=True)
    print("CHLOROPHYLL ANOMALY DETECTION COMPLETE", flush=True)
    print(f"  Chl mean Z: {chl_z_mean:+.2f}", flush=True)
    print(f"  Nearshore Z: {near_stats.get('mean_z', 0):+.2f}", flush=True)
    print(f"  Offshore Z: {off_stats.get('mean_z', 0):+.2f}", flush=True)
    print(f"  Compound events: {compound_stats.get('compound_pct', 0):.1f}%", flush=True)
    print(f"  Dashboard: outputs/chl_compound_dashboard_{year_label}.png", flush=True)
    if is_backtest:
        print(f"  Time series: outputs/chl_backtest_{year}.png", flush=True)
    print("=" * 60, flush=True)
