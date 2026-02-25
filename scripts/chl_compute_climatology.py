"""
PAEWS Chlorophyll Climatology Calculator
=========================================
Computes monthly mean and standard deviation of chlorophyll-a
from the 2003-2022 MODIS Aqua baseline.

This produces the chlorophyll equivalent of your SST climatology —
the "normal" chlorophyll for each month, so we can detect anomalies.

KEY DIFFERENCE FROM SST:
    Chlorophyll is log-normally distributed, not normally distributed.
    Values range from ~0.01 to ~50 mg/m³, with most ocean pixels around
    0.1-0.5 and upwelling zones around 1-10. This huge range means:
    
    1. We compute statistics on log10(chlorophyll), not raw values
    2. Z-scores are computed in log-space
    3. Maps use log color scales
    
    This is standard practice in ocean color remote sensing.
    Every paper you'll read does this.

THE SEASONAL PARADOX:
    Off Peru, chlorophyll is LOWEST in winter (Jun-Sep) when upwelling
    is STRONGEST. This seems backwards but it's because:
    - Strong upwelling = deep mixed layer = phytoplankton diluted over depth
    - Strong upwelling = more clouds = less light for photosynthesis
    
    The monthly climatology captures this pattern automatically.
    A Z-score of -2 in July means "unusually low even for winter" —
    that's a real signal, not the normal seasonal dip.

Usage:
    python chl_compute_climatology.py
"""

import sys
print("PAEWS Chlorophyll Climatology Calculator starting...", flush=True)

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import warnings

print("Imports done", flush=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("c:/Users/josep/Documents/paews")
DATA_CHL_BASELINE = BASE_DIR / "data" / "baseline_v2_chl"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUTS = BASE_DIR / "outputs"

BASELINE_START_YEAR = 2003
BASELINE_END_YEAR = 2022


def load_all_years():
    """
    Load all yearly chlorophyll files and concatenate into one dataset.
    
    MODIS chlorophyll has MUCH more missing data than OISST because:
    - Optical sensor: blocked by clouds (OISST uses microwave, sees through clouds)
    - Sun glint: reflected sunlight blinds the sensor at certain angles
    - High aerosols: dust/haze corrupts the retrieval
    - La Garua: Peru's coastal fog season (Jun-Sep) = worst coverage
    
    Missing pixels are NaN. We work around them by computing statistics
    only from valid observations (skipna=True).
    """
    datasets = []
    
    for year in range(BASELINE_START_YEAR, BASELINE_END_YEAR + 1):
        fpath = DATA_CHL_BASELINE / f"chl_{year}.nc"
        if not fpath.exists():
            print(f"  WARNING: Missing {fpath.name}, skipping", flush=True)
            continue
        
        try:
            ds = xr.open_dataset(fpath)
            n_times = ds.sizes.get('time', 0)
            
            # Check what percentage of data is valid (not NaN)
            chl = ds["chlorophyll"]
            total_pixels = chl.size
            valid_pixels = int(chl.count().values)
            coverage = (valid_pixels / total_pixels * 100) if total_pixels > 0 else 0
            
            print(f"  Loaded {year}: {n_times} composites, {coverage:.0f}% valid pixels", flush=True)
            datasets.append(ds)
        except Exception as e:
            print(f"  ERROR loading {year}: {e}", flush=True)
    
    if not datasets:
        print("  FATAL: No chlorophyll data files found!", flush=True)
        return None
    
    print(f"\n  Concatenating {len(datasets)} years...", flush=True)
    combined = xr.concat(datasets, dim="time")
    
    # Sort by time (should already be sorted, but be safe)
    combined = combined.sortby("time")
    
    total_composites = combined.sizes.get('time', 0)
    print(f"  Total: {total_composites} 8-day composites over {len(datasets)} years", flush=True)
    
    return combined


def compute_climatology(combined_ds):
    """
    Compute monthly mean and standard deviation of log10(chlorophyll).
    
    Why log-transform?
        Raw chlorophyll ranges from 0.01 to 50+ mg/m³.
        The distribution is heavily right-skewed (most values are small,
        a few are very large). Taking log10 makes it approximately normal,
        which means Z-scores are meaningful.
        
        In log space:
            log10(0.1) = -1.0  (oligotrophic open ocean)
            log10(1.0) =  0.0  (moderate productivity)
            log10(10)  =  1.0  (highly productive upwelling)
        
        A Z-score of -2 in log space means "chlorophyll is ~100x lower
        than normal" which is a massive biological signal.
    """
    print("\nStep 2: Computing monthly climatology...", flush=True)
    
    # Log-transform chlorophyll
    # Replace zeros and negatives with NaN before log (shouldn't exist, but safety)
    chl = combined_ds["chlorophyll"].where(combined_ds["chlorophyll"] > 0)
    log_chl = np.log10(chl)
    
    print("  Computing log10(chlorophyll) monthly statistics...", flush=True)
    
    # Suppress warnings for all-NaN slices (land/persistent cloud pixels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        monthly_mean = log_chl.groupby("time.month").mean(dim="time", skipna=True)
        monthly_std = log_chl.groupby("time.month").std(dim="time", ddof=1, skipna=True)
        monthly_count = log_chl.groupby("time.month").count(dim="time")
    
    # Report statistics
    for m in range(1, 13):
        mean_val = float(monthly_mean.sel(month=m).mean(skipna=True))
        std_val = float(monthly_std.sel(month=m).mean(skipna=True))
        count_val = int(monthly_count.sel(month=m).mean(skipna=True))
        # Convert log mean back to real units for display
        real_mean = 10**mean_val
        print(f"  Month {m:2d}: log10 mean={mean_val:+.3f} "
              f"(≈{real_mean:.2f} mg/m³), std={std_val:.3f}, "
              f"avg obs/pixel={count_val}", flush=True)
    
    return monthly_mean, monthly_std, monthly_count


def save_climatology(monthly_mean, monthly_std, monthly_count):
    """Save climatology as NetCDF."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    clim_ds = xr.Dataset({
        "chl_log_mean": monthly_mean,     # log10(mg/m³) monthly mean
        "chl_log_std": monthly_std,        # log10(mg/m³) monthly std
        "chl_obs_count": monthly_count,    # number of valid 8-day composites per pixel per month
    })
    
    clim_ds.attrs["description"] = (
        f"PAEWS Chlorophyll Climatology v2 — log10(chlorophyll-a) monthly statistics"
    )
    clim_ds.attrs["baseline_period"] = f"{BASELINE_START_YEAR}-{BASELINE_END_YEAR}"
    clim_ds.attrs["source"] = "MODIS Aqua 8-day composite (erdMH1chla8day, Global, 4km, Science Quality)"
    clim_ds.attrs["units"] = "log10(mg/m³)"
    clim_ds.attrs["bounding_box"] = "0S-16S, 85W-70W"
    clim_ds.attrs["note"] = (
        "Statistics computed in log10 space because chlorophyll is log-normally distributed. "
        "To get Z-scores: Z = (log10(current_chl) - chl_log_mean) / chl_log_std"
    )
    
    outpath = DATA_PROCESSED / "chl_climatology_v2.nc"
    clim_ds.to_netcdf(outpath)
    size_kb = outpath.stat().st_size / 1024
    print(f"\n  Saved: {outpath} ({size_kb:.0f} KB)", flush=True)
    
    return clim_ds


def plot_climatology(clim_ds):
    """Create a 12-panel plot showing monthly mean chlorophyll."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    
    month_names = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(
        f"PAEWS Chlorophyll-a Monthly Climatology v2 ({BASELINE_START_YEAR}-{BASELINE_END_YEAR})\n"
        f"MODIS Aqua 8-day composite — Box (0S-16S, 85W-70W)\n"
        f"Values in log10(mg/m³)",
        fontsize=14, fontweight='bold'
    )
    
    for m in range(12):
        ax = axes[m // 4, m % 4]
        data = clim_ds["chl_log_mean"].sel(month=m + 1)
        
        # Plot in log space — color range from -1.5 to 1.5 
        # (corresponds to ~0.03 to ~30 mg/m³)
        im = data.plot(ax=ax, vmin=-1.5, vmax=1.5, cmap="YlGn",
                       add_colorbar=False)
        ax.set_title(month_names[m], fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    # Add single colorbar
    fig.colorbar(im, ax=axes, label="log10(Chlorophyll-a) [mg/m³]",
                 shrink=0.6, pad=0.02)
    
    plt.tight_layout()
    outpath = OUTPUTS / "chl_climatology_preview.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Climatology preview saved: {outpath}", flush=True)


def plot_coverage(clim_ds):
    """Show data coverage by month — important for understanding where cloud gaps are."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(
        f"PAEWS Chlorophyll Data Coverage by Month ({BASELINE_START_YEAR}-{BASELINE_END_YEAR})\n"
        f"Number of valid 8-day composites per pixel (higher = better coverage)",
        fontsize=14, fontweight='bold'
    )
    
    for m in range(12):
        ax = axes[m // 4, m % 4]
        count = clim_ds["chl_obs_count"].sel(month=m + 1)
        
        # Max possible: ~20 years × ~4 composites per month = ~80
        im = count.plot(ax=ax, vmin=0, vmax=80, cmap="viridis",
                        add_colorbar=False)
        ax.set_title(month_names[m], fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    fig.colorbar(im, ax=axes, label="Valid observations (8-day composites)",
                 shrink=0.6, pad=0.02)
    
    plt.tight_layout()
    outpath = OUTPUTS / "chl_coverage_by_month.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Coverage map saved: {outpath}", flush=True)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("PAEWS Chlorophyll Climatology Calculator", flush=True)
    print(f"Baseline: {BASELINE_START_YEAR}-{BASELINE_END_YEAR}", flush=True)
    print(f"Source: {DATA_CHL_BASELINE}", flush=True)
    print("=" * 60, flush=True)
    
    # Step 1: Load all years
    print("\nStep 1: Loading chlorophyll data...", flush=True)
    combined = load_all_years()
    if combined is None:
        sys.exit(1)
    
    # Step 2: Compute climatology
    monthly_mean, monthly_std, monthly_count = compute_climatology(combined)
    
    # Step 3: Save
    print("\nStep 3: Saving climatology...", flush=True)
    clim_ds = save_climatology(monthly_mean, monthly_std, monthly_count)
    
    # Step 4: Visualize
    print("\nStep 4: Generating preview plots...", flush=True)
    plot_climatology(clim_ds)
    plot_coverage(clim_ds)
    
    print("\n" + "=" * 60, flush=True)
    print("CHLOROPHYLL CLIMATOLOGY COMPLETE", flush=True)
    print(f"  Climatology: data/processed/chl_climatology_v2.nc", flush=True)
    print(f"  Preview: outputs/chl_climatology_preview.png", flush=True)
    print(f"  Coverage: outputs/chl_coverage_by_month.png", flush=True)
    print("=" * 60, flush=True)
