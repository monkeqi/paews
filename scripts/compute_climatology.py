"""
PAEWS Phase 2 - Step 2: Compute SST Climatology
=================================================
Reads all 23 years of baseline SST data and computes monthly
climatological statistics: mean and standard deviation for
each pixel for each calendar month (January through December).

WHAT IS A CLIMATOLOGY?
A climatology answers: "What does this pixel normally look like
in February?" If we have 23 Februaries of data, we average them
all to get the mean, and compute the spread (standard deviation).

For example, if pixel (-12.5, -77.5) has these February averages
across 23 years: [20.1, 19.8, 21.3, 20.5, 22.1, ...]
  Mean = 20.4 C  (what "normal February" looks like here)
  Std  = 0.8 C   (how much February typically varies)

Then if current February reads 23.0 C at this pixel:
  Z-score = (23.0 - 20.4) / 0.8 = +3.25
That is 3.25 standard deviations above normal -- a strong anomaly.

WHY MONTHLY (NOT WEEKLY)?
Weekly climatologies would be noisier because you only have 23
samples per week. Monthly gives you ~23*30 = 690 daily values
per month per pixel, which produces much more stable statistics.
For our seasonal alert calendar, monthly resolution is sufficient.
We can refine to biweekly later if needed.

OUTPUT:
  data/processed/sst_climatology.nc
  Contains:
    - sst_mean:     (month, latitude, longitude) - 12 monthly mean maps
    - sst_std:      (month, latitude, longitude) - 12 monthly std maps
    - sst_count:    (month, latitude, longitude) - number of valid days
    - month:        [1, 2, 3, ..., 12] coordinate

  outputs/sst_climatology_preview.png - visual check of the results

Usage:
  & C:/Users/josep/miniconda3/Scripts/conda.exe run -n geosentinel python c:/Users/josep/Documents/paews/scripts/compute_climatology.py
"""

import sys
print("PAEWS Climatology Calculator starting...", flush=True)

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

print("Imports done", flush=True)

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path("c:/Users/josep/Documents/paews")
DATA_BASELINE = BASE_DIR / "data" / "baseline"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUTS = BASE_DIR / "outputs"

BASELINE_START_YEAR = 2003
BASELINE_END_YEAR = 2025


# ============================================================
# STEP 1: LOAD ALL BASELINE FILES INTO ONE DATASET
# ============================================================
# We open all 23 yearly NetCDF files and concatenate them along
# the time dimension. xarray makes this easy with open_mfdataset
# ("multi-file dataset"), but we will do it manually so you can
# see exactly what is happening at each step.

def load_all_baseline_data():
    """Load all yearly SST files and combine into one big dataset."""
    print("\nStep 1: Loading baseline files...", flush=True)

    datasets = []
    total_days = 0

    for year in range(BASELINE_START_YEAR, BASELINE_END_YEAR + 1):
        filepath = DATA_BASELINE / f"sst_{year}.nc"
        if not filepath.exists():
            print(f"  WARNING: {filepath.name} not found - skipping", flush=True)
            continue

        ds = xr.open_dataset(filepath)

        # Get the SST variable -- drop the altitude/zlev dimension
        # because it is always 0 (surface) and just gets in the way
        if "zlev" in ds.dims:
            ds = ds.squeeze("zlev", drop=True)
        if "altitude" in ds.dims:
            ds = ds.squeeze("altitude", drop=True)

        n_days = ds.sizes["time"]
        total_days += n_days
        datasets.append(ds)
        print(f"  Loaded {filepath.name}: {n_days} days", flush=True)

    if not datasets:
        print("ERROR: No baseline files found!", flush=True)
        return None

    # Concatenate all years along the time axis
    # This creates one continuous time series from 2003-01-01 to 2025-12-31
    print(f"\n  Combining {len(datasets)} files ({total_days} total days)...", flush=True)
    combined = xr.concat(datasets, dim="time")

    # Sort by time just in case files were loaded out of order
    combined = combined.sortby("time")

    print(f"  Combined dataset shape: {dict(combined.sizes)}", flush=True)
    print(f"  Time range: {str(combined.time.values[0])[:10]} to {str(combined.time.values[-1])[:10]}", flush=True)

    return combined


# ============================================================
# STEP 2: COMPUTE MONTHLY CLIMATOLOGY
# ============================================================
# For each of the 12 calendar months, we:
#   1. Select all days that fall in that month (across all years)
#   2. Compute the mean SST at each pixel
#   3. Compute the standard deviation at each pixel
#   4. Count how many valid days went into each calculation
#
# The .groupby("time.month") operation in xarray does this
# automatically -- it groups all January days together, all
# February days together, etc., regardless of which year they
# came from. This is the core of climatological analysis.
#
# IMPORTANT: We use ddof=1 in std() for the standard deviation.
# This gives us the "sample standard deviation" (dividing by N-1
# instead of N). This is the correct choice when our 23 years
# are a SAMPLE of all possible years, not the complete population.
# It is a small correction but it is statistically proper.

def compute_monthly_climatology(combined_ds):
    """Compute mean and std of SST for each calendar month."""
    print("\nStep 2: Computing monthly climatology...", flush=True)

    sst = combined_ds["sst"]

    # Group by calendar month and compute statistics
    # This is the key operation -- it takes ~8400 daily maps and
    # reduces them to 12 monthly statistics maps
    print("  Computing monthly means...", flush=True)
    monthly_mean = sst.groupby("time.month").mean(dim="time")

    print("  Computing monthly standard deviations...", flush=True)
    monthly_std = sst.groupby("time.month").std(dim="time", ddof=1)

    print("  Counting valid observations per month...", flush=True)
    monthly_count = sst.groupby("time.month").count(dim="time")

    # Report results for each month
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    print("\n  Monthly climatology summary (spatial averages):", flush=True)
    print(f"  {'Month':<6} {'Mean SST':>10} {'Std Dev':>10} {'Avg Days':>10}", flush=True)
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}", flush=True)

    for m in range(1, 13):
        mean_val = float(monthly_mean.sel(month=m).mean(skipna=True).values)
        std_val = float(monthly_std.sel(month=m).mean(skipna=True).values)
        count_val = float(monthly_count.sel(month=m).mean(skipna=True).values)
        print(f"  {month_names[m-1]:<6} {mean_val:>10.2f}C {std_val:>10.2f}C {count_val:>10.0f}", flush=True)

    return monthly_mean, monthly_std, monthly_count


# ============================================================
# STEP 3: SAVE CLIMATOLOGY TO NETCDF
# ============================================================
# We save the climatology as a single NetCDF file with three
# variables. This file is small (~50 KB) because it only has
# 12 time steps (months) instead of 8400 (days).
#
# This is the file that the anomaly detector will load to
# compare current conditions against.

def save_climatology(monthly_mean, monthly_std, monthly_count):
    """Save climatology as a NetCDF file."""
    print("\nStep 3: Saving climatology...", flush=True)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Create a clean dataset with descriptive variable names
    clim_ds = xr.Dataset(
        {
            "sst_mean": monthly_mean,
            "sst_std": monthly_std,
            "sst_count": monthly_count,
        },
        attrs={
            "title": "PAEWS SST Monthly Climatology",
            "description": "Monthly mean and standard deviation of SST from NOAA OISST",
            "baseline_period": f"{BASELINE_START_YEAR}-{BASELINE_END_YEAR}",
            "source": "ncdcOisst21Agg_LonPM180 via ERDDAP",
            "created_by": "PAEWS compute_climatology.py",
        }
    )

    outpath = DATA_PROCESSED / "sst_climatology.nc"
    clim_ds.to_netcdf(outpath)
    size_kb = outpath.stat().st_size / 1024
    print(f"  Saved: {outpath} ({size_kb:.0f} KB)", flush=True)

    return clim_ds


# ============================================================
# STEP 4: VISUALIZE THE CLIMATOLOGY
# ============================================================
# We create a 4x3 grid showing all 12 monthly mean SST maps.
# This is a visual sanity check -- you should see:
#   - Cooler SST in austral winter (Jun-Aug) especially near coast
#   - Warmer SST in austral summer (Dec-Feb) especially in north
#   - The cold upwelling tongue along the coast in all months
#   - A clear seasonal cycle
#
# If something looks wrong (e.g., all months look identical, or
# there are weird artifacts), it means our climatology has a bug.

def plot_climatology(clim_ds):
    """Create a 12-panel plot showing monthly mean SST."""
    print("\nStep 4: Creating climatology preview...", flush=True)

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    month_names = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(
        f"PAEWS SST Monthly Climatology ({BASELINE_START_YEAR}-{BASELINE_END_YEAR})\n"
        f"NOAA OISST - Peru Coast",
        fontsize=16, fontweight="bold"
    )

    for m in range(1, 13):
        row = (m - 1) // 4
        col = (m - 1) % 4
        ax = axes[row, col]

        sst_month = clim_ds["sst_mean"].sel(month=m)

        im = sst_month.plot(
            ax=ax,
            vmin=14,
            vmax=28,
            cmap="RdYlBu_r",
            add_colorbar=False
        )
        ax.set_title(month_names[m - 1], fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Add one shared colorbar
    fig.subplots_adjust(right=0.92, hspace=0.3, wspace=0.2)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Mean SST (C)")

    outpath = OUTPUTS / "sst_climatology_preview.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Preview saved: {outpath}", flush=True)


# ============================================================
# STEP 5: COMPUTE AND PLOT THE COASTAL-OFFSHORE SST GRADIENT
# ============================================================
# This is the spatial analysis that leverages your GIS skills.
#
# Upwelling strength shows up as a DIFFERENCE between cold
# nearshore water and warm offshore water. We compute this
# gradient for each month.
#
# Since OISST is on a regular 0.25 degree grid, we approximate:
#   Nearshore zone: pixels closest to the coast (lon > -78)
#   Offshore zone:  pixels far from coast (lon < -80)
#
# A strong gradient (e.g., offshore 4-6C warmer than nearshore)
# means upwelling is active. When the gradient weakens toward
# zero, upwelling is failing -- this is an anomaly signal.
#
# In Phase 2 anomaly detection, we will compute this gradient
# for current data and compare it against the climatological
# gradient. For now we just compute and visualize the baseline.

def compute_gradient_climatology(clim_ds):
    """Compute the coastal-offshore SST gradient for each month."""
    print("\nStep 5: Computing coastal-offshore SST gradient...", flush=True)

    sst_mean = clim_ds["sst_mean"]

    # Define nearshore and offshore zones
    # Nearshore: within ~50km of coast (roughly lon > -78 for central Peru)
    # Offshore: 100-200km from coast (roughly lon < -80)
    nearshore = sst_mean.sel(longitude=slice(-78, -70)).mean(dim=["latitude", "longitude"])
    offshore = sst_mean.sel(longitude=slice(-82, -80)).mean(dim=["latitude", "longitude"])

    # Gradient = offshore minus nearshore
    # Positive means offshore is warmer (normal upwelling)
    # Near zero means upwelling has collapsed
    gradient = offshore - nearshore

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    print(f"\n  Monthly SST Gradient (offshore - nearshore):", flush=True)
    print(f"  {'Month':<6} {'Nearshore':>10} {'Offshore':>10} {'Gradient':>10}", flush=True)
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}", flush=True)

    for m in range(1, 13):
        ns_val = float(nearshore.sel(month=m).values)
        os_val = float(offshore.sel(month=m).values)
        gr_val = float(gradient.sel(month=m).values)
        print(f"  {month_names[m-1]:<6} {ns_val:>10.2f}C {os_val:>10.2f}C {gr_val:>+10.2f}C", flush=True)

    # Plot the seasonal gradient cycle
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    months = list(range(1, 13))

    # Top panel: nearshore vs offshore temperatures
    ax1.plot(months, [float(nearshore.sel(month=m).values) for m in months],
             'b-o', linewidth=2, label='Nearshore (< 50km from coast)')
    ax1.plot(months, [float(offshore.sel(month=m).values) for m in months],
             'r-o', linewidth=2, label='Offshore (100-200km)')
    ax1.set_ylabel("SST (C)")
    ax1.set_title("Seasonal SST Cycle: Nearshore vs Offshore")
    ax1.legend()
    ax1.set_xticks(months)
    ax1.set_xticklabels(month_names)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: the gradient
    gradient_vals = [float(gradient.sel(month=m).values) for m in months]
    colors = ['green' if g > 0 else 'red' for g in gradient_vals]
    ax2.bar(months, gradient_vals, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel("SST Gradient (C)")
    ax2.set_title("Coastal-Offshore SST Gradient (positive = active upwelling)")
    ax2.set_xticks(months)
    ax2.set_xticklabels(month_names)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f"PAEWS Upwelling Gradient Analysis ({BASELINE_START_YEAR}-{BASELINE_END_YEAR})",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    outpath = OUTPUTS / "sst_gradient_seasonal.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gradient plot saved: {outpath}", flush=True)

    return gradient


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("=" * 55, flush=True)
    print("PAEWS Climatology Calculator", flush=True)
    print(f"Baseline: {BASELINE_START_YEAR} to {BASELINE_END_YEAR}", flush=True)
    print("=" * 55, flush=True)

    # Step 1: Load all baseline data
    combined = load_all_baseline_data()
    if combined is None:
        sys.exit(1)

    # Step 2: Compute monthly statistics
    monthly_mean, monthly_std, monthly_count = compute_monthly_climatology(combined)

    # Step 3: Save to NetCDF
    clim_ds = save_climatology(monthly_mean, monthly_std, monthly_count)

    # Step 4: Visualize
    plot_climatology(clim_ds)

    # Step 5: Compute upwelling gradient
    gradient = compute_gradient_climatology(clim_ds)

    print("\n" + "=" * 55, flush=True)
    print("CLIMATOLOGY COMPLETE", flush=True)
    print(f"  Climatology file: data/processed/sst_climatology.nc", flush=True)
    print(f"  Preview map: outputs/sst_climatology_preview.png", flush=True)
    print(f"  Gradient plot: outputs/sst_gradient_seasonal.png", flush=True)
    print(f"\nNext step: Run anomaly_detector.py", flush=True)
    print("=" * 55, flush=True)
