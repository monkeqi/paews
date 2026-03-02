"""
predict_2026_s1.py - PAEWS 2026 S1 Prediction (v2)

MODEL v2 CHANGES (Sessions 12-13):
    - 3 features: sst_z, chl_z, nino12_t1
    - Dropped is_summer and bio_thresh_pct (r=0.963 multicollinearity)
    - Risk tiers replace binary threshold (SEVERE tier 100% historical accuracy)
    - LOO ROC-AUC: 0.629 (up from 0.583)

Usage:
    cd C:/Users/josep/Documents/paews
    python scripts/predict_2026_s1.py

Data sources (all on disk):
    - SST 2026: data/current/sst_current.nc (OISST, same sensor as training)
    - Chl PRIMARY: data/external/chl_copernicus_full.nc (Copernicus monthly, same as training)
    - Chl FALLBACK: data/current/chlorophyll_current.nc (VIIRS daily, REQUIRES BIAS CORRECTION)
    - SST climatology: data/processed/sst_climatology_v2.nc
    - Chl climatology: data/processed/chl_climatology_copernicus.nc
    - Nino indices: data/external/nino_indices_monthly.csv
    - Training data: data/external/paews_feature_matrix.csv

SENSOR NOTE:
    Training Chl uses Copernicus L4 monthly. VIIRS daily reads ~0.4 higher in log10 space.
    This script uses Copernicus as primary source. If the target month is not yet in
    Copernicus, it falls back to the latest available Copernicus month as proxy.
    VIIRS with bias correction is last resort only.

    S1 decision month = March. Rerun when March data becomes available.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# === PATHS ===
BASE = Path(__file__).resolve().parent.parent
DATA_CURRENT = BASE / "data" / "current"
DATA_BASELINE = BASE / "data" / "baseline"
DATA_BASELINE_V2 = BASE / "data" / "baseline_v2"
DATA_PROCESSED = BASE / "data" / "processed"
DATA_EXTERNAL = BASE / "data" / "external"
FEATURE_CSV = DATA_EXTERNAL / "paews_feature_matrix.csv"
CHL_COPERNICUS = DATA_EXTERNAL / "chl_copernicus_full.nc"

# === MODEL v2: 3 features, risk tiers ===
FEATURE_COLS = ['sst_z', 'chl_z', 'nino12_t1']
VIIRS_BIAS_CORRECTION = -0.4  # log10 offset: VIIRS reads ~0.4 higher than Copernicus
N_BOOTSTRAP = 2000

# Risk tiers calibrated from LOO (Sessions 12-13)
# SEVERE: 100% historical disruption rate, 0 false positives
# LOW/MODERATE/ELEVATED: ~25-33% disruption rate (model cannot separate)
RISK_TIERS = [
    (0.00, 0.20, "LOW",      "No significant environmental stress"),
    (0.20, 0.50, "MODERATE", "Some stress indicators; season likely proceeds"),
    (0.50, 0.70, "ELEVATED", "Notable stress; monitor closely"),
    (0.70, 1.01, "SEVERE",   "Extreme stress; cancellation risk high (100% hist.)"),
]

# === LEGACY: old 5-feature model for comparison ===
LEGACY_FEATURE_COLS = ['sst_z', 'chl_z', 'nino12_t1', 'is_summer', 'bio_thresh_pct']
LEGACY_THRESHOLD = 0.38


def compute_sst_features(target_month=2):
    """
    Compute SST Z-score and bio threshold from sst_current.nc.
    Uses last available day in target month, pixel-wise Z-score, spatial mean.
    """
    print("  Loading SST climatology...", flush=True)
    clim = xr.open_dataset(DATA_PROCESSED / "sst_climatology_v2.nc")

    # Try current first, then baseline
    sst_path = DATA_CURRENT / "sst_current.nc"
    if not sst_path.exists():
        sst_path = DATA_BASELINE / "sst_2026.nc"
    if not sst_path.exists():
        sst_path = DATA_BASELINE_V2 / "sst_2026.nc"
    if not sst_path.exists():
        print("  ERROR: No 2026 SST data found", flush=True)
        return None, None, None

    print("  Loading SST from %s..." % sst_path.name, flush=True)
    ds = xr.open_dataset(sst_path)

    # Filter to target month
    time_index = pd.DatetimeIndex(ds["time"].values)
    month_data = ds.sel(time=time_index.month == target_month)

    if len(month_data.time) == 0:
        print("  No data for month %d, using latest available month" % target_month, flush=True)
        latest_month = int(time_index[-1].month)
        month_data = ds.sel(time=time_index.month == latest_month)
        print("  Using month %d instead" % latest_month, flush=True)

    # Last timestep
    sst_snap = month_data["sst"].isel(time=-1).squeeze()
    if 'zlev' in sst_snap.dims:
        sst_snap = sst_snap.isel(zlev=0)

    last_date = str(month_data.time.values[-1])[:10]
    print("  SST snapshot: %s, shape: %s" % (last_date, sst_snap.shape), flush=True)

    # Climatology
    actual_month = int(pd.Timestamp(month_data.time.values[-1]).month)
    clim_mean = clim["sst_mean"].sel(month=actual_month)
    clim_std = clim["sst_std"].sel(month=actual_month)
    std_safe = clim_std.where(clim_std > 0.01, 0.01)

    # Z-score
    z_map = (sst_snap - clim_mean) / std_safe
    sst_z = float(z_map.mean(skipna=True))

    # MHW percentage
    valid_z = z_map.where(z_map.notnull())
    total_valid = int(valid_z.notnull().sum())
    mhw_pct = int((valid_z > 1.28).sum()) / total_valid * 100 if total_valid > 0 else 0

    # Bio threshold (still computed for legacy comparison)
    valid_sst = sst_snap.notnull()
    total_valid_sst = int(valid_sst.sum())
    bio_thresh_pct = float((sst_snap > 23.0).sum()) / total_valid_sst * 100 if total_valid_sst > 0 else 0

    ds.close()
    clim.close()

    print("  SST Z-score:  %+.3f" % sst_z, flush=True)
    print("  MHW pixels:   %.1f%%" % mhw_pct, flush=True)
    print("  Bio > 23C:    %.1f%%" % bio_thresh_pct, flush=True)

    return sst_z, bio_thresh_pct, mhw_pct


def compute_chl_features(target_year, target_month):
    """
    Compute Chl Z-score. Tries Copernicus first (same sensor as training),
    falls back to VIIRS with bias correction.
    """
    print("  Loading Chl climatology...", flush=True)
    clim_path = DATA_PROCESSED / "chl_climatology_copernicus.nc"
    if not clim_path.exists():
        clim_path = DATA_PROCESSED / "chl_climatology_v2.nc"
    clim = xr.open_dataset(clim_path)

    chl_z = None
    source_used = None

    # === TRY 1: Copernicus monthly (preferred) ===
    if CHL_COPERNICUS.exists():
        cop = xr.open_dataset(CHL_COPERNICUS)
        time_str = '%d-%02d' % (target_year, target_month)
        try:
            csel = cop.sel(time=time_str)
            if len(csel.time) > 0:
                chl_z = _compute_chl_z_from_copernicus(csel, clim, target_month)
                source_used = "Copernicus %s" % time_str
        except (KeyError, ValueError):
            pass

        # If exact month not available, try most recent month as proxy
        if chl_z is None:
            cop_times = pd.DatetimeIndex(cop.time.values)
            latest = cop_times[-1]
            proxy_month = latest.month
            proxy_year = latest.year
            print("  Copernicus %s not available" % time_str, flush=True)
            print("  Using proxy: Copernicus %d-%02d" % (proxy_year, proxy_month), flush=True)
            csel = cop.sel(time='%d-%02d' % (proxy_year, proxy_month))
            if len(csel.time) > 0:
                chl_z = _compute_chl_z_from_copernicus(csel, clim, proxy_month)
                source_used = "Copernicus %d-%02d (proxy for %s)" % (proxy_year, proxy_month, time_str)

        cop.close()

    # === TRY 2: VIIRS daily with bias correction (last resort) ===
    if chl_z is None:
        viirs_path = DATA_CURRENT / "chlorophyll_current.nc"
        if viirs_path.exists():
            print("  Falling back to VIIRS with bias correction (%+.1f log10)" % VIIRS_BIAS_CORRECTION, flush=True)
            chl_z = _compute_chl_z_from_viirs(viirs_path, clim, target_month)
            if chl_z is not None:
                source_used = "VIIRS + bias correction (%+.1f)" % VIIRS_BIAS_CORRECTION

    clim.close()

    if chl_z is not None:
        print("  Chl Z-score:  %+.3f (source: %s)" % (chl_z, source_used), flush=True)
    else:
        print("  WARNING: No Chl data available. Using 0.0 (neutral)", flush=True)
        chl_z = 0.0
        source_used = "none (default 0.0)"

    return chl_z, source_used


def _compute_chl_z_from_copernicus(csel, clim, month):
    """Compute Chl Z from Copernicus monthly data."""
    chl_log = np.log10(csel['CHL'].where(csel['CHL'] > 0)).squeeze()

    clim_mean = clim['chl_log_mean'].sel(month=month)
    clim_std = clim['chl_log_std'].sel(month=month)
    std_safe = clim_std.where(clim_std > 0.01, 0.01)

    chl_lat = [d for d in chl_log.dims if 'lat' in d.lower()][0]
    chl_lon = [d for d in chl_log.dims if 'lon' in d.lower()][0]
    clim_lat = [d for d in clim_mean.dims if 'lat' in d.lower()][0]
    clim_lon = [d for d in clim_mean.dims if 'lon' in d.lower()][0]

    try:
        ci = chl_log.interp({chl_lat: clim_mean[clim_lat], chl_lon: clim_mean[clim_lon]})
        z_map = (ci - clim_mean) / std_safe
        return float(z_map.mean(skipna=True))
    except Exception as e:
        print("  Copernicus grid interp failed: %s" % e, flush=True)
        return None


def _compute_chl_z_from_viirs(viirs_path, clim, target_month):
    """Compute Chl Z from VIIRS daily data with bias correction."""
    ds = xr.open_dataset(viirs_path)
    time_index = pd.DatetimeIndex(ds["time"].values)
    month_data = ds.sel(time=time_index.month == target_month)

    if len(month_data.time) == 0:
        month_data = ds  # use all available

    chl_var = 'chlor_a'
    for candidate in ['CHL', 'chl', 'chlorophyll']:
        if candidate in month_data:
            chl_var = candidate
            break

    chl_monthly = month_data[chl_var].mean(dim='time').squeeze()
    if 'altitude' in chl_monthly.dims:
        chl_monthly = chl_monthly.isel(altitude=0)

    # Apply bias correction in log10 space
    chl_log = np.log10(chl_monthly.where(chl_monthly > 0)) + VIIRS_BIAS_CORRECTION

    clim_mean = clim['chl_log_mean'].sel(month=target_month)
    clim_std = clim['chl_log_std'].sel(month=target_month)
    std_safe = clim_std.where(clim_std > 0.01, 0.01)

    chl_lat = [d for d in chl_log.dims if 'lat' in d.lower()][0]
    chl_lon = [d for d in chl_log.dims if 'lon' in d.lower()][0]
    clim_lat = [d for d in clim_mean.dims if 'lat' in d.lower()][0]
    clim_lon = [d for d in clim_mean.dims if 'lon' in d.lower()][0]

    try:
        ci = chl_log.interp({chl_lat: clim_mean[clim_lat], chl_lon: clim_mean[clim_lon]})
        z_map = (ci - clim_mean) / std_safe
        ds.close()
        return float(z_map.mean(skipna=True))
    except Exception as e:
        print("  VIIRS grid interp failed: %s" % e, flush=True)
        ds.close()
        return None


def get_nino12_t1():
    """Get latest Nino 1+2 anomaly."""
    nino = pd.read_csv(DATA_EXTERNAL / "nino_indices_monthly.csv")

    # Try Feb 2026, then Jan, then latest
    for yr, mo in [(2026, 2), (2026, 1)]:
        row = nino[(nino.year == yr) & (nino.month == mo)]
        if len(row) > 0:
            val = float(row['nino12_anom'].iloc[0])
            print("  Nino 1+2 (%d-%02d): %+.2f" % (yr, mo, val), flush=True)
            return val, "%d-%02d" % (yr, mo)

    latest = nino.dropna(subset=['nino12_anom']).iloc[-1]
    val = float(latest['nino12_anom'])
    label = "%d-%02d" % (int(latest['year']), int(latest['month']))
    print("  Nino 1+2 (latest: %s): %+.2f" % (label, val), flush=True)
    return val, label


def get_risk_tier(prob):
    """Return tier name and description for a probability."""
    for lo, hi, name, desc in RISK_TIERS:
        if lo <= prob < hi:
            return name, desc
    return "UNKNOWN", ""


def run_prediction(features, feat_cols, n_boot=N_BOOTSTRAP):
    """Train on all data, predict, bootstrap CI."""
    train_df = pd.read_csv(FEATURE_CSV)
    train_df = train_df.dropna(subset=feat_cols + ['target'])

    X_train = train_df[feat_cols].values
    y_train = train_df['target'].values

    n_pos = int(y_train.sum())
    print("  Training: %d samples (%d positives, %d features)" % (
        len(X_train), n_pos, len(feat_cols)), flush=True)

    # Point estimate
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_s, y_train)

    x_new = np.array([[features[f] for f in feat_cols]])
    x_new_s = scaler.transform(x_new)
    prob_point = model.predict_proba(x_new_s)[0, 1]

    coefs = model.coef_[0]
    contributions = coefs * x_new_s[0]

    # Bootstrap
    rng = np.random.RandomState(42)
    boot_probs = []
    for _ in range(n_boot):
        idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        X_b, y_b = X_train[idx], y_train[idx]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue
        sc = StandardScaler()
        m = LogisticRegression(max_iter=1000, solver='lbfgs')
        m.fit(sc.fit_transform(X_b), y_b)
        p = m.predict_proba(sc.transform(x_new))[0, 1]
        boot_probs.append(p)

    boot_probs = np.array(boot_probs)

    return prob_point, coefs, x_new[0], x_new_s[0], contributions, model.intercept_[0], boot_probs


def main():
    print("=" * 64)
    print("  PAEWS 2026 S1 PREDICTION (Model v2: 3-feature)")
    print("  Run date: %s" % datetime.now().strftime('%Y-%m-%d %H:%M'))
    print("  Features: sst_z, chl_z, nino12_t1")
    print("  Validation: LOO ROC-AUC 0.629, SEVERE tier 4/4 (100%%)")
    print("=" * 64)

    # ---- SST ----
    print("\nSTEP 1: SST Features")
    print("-" * 40)
    sst_z, bio_thresh_pct, mhw_pct = compute_sst_features(target_month=2)
    if sst_z is None:
        print("FATAL: No SST data.", flush=True)
        return

    # ---- Chl ----
    print("\nSTEP 2: Chlorophyll Features")
    print("-" * 40)
    chl_z, chl_source = compute_chl_features(2026, 2)

    # ---- Nino ----
    print("\nSTEP 3: Nino 1+2 Index")
    print("-" * 40)
    nino12, nino_label = get_nino12_t1()

    # ---- Feature Summary ----
    features = {
        'sst_z': sst_z,
        'chl_z': chl_z,
        'nino12_t1': nino12,
        # Legacy features (for comparison only)
        'is_summer': 1.0,
        'bio_thresh_pct': bio_thresh_pct,
    }

    hist = pd.read_csv(FEATURE_CSV)

    print("\nSTEP 4: Feature Summary")
    print("-" * 40)
    print("  %-18s %10s  %20s  %s" % ("Feature", "2026 S1", "Training Range", "Source"))
    print("  %s %s  %s  %s" % ("-" * 18, "-" * 10, "-" * 20, "-" * 30))
    sources = {
        'sst_z': "sst_current.nc",
        'chl_z': chl_source,
        'nino12_t1': "CPC %s" % nino_label,
    }
    for feat in FEATURE_COLS:
        val = features[feat]
        col = hist[feat].dropna()
        rng = "[%+.2f, %+.2f]" % (col.min(), col.max())
        flag = " !!!" if val > col.max() or val < col.min() else ""
        print("  %-18s %+10.3f  %20s%s  %s" % (feat, val, rng, flag, sources[feat]))

    # ---- v2 Prediction (3 features) ----
    print("\nSTEP 5: Model v2 Prediction (3 features)")
    print("-" * 40)
    (prob, coefs, x_raw, x_scaled, contribs,
     intercept, boot_probs) = run_prediction(features, FEATURE_COLS)

    tier, tier_desc = get_risk_tier(prob)

    print()
    print("=" * 64)
    print("  2026 S1 RISK INDEX: %.3f  [%s]" % (prob, tier))
    print("  %s" % tier_desc)
    print("=" * 64)

    # Feature contributions
    print("\n  Feature Contributions (log-odds):")
    print("  %-18s %8s %8s %8s %10s" % ("Feature", "Raw", "Scaled", "Coef", "Contrib"))
    print("  %s %s %s %s %s" % ("-" * 18, "-" * 8, "-" * 8, "-" * 8, "-" * 10))
    for i, feat in enumerate(FEATURE_COLS):
        d = "+" if contribs[i] > 0 else ""
        print("  %-18s %8.3f %8.3f %8.3f %s%9.3f" % (
            feat, x_raw[i], x_scaled[i], coefs[i], d, contribs[i]))
    print("  %-18s %8s %8s %8s %+10.3f" % ("intercept", "", "", "", intercept))

    # Bootstrap with tier distribution
    median_p = np.median(boot_probs)
    ci_lo = np.percentile(boot_probs, 2.5)
    ci_hi = np.percentile(boot_probs, 97.5)
    iqr_lo = np.percentile(boot_probs, 25)
    iqr_hi = np.percentile(boot_probs, 75)

    print("\n  Bootstrap (%d resamples):" % len(boot_probs))
    print("    Median:    %.3f" % median_p)
    print("    95%% CI:    [%.3f, %.3f]" % (ci_lo, ci_hi))
    print("    IQR:       [%.3f, %.3f]" % (iqr_lo, iqr_hi))

    print("\n  Bootstrap tier distribution:")
    for lo, hi, name, _ in RISK_TIERS:
        pct = ((boot_probs >= lo) & (boot_probs < hi)).mean() * 100
        bar = "#" * int(pct / 2)
        print("    %-10s %5.1f%% %s" % (name, pct, bar))

    # ---- Legacy comparison (5 features) ----
    print("\n  LEGACY COMPARISON (5-feature model, threshold=0.38):")
    print("  " + "-" * 50)
    (prob_leg, _, _, _, _, _, boot_leg) = run_prediction(features, LEGACY_FEATURE_COLS)
    leg_status = "AT RISK" if prob_leg >= LEGACY_THRESHOLD else "NORMAL"
    leg_med = np.median(boot_leg)
    leg_lo = np.percentile(boot_leg, 2.5)
    leg_hi = np.percentile(boot_leg, 97.5)
    print("    Probability: %.3f [%s]" % (prob_leg, leg_status))
    print("    Bootstrap median: %.3f, 95%% CI: [%.3f, %.3f]" % (leg_med, leg_lo, leg_hi))

    # ---- Historical S1 comparison ----
    print("\n  Historical S1 Comparison:")
    s1 = hist[hist['season'] == 1].dropna(subset=['sst_z', 'chl_z'])
    for _, row in s1.iterrows():
        yr = int(row['year'])
        t = int(row['target'])
        marker = " <-- DISRUPTED" if t == 1 else ""
        print("    %d: SST_Z=%+.2f Chl_Z=%+.2f Nino=%+.2f%s" % (
            yr, row['sst_z'], row['chl_z'], row['nino12_t1'], marker))
    print("    2026: SST_Z=%+.2f Chl_Z=%+.2f Nino=%+.2f  [%s, p=%.3f]" % (
        sst_z, chl_z, nino12, tier, prob))

    # ---- Interpretation ----
    print("\n" + "=" * 64)
    print("  INTERPRETATION")
    print("=" * 64)
    print()
    if tier == "SEVERE":
        print("  ACTION: High confidence disruption signal.")
        print("  Historical SEVERE tier: 4/4 disrupted, 0 false positives.")
        print("  Recommend: prepare for reduced/cancelled season.")
    elif tier in ["ELEVATED", "MODERATE"]:
        print("  MONITOR: Environmental stress present but not extreme.")
        print("  Historical %s tier: ~25-30%% disruption rate." % tier)
        print("  Model cannot reliably distinguish outcomes in this range.")
        print("  Biomass state and IMARPE survey will be decisive.")
    else:
        print("  LOW RISK: Minimal environmental stress detected.")
        print("  Historical LOW tier: ~30%% disruption rate (mostly")
        print("  from biomass-driven events the model cannot see).")

    print()
    print("  Model honest limitations:")
    print("    - 32 training samples, 3 features")
    print("    - LOO ROC-AUC = 0.629 (moderate discrimination)")
    print("    - Strong at extremes (SEVERE = 100%% disruption)")
    print("    - Weak in middle tiers (cannot see biomass state)")
    print("    - Multicollinearity fixed (dropped is_summer, bio_thresh)")

    # Data provenance
    try:
        sst_ds = xr.open_dataset(DATA_CURRENT / "sst_current.nc")
        sst_last = str(pd.DatetimeIndex(sst_ds.time.values)[-1])[:10]
        sst_ds.close()
    except Exception:
        sst_last = "unknown"

    print("\n  Data provenance:")
    print("    Chl source:  %s" % chl_source)
    print("    Nino source: CPC %s" % nino_label)
    print("    SST through: %s" % sst_last)

    print("\n  Next decision point: ~March 15")
    print("    - Copernicus Feb Chl resolves Chl uncertainty")
    print("    - CPC Feb Nino 1+2 shows if Costero confirmed")
    print("    - March SST confirms or reverses warming trend")
    print("    - If prob crosses 0.70: SEVERE tier -> act")
    print("    - Rerun: python scripts/predict_2026_s1.py")
    print("=" * 64)


if __name__ == "__main__":
    main()
