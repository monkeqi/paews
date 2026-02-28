"""
predict_2026_s1.py - PAEWS 2026 S1 Prediction

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

# === PATHS ===
BASE = Path(__file__).resolve().parent.parent
DATA_CURRENT = BASE / "data" / "current"
DATA_BASELINE = BASE / "data" / "baseline"
DATA_BASELINE_V2 = BASE / "data" / "baseline_v2"
DATA_PROCESSED = BASE / "data" / "processed"
DATA_EXTERNAL = BASE / "data" / "external"
FEATURE_CSV = DATA_EXTERNAL / "paews_feature_matrix.csv"
CHL_COPERNICUS = DATA_EXTERNAL / "chl_copernicus_full.nc"

# === CONSTANTS ===
FEATURE_COLS = ['sst_z', 'chl_z', 'nino12_t1', 'is_summer', 'bio_thresh_pct']
VIIRS_BIAS_CORRECTION = -0.4  # log10 offset: VIIRS reads ~0.4 higher than Copernicus
THRESHOLD = 0.38
N_BOOTSTRAP = 2000


def compute_sst_features(target_month=2):
    """
    Compute SST Z-score and bio threshold from sst_current.nc.
    Uses last available day in target month, pixel-wise Z-score, spatial mean.
    Matches composite_score.py methodology.
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

    print(f"  Loading SST from {sst_path.name}...", flush=True)
    ds = xr.open_dataset(sst_path)

    # Filter to target month
    time_index = pd.DatetimeIndex(ds["time"].values)
    month_data = ds.sel(time=time_index.month == target_month)

    if len(month_data.time) == 0:
        print(f"  No data for month {target_month}, using latest available month", flush=True)
        latest_month = int(time_index[-1].month)
        month_data = ds.sel(time=time_index.month == latest_month)
        print(f"  Using month {latest_month} instead", flush=True)

    # Last timestep (matches composite_score.py)
    sst_snap = month_data["sst"].isel(time=-1).squeeze()
    if 'zlev' in sst_snap.dims:
        sst_snap = sst_snap.isel(zlev=0)

    last_date = str(month_data.time.values[-1])[:10]
    print(f"  SST snapshot: {last_date}, shape: {sst_snap.shape}", flush=True)

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

    # Bio threshold
    valid_sst = sst_snap.notnull()
    total_valid_sst = int(valid_sst.sum())
    bio_thresh_pct = float((sst_snap > 23.0).sum()) / total_valid_sst * 100 if total_valid_sst > 0 else 0

    ds.close()
    clim.close()

    print(f"  SST Z-score:  {sst_z:+.3f}", flush=True)
    print(f"  MHW pixels:   {mhw_pct:.1f}%", flush=True)
    print(f"  Bio > 23C:    {bio_thresh_pct:.1f}%", flush=True)

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
        time_str = f'{target_year}-{target_month:02d}'
        try:
            csel = cop.sel(time=time_str)
            if len(csel.time) > 0:
                chl_z = _compute_chl_z_from_copernicus(csel, clim, target_month)
                source_used = f"Copernicus {time_str}"
        except (KeyError, ValueError):
            pass

        # If exact month not available, try most recent month as proxy
        if chl_z is None:
            cop_times = pd.DatetimeIndex(cop.time.values)
            latest = cop_times[-1]
            proxy_month = latest.month
            proxy_year = latest.year
            print(f"  Copernicus {time_str} not available", flush=True)
            print(f"  Using proxy: Copernicus {proxy_year}-{proxy_month:02d}", flush=True)
            csel = cop.sel(time=f'{proxy_year}-{proxy_month:02d}')
            if len(csel.time) > 0:
                chl_z = _compute_chl_z_from_copernicus(csel, clim, proxy_month)
                source_used = f"Copernicus {proxy_year}-{proxy_month:02d} (proxy for {time_str})"

        cop.close()

    # === TRY 2: VIIRS daily with bias correction (last resort) ===
    if chl_z is None:
        viirs_path = DATA_CURRENT / "chlorophyll_current.nc"
        if viirs_path.exists():
            print(f"  Falling back to VIIRS with bias correction ({VIIRS_BIAS_CORRECTION:+.1f} log10)", flush=True)
            chl_z = _compute_chl_z_from_viirs(viirs_path, clim, target_month)
            if chl_z is not None:
                source_used = f"VIIRS + bias correction ({VIIRS_BIAS_CORRECTION:+.1f})"

    clim.close()

    if chl_z is not None:
        print(f"  Chl Z-score:  {chl_z:+.3f} (source: {source_used})", flush=True)
    else:
        print(f"  WARNING: No Chl data available. Using 0.0 (neutral)", flush=True)
        chl_z = 0.0
        source_used = "none (default 0.0)"

    return chl_z, source_used


def _compute_chl_z_from_copernicus(csel, clim, month):
    """Compute Chl Z from Copernicus monthly data."""
    chl_log = np.log10(csel['CHL'].where(csel['CHL'] > 0)).squeeze()

    clim_mean = clim['chl_log_mean'].sel(month=month)
    clim_std = clim['chl_log_std'].sel(month=month)
    std_safe = clim_std.where(clim_std > 0.01, 0.01)

    # Interpolate to climatology grid
    chl_lat = [d for d in chl_log.dims if 'lat' in d.lower()][0]
    chl_lon = [d for d in chl_log.dims if 'lon' in d.lower()][0]
    clim_lat = [d for d in clim_mean.dims if 'lat' in d.lower()][0]
    clim_lon = [d for d in clim_mean.dims if 'lon' in d.lower()][0]

    try:
        ci = chl_log.interp({chl_lat: clim_mean[clim_lat], chl_lon: clim_mean[clim_lon]})
        z_map = (ci - clim_mean) / std_safe
        return float(z_map.mean(skipna=True))
    except Exception as e:
        print(f"  Copernicus grid interp failed: {e}", flush=True)
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
        print(f"  VIIRS grid interp failed: {e}", flush=True)
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
            print(f"  Nino 1+2 ({yr}-{mo:02d}): {val:+.2f}", flush=True)
            return val, f"{yr}-{mo:02d}"

    latest = nino.dropna(subset=['nino12_anom']).iloc[-1]
    val = float(latest['nino12_anom'])
    label = f"{int(latest['year'])}-{int(latest['month']):02d}"
    print(f"  Nino 1+2 (latest: {label}): {val:+.2f}", flush=True)
    return val, label


def run_prediction_with_bootstrap(features, n_boot=N_BOOTSTRAP):
    """Train, predict, and bootstrap CI."""
    train_df = pd.read_csv(FEATURE_CSV)
    train_df = train_df.dropna(subset=FEATURE_COLS + ['target'])

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['target'].values

    n_pos = int(y_train.sum())
    print(f"  Training: {len(X_train)} samples ({n_pos} positives)", flush=True)

    # Point estimate
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_s, y_train)

    x_new = np.array([[
        features['sst_z'],
        features['chl_z'],
        features['nino12_t1'],
        1.0,  # is_summer for S1
        features['bio_thresh_pct'],
    ]])
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
        X_bs = sc.fit_transform(X_b)
        m = LogisticRegression(max_iter=1000, solver='lbfgs')
        m.fit(X_bs, y_b)
        p = m.predict_proba(sc.transform(x_new))[0, 1]
        boot_probs.append(p)

    boot_probs = np.array(boot_probs)

    return (prob_point, coefs, x_new[0], x_new_s[0], contributions,
            model.intercept_[0], boot_probs)


def main():
    print("=" * 64)
    print("  PAEWS 2026 S1 PREDICTION")
    print(f"  Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Decision month: March (using best available data)")
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
        'bio_thresh_pct': bio_thresh_pct,
    }

    hist = pd.read_csv(FEATURE_CSV)

    print("\nSTEP 4: Feature Summary")
    print("-" * 40)
    print(f"  {'Feature':<18} {'2026 S1':>10}  {'Training Range':>20}  {'Source'}")
    print(f"  {'-'*18} {'-'*10}  {'-'*20}  {'-'*30}")
    sources = {
        'sst_z': "sst_current.nc",
        'chl_z': chl_source,
        'nino12_t1': f"CPC {nino_label}",
        'bio_thresh_pct': "sst_current.nc",
    }
    for feat in ['sst_z', 'chl_z', 'nino12_t1', 'bio_thresh_pct']:
        val = features[feat]
        col = hist[feat].dropna()
        rng = f"[{col.min():+.2f}, {col.max():+.2f}]"
        flag = " !!!" if val > col.max() or val < col.min() else ""
        print(f"  {feat:<18} {val:>+10.3f}  {rng:>20}{flag}  {sources[feat]}")
    print(f"  {'is_summer':<18} {'1':>10}  {'[0, 1]':>20}")

    # ---- Prediction ----
    print("\nSTEP 5: Model Prediction + Bootstrap")
    print("-" * 40)
    (prob, coefs, x_raw, x_scaled, contribs,
     intercept, boot_probs) = run_prediction_with_bootstrap(features)

    status = "AT RISK" if prob >= THRESHOLD else "NORMAL"

    print()
    print("=" * 64)
    if status == "AT RISK":
        print(f"  >>> 2026 S1 PROBABILITY: {prob:.3f}  [{status}] <<<")
    else:
        print(f"      2026 S1 PROBABILITY: {prob:.3f}  [{status}]")
    print(f"      Threshold: {THRESHOLD}")
    print("=" * 64)

    # Feature contributions
    print("\n  Feature Contributions (log-odds):")
    print(f"  {'Feature':<18} {'Raw':>8} {'Scaled':>8} {'Coef':>8} {'Contrib':>10}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for i, feat in enumerate(FEATURE_COLS):
        d = "+" if contribs[i] > 0 else ""
        print(f"  {feat:<18} {x_raw[i]:>8.3f} {x_scaled[i]:>8.3f} {coefs[i]:>8.3f} {d}{contribs[i]:>9.3f}")
    print(f"  {'intercept':<18} {'':>8} {'':>8} {'':>8} {intercept:>+10.3f}")

    # Bootstrap
    median_p = np.median(boot_probs)
    ci_lo = np.percentile(boot_probs, 2.5)
    ci_hi = np.percentile(boot_probs, 97.5)
    iqr_lo = np.percentile(boot_probs, 25)
    iqr_hi = np.percentile(boot_probs, 75)
    pct_risk = (boot_probs >= THRESHOLD).mean() * 100

    print(f"\n  Bootstrap ({len(boot_probs)} resamples):")
    print(f"    Median:    {median_p:.3f}")
    print(f"    95% CI:    [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"    IQR:       [{iqr_lo:.3f}, {iqr_hi:.3f}]")
    print(f"    % AT RISK: {pct_risk:.0f}%")

    # Historical comparison
    print("\n  Historical S1 Comparison:")
    s1 = hist[hist['season'] == 1].dropna(subset=['sst_z', 'chl_z'])
    for _, row in s1.iterrows():
        yr = int(row['year'])
        t = int(row['target'])
        marker = " <-- DISRUPTED" if t == 1 else ""
        print(f"    {yr}: SST_Z={row['sst_z']:+.2f} Chl_Z={row['chl_z']:+.2f} "
              f"Bio={row['bio_thresh_pct']:5.1f}%{marker}")
    print(f"    2026: SST_Z={sst_z:+.2f} Chl_Z={chl_z:+.2f} "
          f"Bio={bio_thresh_pct:5.1f}%  [{status}, p={prob:.3f}]")

    # Interpretation
    print("\n  INTERPRETATION:")
    if ci_lo < THRESHOLD < ci_hi:
        print("  Threshold falls within 95% CI -> BORDERLINE")
        print("  Wait for March data before final determination.")
    elif ci_lo >= THRESHOLD:
        print("  Entire 95% CI above threshold -> HIGH CONFIDENCE AT RISK")
    else:
        print("  Entire 95% CI below threshold -> LIKELY NORMAL")

    sst_ds = xr.open_dataset(DATA_CURRENT / "sst_current.nc")
    sst_last = str(pd.DatetimeIndex(sst_ds.time.values)[-1])[:10]
    sst_ds.close()

    print(f"\n  Chl source: {chl_source}")
    print(f"  Nino source: CPC {nino_label}")
    print(f"  SST through: {sst_last}")
    print(f"  Rerun after: March SST/Chl available + Feb Nino published")
    print("=" * 64)


if __name__ == "__main__":
    main()
