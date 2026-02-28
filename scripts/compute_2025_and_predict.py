"""
compute_2025_and_predict.py
Computes 2025 S1/S2 features, adds them to training data,
retrains model, runs bootstrap CI for 2026 S1 prediction.

Usage: python scripts/compute_2025_and_predict.py
"""
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
FEAT_COLS = ['sst_z', 'chl_z', 'nino12_t1', 'is_summer', 'bio_thresh_pct']

def compute_2025_features():
    """Compute 2025 S1 and S2 features from Copernicus + OISST (same sources as training)."""
    sst_clim = xr.open_dataset(BASE / "data/processed/sst_climatology_v2.nc")
    chl_clim = xr.open_dataset(BASE / "data/processed/chl_climatology_copernicus.nc")
    cop = xr.open_dataset(BASE / "data/external/chl_copernicus_full.nc")
    sst_ds = xr.open_dataset(BASE / "data/baseline/sst_2025.nc")
    nino = pd.read_csv(BASE / "data/external/nino_indices_monthly.csv")
    gt = pd.read_csv(BASE / "data/external/imarpe_ground_truth.csv")

    rows = []
    for season, dec_month in [(1, 3), (2, 10)]:
        print(f"\n--- 2025 S{season} (decision month = {dec_month}) ---")

        # SST Z-score
        ti = pd.DatetimeIndex(sst_ds.time.values)
        md = sst_ds.sel(time=ti.month == dec_month)
        snap = md['sst'].isel(time=-1).squeeze()
        if 'zlev' in snap.dims:
            snap = snap.isel(zlev=0)
        cm = sst_clim['sst_mean'].sel(month=dec_month)
        cs = sst_clim['sst_std'].sel(month=dec_month)
        ss = cs.where(cs > 0.01, 0.01)
        z = (snap - cm) / ss
        sst_z = float(z.mean(skipna=True))

        # Bio threshold
        valid_sst = snap.notnull()
        above_23 = (snap > 23.0) & valid_sst
        bio = float(above_23.sum()) / float(valid_sst.sum()) * 100

        # MHW
        mhw_count = int((z > 1.28).sum())
        total_z = int(z.notnull().sum())
        mhw_pct = mhw_count / total_z * 100 if total_z > 0 else 0

        print(f"  SST Z: {sst_z:+.3f}, Bio>23: {bio:.1f}%, MHW: {mhw_pct:.1f}%")

        # Chl Z-score from Copernicus (same sensor as training)
        csel = cop.sel(time=f'2025-{dec_month:02d}')
        if len(csel.time) == 0:
            print(f"  WARNING: No Copernicus data for 2025-{dec_month:02d}")
            chl_z = np.nan
        else:
            clog = np.log10(csel['CHL'].where(csel['CHL'] > 0)).squeeze()
            chl_lat = [d for d in clog.dims if 'lat' in d.lower()][0]
            chl_lon = [d for d in clog.dims if 'lon' in d.lower()][0]
            clim_lat = [d for d in chl_clim['chl_log_mean'].dims if 'lat' in d.lower()][0]
            clim_lon = [d for d in chl_clim['chl_log_mean'].dims if 'lon' in d.lower()][0]
            ci = clog.interp({chl_lat: chl_clim[clim_lat], chl_lon: chl_clim[clim_lon]})
            clm = chl_clim['chl_log_mean'].sel(month=dec_month)
            cls = chl_clim['chl_log_std'].sel(month=dec_month)
            cls_safe = cls.where(cls > 0.01, 0.01)
            chl_z = float(((ci - clm) / cls_safe).mean(skipna=True))

            # Low-chl pct
            cz_map = (ci - clm) / cls_safe
            lchl_pct = float((cz_map < -1.28).sum()) / float(cz_map.notnull().sum()) * 100
            print(f"  Chl Z: {chl_z:+.3f}, Low-Chl: {lchl_pct:.1f}%")

        # Nino 1+2 (t-1)
        nm = dec_month - 1 if dec_month > 1 else 12
        ny = 2025 if nm < dec_month else 2024
        nr = nino[(nino.year == ny) & (nino.month == nm)]
        n12 = float(nr['nino12_anom'].iloc[0]) if len(nr) > 0 else np.nan
        print(f"  Nino 1+2 (t-1): {n12:+.2f}")

        # Ground truth
        gtr = gt[(gt.year == 2025) & (gt.season == season)]
        if len(gtr) > 0:
            outcome = str(gtr.iloc[0]['outcome'])
            target = 1 if any(w in outcome.lower() for w in ['cancel', 'disrupt', 'reduced', 'suspend']) else 0
        else:
            outcome = 'NORMAL'
            target = 0
        print(f"  Outcome: {outcome}, Target: {target}")

        # Composite
        if not np.isnan(sst_z) and not np.isnan(chl_z) and not np.isnan(n12):
            composite = 0.4 * sst_z + 0.4 * (-chl_z) + 0.2 * n12
        else:
            composite = np.nan

        rows.append({
            'year': 2025, 'season': season, 'outcome': outcome, 'target': target,
            'decision_month': dec_month, 'is_summer': 1 if season == 1 else 0,
            'sst_z': sst_z, 'chl_z': chl_z, 'mhw_pct': mhw_pct,
            'lchl_pct': lchl_pct if not np.isnan(chl_z) else np.nan,
            'bio_thresh_pct': bio, 'nino12_t1': n12,
            'nino12_t2': np.nan, 'composite_hard': composite,
            'chl_source': 'copernicus', 'sla_z': np.nan, 'sla_cm': np.nan,
            'sla_pct_pos': np.nan, 'thermal_shock': np.nan,
        })

    sst_clim.close()
    chl_clim.close()
    cop.close()
    sst_ds.close()

    return pd.DataFrame(rows)


def run_bootstrap_prediction(train_df, x_2026, n_boot=2000):
    """Bootstrap resampling to get prediction interval."""
    rng = np.random.RandomState(42)
    probs = []

    for i in range(n_boot):
        idx = rng.choice(len(train_df), size=len(train_df), replace=True)
        X_b = train_df[FEAT_COLS].values[idx]
        y_b = train_df['target'].values[idx]

        # Skip if single class
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue

        scaler = StandardScaler()
        X_bs = scaler.fit_transform(X_b)
        model = LogisticRegression(max_iter=1000, solver='lbfgs')
        model.fit(X_bs, y_b)

        x_s = scaler.transform(x_2026.reshape(1, -1))
        p = model.predict_proba(x_s)[0, 1]
        probs.append(p)

    probs = np.array(probs)
    return probs


def main():
    print("=" * 64)
    print("  PAEWS: ADD 2025 + RETRAIN + BOOTSTRAP 2026 S1")
    print("=" * 64)

    # Step 1: Compute 2025 features
    print("\nSTEP 1: Computing 2025 S1 & S2 features")
    new_rows = compute_2025_features()

    # Step 2: Merge with existing training data
    print("\n\nSTEP 2: Updating feature matrix")
    print("-" * 40)
    old_df = pd.read_csv(BASE / "data/external/paews_feature_matrix.csv")

    # Remove any existing 2025 rows
    old_df = old_df[old_df.year != 2025]

    # Align columns
    for col in old_df.columns:
        if col not in new_rows.columns:
            new_rows[col] = np.nan

    updated_df = pd.concat([old_df, new_rows[old_df.columns]], ignore_index=True)
    updated_df = updated_df.sort_values(['year', 'season']).reset_index(drop=True)

    outpath = BASE / "data/external/paews_feature_matrix.csv"
    updated_df.to_csv(outpath, index=False)
    print(f"  Saved: {outpath}")
    print(f"  Total rows: {len(updated_df)} (was {len(old_df)})")

    # Step 3: Compare old vs new model
    print("\n\nSTEP 3: Model comparison (old 30 vs new 32 samples)")
    print("-" * 40)

    old_train = old_df.dropna(subset=FEAT_COLS + ['target'])
    new_train = updated_df.dropna(subset=FEAT_COLS + ['target'])

    for label, tdf in [("Old (30)", old_train), ("New (32)", new_train)]:
        X = tdf[FEAT_COLS].values
        y = tdf['target'].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000, solver='lbfgs')
        model.fit(Xs, y)

        print(f"\n  {label}: {len(X)} samples, {int(y.sum())} positives")
        print(f"  Coefficients:")
        for feat, coef in zip(FEAT_COLS, model.coef_[0]):
            print(f"    {feat:<18} {coef:+.3f}")
        print(f"    intercept          {model.intercept_[0]:+.3f}")

        # Predict key years
        for yr, s in [(2015, 1), (2020, 1), (2023, 1)]:
            row = tdf[(tdf.year == yr) & (tdf.season == s)]
            if len(row) > 0:
                xr_val = row[FEAT_COLS].values
                p = model.predict_proba(scaler.transform(xr_val))[0, 1]
                print(f"    {yr} S{s}: prob={p:.3f}")

    # Step 4: 2026 S1 prediction with corrected Chl
    print("\n\nSTEP 4: 2026 S1 Prediction (Copernicus Chl)")
    print("-" * 40)

    # Use Copernicus Feb 2025 as proxy (best available same-sensor)
    # SST from current (Feb 8, 2026)
    # Nino from latest available
    nino = pd.read_csv(BASE / "data/external/nino_indices_monthly.csv")
    latest_nino = nino.dropna(subset=['nino12_anom']).iloc[-1]
    n12 = float(latest_nino['nino12_anom'])

    x_2026 = np.array([0.618, -0.403, n12, 1.0, 78.016])
    print(f"  Features: SST_Z={x_2026[0]:+.3f} Chl_Z={x_2026[1]:+.3f} "
          f"Nino={x_2026[2]:+.2f} Summer=1 Bio={x_2026[4]:.1f}%")

    # Point estimate with new model
    scaler = StandardScaler()
    X_new = new_train[FEAT_COLS].values
    y_new = new_train['target'].values
    Xs = scaler.fit_transform(X_new)
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(Xs, y_new)

    x_s = scaler.transform(x_2026.reshape(1, -1))
    prob_point = model.predict_proba(x_s)[0, 1]
    print(f"  Point estimate: {prob_point:.3f}")

    # Step 5: Bootstrap
    print("\n\nSTEP 5: Bootstrap Prediction Interval (2000 resamples)")
    print("-" * 40)
    probs = run_bootstrap_prediction(new_train, x_2026, n_boot=2000)

    median_p = np.median(probs)
    ci_lo = np.percentile(probs, 2.5)
    ci_hi = np.percentile(probs, 97.5)
    ci_25 = np.percentile(probs, 25)
    ci_75 = np.percentile(probs, 75)
    pct_above_thresh = (probs >= 0.38).mean() * 100

    print(f"  Valid bootstrap samples: {len(probs)}")
    print(f"  Median probability: {median_p:.3f}")
    print(f"  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  IQR:    [{ci_25:.3f}, {ci_75:.3f}]")
    print(f"  % above 0.38 threshold: {pct_above_thresh:.1f}%")
    print()

    # Final summary
    print("=" * 64)
    status = "AT RISK" if prob_point >= 0.38 else "NORMAL"
    print(f"  2026 S1 PROBABILITY: {prob_point:.3f} [{status}]")
    print(f"  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Bootstrap models classifying AT RISK: {pct_above_thresh:.0f}%")
    print()
    if ci_lo < 0.38 < ci_hi:
        print("  >>> BORDERLINE: Threshold falls within confidence interval <<<")
        print("  >>> Recommend waiting for March data before final call      <<<")
    elif ci_lo >= 0.38:
        print("  >>> HIGH CONFIDENCE AT RISK <<<")
    else:
        print("  >>> LIKELY NORMAL, but monitor March data <<<")
    print("=" * 64)


if __name__ == "__main__":
    main()
