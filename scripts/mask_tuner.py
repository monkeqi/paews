"""
PAEWS Coastal Mask Tuner
=========================
Tests multiple PRODUCTIVE_PERCENTILE values to find the sweet spot
that maximizes Chl signal (and PR-AUC) without introducing pixel noise.

What it does:
  For each percentile (30, 40, 50, 60, 70):
    1. Builds a productive mask from the Copernicus climatology
    2. Recomputes Chl Z-scores for all 30 seasons using only masked pixels
    3. Runs the 5-feature logistic regression (LOO CV)
    4. Reports PR-AUC, Chl importance, key season probabilities

Run from scripts/:
    python mask_tuner.py

Requires: chl_copernicus_full.nc, chl_climatology_copernicus.nc,
          paews_feature_matrix.csv, imarpe_ground_truth.csv
"""

import sys
print("PAEWS Mask Tuner starting...", flush=True)

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
except ImportError:
    print("ERROR: scikit-learn required. pip install scikit-learn", flush=True)
    sys.exit(1)

print("Imports done", flush=True)

BASE_DIR = Path("c:/Users/josep/Documents/paews")
DATA_EXTERNAL = BASE_DIR / "data" / "external"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

CHL_FULL_PATH = DATA_EXTERNAL / "chl_copernicus_full.nc"
CLIM_PATH = DATA_PROCESSED / "chl_climatology_copernicus.nc"
FEAT_PATH = DATA_EXTERNAL / "paews_feature_matrix.csv"
GT_PATH = DATA_EXTERNAL / "imarpe_ground_truth.csv"

MIN_OBS = 5
PERCENTILES_TO_TEST = [30, 35, 40, 45, 50, 55, 60, 70]


def load_data():
    """Load all required data."""
    if not CHL_FULL_PATH.exists():
        print(f"  ERROR: {CHL_FULL_PATH} not found", flush=True)
        sys.exit(1)
    if not CLIM_PATH.exists():
        print(f"  ERROR: {CLIM_PATH} not found", flush=True)
        sys.exit(1)
    
    ds = xr.open_dataset(CHL_FULL_PATH)
    clim = xr.open_dataset(CLIM_PATH)
    gt = pd.read_csv(GT_PATH)
    feat = pd.read_csv(FEAT_PATH)
    
    print(f"  Chl data: {ds.sizes['time']} months", flush=True)
    print(f"  Ground truth: {len(gt)} seasons", flush=True)
    
    return ds, clim, gt, feat


def build_mask(clim, percentile):
    """
    Build a productive pixel mask at the given percentile.
    
    percentile=50 means keep top 50% most productive pixels.
    Lower percentile = tighter mask = more coastal focus but fewer pixels.
    """
    annual_mean = clim['chl_log_mean'].mean(dim='month')
    valid_vals = annual_mean.values[~np.isnan(annual_mean.values)]
    threshold = float(np.nanpercentile(valid_vals, 100 - percentile))
    mask = annual_mean >= threshold
    n_pixels = int(mask.sum())
    total_pixels = int(annual_mean.notnull().sum())
    return mask, n_pixels, total_pixels, threshold


def extract_chl_z(ds, clim, year, month, mask, total_pixels):
    """Compute masked Chl Z-score for a season."""
    try:
        month_data = ds.sel(time=f"{year}-{month:02d}")
        if 'time' in month_data["CHL"].dims:
            chl_snap = month_data["CHL"].isel(time=0).squeeze()
        else:
            chl_snap = month_data["CHL"].squeeze()
    except (KeyError, IndexError):
        return np.nan
    
    chl_log = np.log10(chl_snap.where(chl_snap > 0))
    clim_mean = clim['chl_log_mean'].sel(month=month)
    clim_std = clim['chl_log_std'].sel(month=month)
    obs_count = clim['chl_obs_count'].sel(month=month)
    
    std_safe = clim_std.where(clim_std > 0.01)
    z = (chl_log - clim_mean) / std_safe
    z = z.where(obs_count >= MIN_OBS)
    
    # Apply mask
    z_masked = z.where(mask)
    return float(z_masked.mean(skipna=True))


def run_regression(feat_df):
    """
    Run 5-feature LOO logistic regression.
    Returns PR-AUC, ROC-AUC, coefficients, and per-season probabilities.
    """
    feature_cols = ['sst_z', 'chl_z', 'nino12_t1', 'is_summer', 'bio_thresh_pct']
    df = feat_df.dropna(subset=['sst_z', 'chl_z', 'nino12_t1', 'target']).copy()
    df['bio_thresh_pct'] = df['bio_thresh_pct'].fillna(0)
    df['is_summer'] = df['is_summer'].fillna(0)
    
    if len(df) < 10:
        return None
    
    X = df[feature_cols].values.copy()
    y = df['target'].values
    X[:, 1] = -X[:, 1]  # flip Chl
    
    # LOO CV
    loo = LeaveOneOut()
    y_probs = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X[train_idx])
        X_test_s = scaler.transform(X[test_idx])
        model = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
        model.fit(X_train_s, y[train_idx])
        y_probs[test_idx] = model.predict_proba(X_test_s)[:, 1]
    
    # Final model for coefficients
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    model_final = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
    model_final.fit(X_scaled, y)
    
    coefs = model_final.coef_[0]
    abs_coefs = np.abs(coefs)
    rel_weights = abs_coefs / abs_coefs.sum()
    
    # Metrics
    roc = roc_auc_score(y, y_probs)
    prec, rec, thr = precision_recall_curve(y, y_probs)
    pr_auc = auc(rec, prec)
    
    # Best threshold
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    best_idx = np.argmax(f1)
    best_thresh = thr[best_idx] if best_idx < len(thr) else 0.5
    
    # Recall at best threshold
    y_pred = (y_probs >= best_thresh).astype(int)
    caught = ((y_pred == 1) & (y == 1)).sum()
    total_risk = y.sum()
    fp = ((y_pred == 1) & (y == 0)).sum()
    
    return {
        'pr_auc': pr_auc,
        'roc_auc': roc,
        'coefs': coefs,
        'rel_weights': rel_weights,
        'y_probs': y_probs,
        'y_true': y,
        'df': df,
        'best_thresh': best_thresh,
        'recall': caught,
        'total_risk': total_risk,
        'false_pos': fp,
    }


if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("PAEWS COASTAL MASK TUNER", flush=True)
    print(f"Testing percentiles: {PERCENTILES_TO_TEST}", flush=True)
    print("=" * 70, flush=True)
    
    # Load data
    print("\nLoading data...", flush=True)
    ds, clim, gt, feat = load_data()
    
    # Track results
    all_results = []
    
    for pct in PERCENTILES_TO_TEST:
        print(f"\n{'='*70}", flush=True)
        print(f"PERCENTILE = {pct} (top {pct}% most productive pixels)", flush=True)
        print(f"{'='*70}", flush=True)
        
        # Build mask
        mask, n_pixels, total_pixels, threshold = build_mask(clim, pct)
        print(f"  Mask: {n_pixels} / {total_pixels} pixels "
              f"(threshold: log10(Chl) >= {threshold:.3f}, "
              f"Chl >= {10**threshold:.3f} mg/m³)", flush=True)
        
        # Recompute Chl Z-scores for all seasons
        feat_copy = feat.copy()
        
        for i, row in gt.iterrows():
            year = int(row['year'])
            season = int(row['season'])
            decision_month = 3 if season == 1 else 10
            
            chl_z = extract_chl_z(ds, clim, year, decision_month, mask, total_pixels)
            
            # Update feature matrix copy
            idx = feat_copy[(feat_copy['year'] == year) & (feat_copy['season'] == season)].index
            if len(idx) > 0:
                feat_copy.loc[idx[0], 'chl_z'] = chl_z
                
                # Recompute hardcoded composite
                sst = feat_copy.loc[idx[0], 'sst_z']
                nino = feat_copy.loc[idx[0], 'nino12_t1']
                if not pd.isna(sst) and not pd.isna(chl_z):
                    nino_val = nino if not pd.isna(nino) else 0
                    feat_copy.loc[idx[0], 'composite_hard'] = (
                        0.4 * sst + 0.4 * (-chl_z) + 0.2 * nino_val
                    )
        
        # Run regression
        result = run_regression(feat_copy)
        
        if result is None:
            print(f"  SKIP: insufficient valid rows", flush=True)
            continue
        
        w = result['rel_weights']
        print(f"\n  PR-AUC:  {result['pr_auc']:.3f}", flush=True)
        print(f"  ROC-AUC: {result['roc_auc']:.3f}", flush=True)
        print(f"  Recall:  {result['recall']}/{result['total_risk']} "
              f"({result['recall']/result['total_risk']:.0%}), "
              f"FP={result['false_pos']}", flush=True)
        print(f"  Weights: SST={w[0]:.1%}  Chl={w[1]:.1%}  Niño={w[2]:.1%}  "
              f"Season={w[3]:.1%}  Bio={w[4]:.1%}", flush=True)
        
        # Key seasons
        df_r = result['df']
        probs = result['y_probs']
        print(f"\n  KEY SEASONS:", flush=True)
        for label, yr, sn in [("2023 S1 (CANCELLED)", 2023, 1),
                               ("2023 S2 (DISRUPTED)", 2023, 2),
                               ("2022 S2 (DISRUPTED)", 2022, 2),
                               ("2014 S1 (DISRUPTED)", 2014, 1),
                               ("2018 S2 (NORMAL-FA)", 2018, 2),
                               ("2019 S1 (NORMAL-FA)", 2019, 1),
                               ("2020 S1 (NORMAL-FA)", 2020, 1)]:
            match = df_r[(df_r['year'] == yr) & (df_r['season'] == sn)]
            if len(match) > 0:
                idx_pos = df_r.index.get_loc(match.index[0])
                p = probs[idx_pos]
                cz = feat_copy[(feat_copy['year'] == yr) & (feat_copy['season'] == sn)].iloc[0]['chl_z']
                print(f"    {label}: prob={p:.2f}  chl_z={cz:+.2f}", flush=True)
        
        all_results.append({
            'percentile': pct,
            'n_pixels': n_pixels,
            'threshold': threshold,
            'pr_auc': result['pr_auc'],
            'roc_auc': result['roc_auc'],
            'chl_weight': w[1],
            'recall': result['recall'],
            'false_pos': result['false_pos'],
            'p_2023s1': probs[df_r.index.get_loc(df_r[(df_r['year']==2023)&(df_r['season']==1)].index[0])] if len(df_r[(df_r['year']==2023)&(df_r['season']==1)]) > 0 else np.nan,
            'p_2022s2': probs[df_r.index.get_loc(df_r[(df_r['year']==2022)&(df_r['season']==2)].index[0])] if len(df_r[(df_r['year']==2022)&(df_r['season']==2)]) > 0 else np.nan,
            'p_2014s1': probs[df_r.index.get_loc(df_r[(df_r['year']==2014)&(df_r['season']==1)].index[0])] if len(df_r[(df_r['year']==2014)&(df_r['season']==1)]) > 0 else np.nan,
        })
    
    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print(f"\n{'='*70}", flush=True)
    print("COMPARISON TABLE", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Pct':>4} {'Pixels':>7} {'PR-AUC':>7} {'ROC':>5} {'Chl%':>5} "
          f"{'Recall':>6} {'FP':>3} {'2023S1':>7} {'2022S2':>7} {'2014S1':>7}", flush=True)
    print("-" * 70, flush=True)
    
    best_pr = 0
    best_pct = None
    
    for r in all_results:
        marker = ""
        if r['pr_auc'] > best_pr:
            best_pr = r['pr_auc']
            best_pct = r['percentile']
        print(f"{r['percentile']:>4} {r['n_pixels']:>7} {r['pr_auc']:>7.3f} "
              f"{r['roc_auc']:>5.3f} {r['chl_weight']:>5.1%} "
              f"{r['recall']:>4}/{int(all_results[0].get('recall',0)+all_results[0].get('false_pos',0)):>1} "
              f"{r['false_pos']:>3} "
              f"{r['p_2023s1']:>7.2f} {r['p_2022s2']:>7.2f} {r['p_2014s1']:>7.2f}", flush=True)
    
    print(f"\n  ★ BEST: percentile={best_pct}, PR-AUC={best_pr:.3f}", flush=True)
    
    # Recommendation
    print(f"\n{'='*70}", flush=True)
    print("RECOMMENDATION", flush=True)
    print(f"{'='*70}", flush=True)
    
    if best_pct != 50:
        print(f"  Change PRODUCTIVE_PERCENTILE from 50 to {best_pct} in chl_migration.py", flush=True)
        print(f"  Then rerun:", flush=True)
        print(f"    python chl_migration.py", flush=True)
        print(f"    python composite_score.py", flush=True)
    else:
        print(f"  Current setting (50) is already optimal.", flush=True)
        print(f"  Focus on biomass integration for the next improvement.", flush=True)
    
    print(f"\n  NOTE: If a tighter mask (30-40%) wins on PR-AUC but shows", flush=True)
    print(f"  unstable Chl Z-scores (big swings), prefer the more conservative", flush=True)
    print(f"  option. With only 28 samples, stability matters more than", flush=True)
    print(f"  squeezing an extra 0.01 PR-AUC.", flush=True)
    print("=" * 70, flush=True)
    
    ds.close()
