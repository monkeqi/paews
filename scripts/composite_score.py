"""
PAEWS Composite Score & Validation
====================================
Builds the composite anomaly score that merges SST, Chlorophyll, and Niño
indices into a single risk number, then validates against IMARPE season
outcomes.

TWO APPROACHES:
    1. Hardcoded weights (physics-based baseline):
       Composite_Z = 0.4*SST_Z + 0.4*Chl_Z + 0.2*Niño12(t-1)
       
    2. Data-driven weights (logistic regression):
       Let the IMARPE ground truth dictate optimal weights.
       Uses Precision-Recall AUC as primary metric (rare event detection).

CLASSIFICATION TARGET:
    0 = NORMAL season (quota met, no disruption)
    1 = DISRUPTED season (reduced quota, cancelled, or < 80% catch)

Usage:
    python composite_score.py              # Build scores + validate
    python composite_score.py --predict    # Score current conditions
"""

import sys
print("PAEWS Composite Score Builder starting...", flush=True)

import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("Imports done", flush=True)

# Try sklearn
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        roc_auc_score, precision_recall_curve, auc,
        classification_report, confusion_matrix
    )
    HAS_SKLEARN = True
    print("scikit-learn available — logistic regression enabled", flush=True)
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not installed. Using hardcoded weights only.", flush=True)
    print("  Install with: pip install scikit-learn", flush=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("c:/Users/josep/Documents/paews")
DATA_SST = BASE_DIR / "data" / "baseline_v2"
DATA_CHL = BASE_DIR / "data" / "baseline_v2_chl"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_EXTERNAL = BASE_DIR / "data" / "external"
OUTPUTS = BASE_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Coverage threshold for chlorophyll
MIN_CHL_OBS = 10

# Static ocean pixel count for percentage calculations (fixes >100% bug)
# We compute this once from the climatology and reuse it everywhere
STATIC_OCEAN_PIXELS = None


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ground_truth():
    """Load IMARPE season outcomes."""
    path = DATA_EXTERNAL / "imarpe_ground_truth.csv"
    if not path.exists():
        print(f"  ERROR: {path} not found", flush=True)
        return None
    df = pd.read_csv(path)
    
    # Binary target: 0 = NORMAL, 1 = at-risk (REDUCED/DISRUPTED/CANCELLED)
    df['target'] = df['outcome'].map({
        'NORMAL': 0,
        'REDUCED': 1,
        'DISRUPTED': 1,
        'CANCELLED': 1,
    })
    
    print(f"  Loaded {len(df)} season records", flush=True)
    print(f"  NORMAL: {(df['target']==0).sum()}, AT-RISK: {(df['target']==1).sum()}", flush=True)
    return df


def load_nino_indices():
    """Load Niño 1+2 monthly anomalies."""
    path = DATA_EXTERNAL / "nino_indices_monthly.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found — Niño index unavailable", flush=True)
        return None
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  Loaded Niño indices: {len(df)} months", flush=True)
    return df


def load_climatology(var='sst'):
    """Load SST or Chl climatology."""
    if var == 'sst':
        path = DATA_PROCESSED / "sst_climatology_v2.nc"
        if not path.exists():
            path = DATA_PROCESSED / "sst_climatology.nc"
    else:
        path = DATA_PROCESSED / "chl_climatology_v2.nc"
    
    if not path.exists():
        print(f"  WARNING: {var} climatology not found at {path}", flush=True)
        return None
    return xr.open_dataset(path)


def get_static_ocean_mask(sst_clim, chl_clim):
    """
    Build a static ocean pixel count — fixes the >100% bug.
    Uses pixels that have valid data in BOTH SST and Chl climatologies.
    """
    global STATIC_OCEAN_PIXELS
    
    # Use annual mean to find valid ocean pixels
    sst_valid = sst_clim["sst_mean"].isel(month=0).notnull()
    sst_count = int(sst_valid.sum())
    
    chl_valid = chl_clim["chl_log_mean"].isel(month=0).notnull()
    chl_count = int(chl_valid.sum())
    
    STATIC_OCEAN_PIXELS = {
        'sst': sst_count,
        'chl': chl_count,
    }
    print(f"  Static ocean mask: SST={sst_count} pixels, Chl={chl_count} pixels", flush=True)


def compute_sst_zscore_for_season(year, month, sst_clim):
    """Compute mean SST Z-score for a given year/month."""
    sst_path = DATA_SST / f"sst_{year}.nc"
    if not sst_path.exists():
        return np.nan, np.nan
    
    ds = xr.open_dataset(sst_path)
    time_idx = pd.DatetimeIndex(ds["time"].values)
    month_data = ds.sel(time=time_idx.month == month)
    
    if len(month_data.time) == 0:
        ds.close()
        return np.nan, np.nan
    
    sst_snap = month_data["sst"].isel(time=-1).squeeze()
    if 'zlev' in sst_snap.dims:
        sst_snap = sst_snap.isel(zlev=0)
    
    clim_mean = sst_clim["sst_mean"].sel(month=month)
    clim_std = sst_clim["sst_std"].sel(month=month)
    std_safe = clim_std.where(clim_std > 0.01)
    
    z = (sst_snap - clim_mean) / std_safe
    
    mean_z = float(z.mean(skipna=True))
    
    # MHW percentage using static denominator
    mhw_count = int((z > 1.28).sum(skipna=True))
    mhw_pct = mhw_count / STATIC_OCEAN_PIXELS['sst'] * 100 if STATIC_OCEAN_PIXELS else 0
    
    ds.close()
    return mean_z, mhw_pct


def compute_chl_zscore_for_season(year, month, chl_clim):
    """Compute mean Chl Z-score (log space) for a given year/month."""
    chl_path = DATA_CHL / f"chl_{year}.nc"
    if not chl_path.exists():
        return np.nan, np.nan
    
    ds = xr.open_dataset(chl_path)
    time_idx = pd.DatetimeIndex(ds["time"].values)
    month_data = ds.sel(time=time_idx.month == month)
    
    if len(month_data.time) == 0:
        ds.close()
        return np.nan, np.nan
    
    chl_snap = month_data["chlorophyll"].isel(time=-1).squeeze()
    if 'altitude' in chl_snap.dims:
        chl_snap = chl_snap.isel(altitude=0)
    
    clim_mean = chl_clim["chl_log_mean"].sel(month=month)
    clim_std = chl_clim["chl_log_std"].sel(month=month)
    obs_count = chl_clim["chl_obs_count"].sel(month=month)
    std_safe = clim_std.where(clim_std > 0.01)
    
    chl_log = np.log10(chl_snap.where(chl_snap > 0))
    z = (chl_log - clim_mean) / std_safe
    z = z.where(obs_count >= MIN_CHL_OBS)
    
    mean_z = float(z.mean(skipna=True))
    
    # Low Chl percentage using static denominator
    lchl_count = int((z < -1.28).sum(skipna=True))
    lchl_pct = lchl_count / STATIC_OCEAN_PIXELS['chl'] * 100 if STATIC_OCEAN_PIXELS else 0
    
    ds.close()
    return mean_z, lchl_pct


def get_nino12_for_season(year, month, nino_df, lag=1):
    """Get Niño 1+2 anomaly with specified lag (months before)."""
    if nino_df is None:
        return np.nan
    
    target_year = year
    target_month = month - lag
    if target_month <= 0:
        target_month += 12
        target_year -= 1
    
    row = nino_df[(nino_df['year'] == target_year) & (nino_df['month'] == target_month)]
    if len(row) == 0:
        return np.nan
    return float(row['nino12_anom'].iloc[0])


def compute_bio_threshold(year, month, sst_clim):
    """
    Compute % of pixels where absolute SST exceeds 23°C.
    
    Anchovy thermal tolerance ceiling is ~23-24°C. This catches cases
    where a moderate Z-score anomaly is harmless in winter (pushes to 19°C)
    but lethal in summer (pushes past 23°C).
    """
    sst_path = DATA_SST / f"sst_{year}.nc"
    if not sst_path.exists():
        return np.nan
    
    try:
        ds = xr.open_dataset(sst_path)
        time_idx = pd.DatetimeIndex(ds["time"].values)
        month_data = ds.sel(time=time_idx.month == month)
        
        if len(month_data.time) == 0:
            ds.close()
            return np.nan
        
        sst_snap = month_data["sst"].isel(time=-1).squeeze()
        if 'zlev' in sst_snap.dims:
            sst_snap = sst_snap.isel(zlev=0)
        
        valid = sst_snap.notnull()
        above_23 = (sst_snap > 23.0) & valid
        
        total = int(valid.sum())
        if total == 0:
            ds.close()
            return np.nan
        
        pct = float(above_23.sum()) / total * 100
        ds.close()
        return pct
    except Exception:
        return np.nan


# =============================================================================
# FEATURE MATRIX BUILDER
# =============================================================================

def build_feature_matrix(ground_truth, sst_clim, chl_clim, nino_df):
    """
    For each IMARPE season, compute the environmental features
    from the month BEFORE the season started (or the decision month).
    
    For season 1 (April-July start), we look at March conditions.
    For season 2 (November start), we look at October conditions.
    For cancelled seasons, we look at the month before the usual start.
    """
    print("\n  Building feature matrix...", flush=True)
    
    records = []
    for _, row in ground_truth.iterrows():
        year = row['year']
        season = row['season']
        
        # Decision month: 1 month before typical season start
        if season == 1:
            decision_month = 3   # March (before April start)
        else:
            decision_month = 10  # October (before November start)
        
        # SST Z-score
        sst_z, mhw_pct = compute_sst_zscore_for_season(year, decision_month, sst_clim)
        
        # Chl Z-score
        chl_z, lchl_pct = compute_chl_zscore_for_season(year, decision_month, chl_clim)
        
        # Niño 1+2 at t-1 (1 month lag)
        nino12_t1 = get_nino12_for_season(year, decision_month, nino_df, lag=1)
        
        # Niño 1+2 at t-2 (2 month lag — test both)
        nino12_t2 = get_nino12_for_season(year, decision_month, nino_df, lag=2)
        
        # Season flag: 1 = summer (Dec-Mar), 0 = winter (Apr-Nov)
        # Summer anomalies are deadlier — pushes SST past biological limits
        is_summer = 1 if decision_month in [1, 2, 3, 12] else 0
        
        # Biological threshold: % of pixels where absolute SST > 23°C
        # Anchovy thermal tolerance ceiling is ~23-24°C
        bio_thresh_pct = compute_bio_threshold(year, decision_month, sst_clim)
        
        # Hardcoded composite (baseline)
        if not np.isnan(sst_z) and not np.isnan(chl_z):
            # Chl Z is inverted: negative = bad, so we flip sign for composite
            composite_hard = 0.4 * sst_z + 0.4 * (-chl_z) + 0.2 * (nino12_t1 if not np.isnan(nino12_t1) else 0)
        else:
            composite_hard = np.nan
        
        records.append({
            'year': year,
            'season': season,
            'outcome': row['outcome'],
            'target': row['target'],
            'decision_month': decision_month,
            'is_summer': is_summer,
            'sst_z': sst_z,
            'chl_z': chl_z,
            'mhw_pct': mhw_pct,
            'lchl_pct': lchl_pct,
            'bio_thresh_pct': bio_thresh_pct,
            'nino12_t1': nino12_t1,
            'nino12_t2': nino12_t2,
            'composite_hard': composite_hard,
        })
        
        status = "OK" if not np.isnan(sst_z) else "NO DATA"
        if not np.isnan(sst_z) and not np.isnan(chl_z) and not np.isnan(nino12_t1):
            print(f"    {year} S{season} ({row['outcome'][:4]}): SST_Z={sst_z:+.2f}  Chl_Z={chl_z:+.2f}  "
                  f"Niño12={nino12_t1:+.2f}  Bio>23={bio_thresh_pct:.0f}%  Sum={is_summer}  "
                  f"Comp={composite_hard:+.2f}  [{status}]", flush=True)
        else:
            print(f"    {year} S{season} ({row['outcome'][:4]}): [{status}]", flush=True)
    
    df = pd.DataFrame(records)
    return df


# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================

def run_logistic_regression(features_df):
    """
    Fit logistic regression on all available data.
    With only ~30 samples, we use leave-one-out cross-validation
    to get honest performance estimates.
    """
    if not HAS_SKLEARN:
        print("\n  Skipping logistic regression (scikit-learn not installed)", flush=True)
        return None
    
    print("\n--- Logistic Regression ---", flush=True)
    
    # Drop rows with NaN features
    feature_cols = ['sst_z', 'chl_z', 'nino12_t1', 'is_summer', 'bio_thresh_pct']
    df = features_df.dropna(subset=['sst_z', 'chl_z', 'nino12_t1', 'target'])
    
    # Fill bio_thresh NaN with 0 (conservative: assume no thermal breach)
    df = df.copy()
    df['bio_thresh_pct'] = df['bio_thresh_pct'].fillna(0)
    df['is_summer'] = df['is_summer'].fillna(0)
    
    if len(df) < 10:
        print(f"  Only {len(df)} valid rows — insufficient for regression", flush=True)
        return None
    
    X = df[feature_cols].values
    y = df['target'].values
    
    print(f"  Samples: {len(df)} ({int(y.sum())} at-risk, {int((1-y).sum())} normal)", flush=True)
    print(f"  Features: SST_Z, -Chl_Z, Niño12, Season, Bio>23°C", flush=True)
    
    # Flip Chl sign — negative Chl Z = bad for anchovy = higher risk
    X_model = X.copy()
    X_model[:, 1] = -X_model[:, 1]  # invert Chl so positive = bad
    
    # Leave-one-out cross-validation
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    y_probs_loo = np.zeros(len(y))
    
    for train_idx, test_idx in loo.split(X_model):
        X_train, X_test = X_model[train_idx], X_model[test_idx]
        y_train = y[train_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, class_weight='balanced')
        model.fit(X_train_s, y_train)
        y_probs_loo[test_idx] = model.predict_proba(X_test_s)[:, 1]
    
    # Final model on all data (for coefficients)
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X_model)
    model_final = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, class_weight='balanced')
    model_final.fit(X_scaled, y)
    
    # Coefficients (these are the data-driven "weights")
    coefs = model_final.coef_[0]
    intercept = model_final.intercept_[0]
    
    print(f"\n  DATA-DRIVEN WEIGHTS (standardized):", flush=True)
    print(f"    SST_Z:       {coefs[0]:+.3f}", flush=True)
    print(f"    -Chl_Z:      {coefs[1]:+.3f}", flush=True)
    print(f"    Niño 1+2:    {coefs[2]:+.3f}", flush=True)
    print(f"    Summer flag:  {coefs[3]:+.3f}", flush=True)
    print(f"    Bio >23°C:   {coefs[4]:+.3f}", flush=True)
    print(f"    Intercept:   {intercept:+.3f}", flush=True)
    
    # Normalized relative weights
    abs_coefs = np.abs(coefs)
    rel_weights = abs_coefs / abs_coefs.sum()
    print(f"\n  RELATIVE IMPORTANCE:", flush=True)
    print(f"    SST:     {rel_weights[0]:.1%}", flush=True)
    print(f"    Chl:     {rel_weights[1]:.1%}", flush=True)
    print(f"    Niño:    {rel_weights[2]:.1%}", flush=True)
    print(f"    Season:  {rel_weights[3]:.1%}", flush=True)
    print(f"    Bio>23:  {rel_weights[4]:.1%}", flush=True)
    
    # Add predictions to dataframe
    df = df.copy()
    df['prob_atrisk'] = y_probs_loo
    df['pred_label'] = (y_probs_loo >= 0.5).astype(int)
    
    return {
        'model': model_final,
        'scaler': scaler_final,
        'coefs': coefs,
        'intercept': intercept,
        'rel_weights': rel_weights,
        'feature_cols': feature_cols,
        'df': df,
        'y_true': y,
        'y_probs': y_probs_loo,
    }


# =============================================================================
# VALIDATION METRICS
# =============================================================================

def compute_validation_metrics(results, features_df):
    """Compute ROC-AUC, PR-AUC, confusion matrix, and hit/miss analysis."""
    print("\n--- Validation Metrics ---", flush=True)
    
    # ---- Hardcoded composite threshold analysis ----
    df_valid = features_df.dropna(subset=['composite_hard', 'target'])
    if len(df_valid) > 0:
        y_true_hard = df_valid['target'].values
        composite = df_valid['composite_hard'].values
        
        # Find best threshold for hardcoded composite
        best_f1 = 0
        best_thresh = 0
        for thresh in np.arange(0.0, 2.0, 0.05):
            preds = (composite >= thresh).astype(int)
            tp = ((preds == 1) & (y_true_hard == 1)).sum()
            fp = ((preds == 1) & (y_true_hard == 0)).sum()
            fn = ((preds == 0) & (y_true_hard == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        print(f"\n  HARDCODED COMPOSITE (0.4*SST + 0.4*(-Chl) + 0.2*Niño):", flush=True)
        print(f"  Best threshold: {best_thresh:.2f} (F1={best_f1:.2f})", flush=True)
        
        preds_hard = (composite >= best_thresh).astype(int)
        tp = ((preds_hard == 1) & (y_true_hard == 1)).sum()
        fp = ((preds_hard == 1) & (y_true_hard == 0)).sum()
        fn = ((preds_hard == 0) & (y_true_hard == 1)).sum()
        tn = ((preds_hard == 0) & (y_true_hard == 0)).sum()
        print(f"  Confusion Matrix: TP={tp} FP={fp} FN={fn} TN={tn}", flush=True)
    
    # ---- Logistic regression metrics (LOO) ----
    if results is None:
        return
    
    y_true = results['y_true']
    y_probs = results['y_probs']
    df = results['df']
    
    # ROC-AUC
    try:
        roc = roc_auc_score(y_true, y_probs)
        print(f"\n  LOGISTIC REGRESSION (Leave-One-Out CV):", flush=True)
        print(f"  ROC-AUC: {roc:.3f}", flush=True)
    except:
        roc = None
        print(f"  ROC-AUC: could not compute", flush=True)
    
    # PR-AUC (primary metric for rare events)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    print(f"  PR-AUC:  {pr_auc:.3f}  (PRIMARY METRIC)", flush=True)
    
    # Best threshold from PR curve
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else 0.5
    print(f"  Best threshold: {best_threshold:.2f} (F1={f1_scores[best_idx]:.2f})", flush=True)
    
    # Classification report at best threshold
    y_pred = (y_probs >= best_threshold).astype(int)
    print(f"\n  Classification Report (threshold={best_threshold:.2f}):", flush=True)
    print(classification_report(y_true, y_pred, target_names=['NORMAL', 'AT-RISK']), flush=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:", flush=True)
    print(f"                Predicted NORMAL  Predicted AT-RISK", flush=True)
    print(f"  Actual NORMAL     {cm[0,0]:3d}              {cm[0,1]:3d}", flush=True)
    print(f"  Actual AT-RISK    {cm[1,0]:3d}              {cm[1,1]:3d}", flush=True)
    
    # Hit/Miss detail
    print(f"\n  SEASON-BY-SEASON DETAIL:", flush=True)
    print(f"  {'Year':>4} {'S':>1} {'Outcome':>10} {'Prob':>5} {'Pred':>8} {'Result':>8}", flush=True)
    print(f"  {'-'*45}", flush=True)
    for _, row in df.iterrows():
        pred_label = "AT-RISK" if row['prob_atrisk'] >= best_threshold else "NORMAL"
        actual = "AT-RISK" if row['target'] == 1 else "NORMAL"
        hit = "HIT" if pred_label == actual else "MISS"
        print(f"  {int(row['year']):>4} {int(row['season']):>1} {row['outcome']:>10} "
              f"{row['prob_atrisk']:>5.2f} {pred_label:>8} {hit:>8}", flush=True)
    
    return {
        'roc_auc': roc,
        'pr_auc': pr_auc,
        'best_threshold': best_threshold,
        'confusion_matrix': cm,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_validation_dashboard(features_df, results, metrics):
    """Generate composite score validation dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    fig.suptitle(
        f"PAEWS Composite Score Validation\n"
        f"ROC-AUC: {metrics.get('roc_auc', 0):.3f} | "
        f"PR-AUC: {metrics.get('pr_auc', 0):.3f}",
        fontsize=14, fontweight='bold'
    )
    
    # Panel 1: Composite score time series
    ax1 = axes[0, 0]
    df = features_df.dropna(subset=['composite_hard'])
    normal = df[df['target'] == 0]
    atrisk = df[df['target'] == 1]
    x_labels = [f"{int(r['year'])}\nS{int(r['season'])}" for _, r in df.iterrows()]
    
    ax1.bar(range(len(normal)), normal['composite_hard'], color='forestgreen', alpha=0.7, label='Normal')
    atrisk_indices = [df.index.get_loc(i) for i in atrisk.index]
    ax1.bar(atrisk_indices, atrisk['composite_hard'], color='red', alpha=0.7, label='At-Risk')
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(x_labels, rotation=90, fontsize=6)
    ax1.set_ylabel("Hardcoded Composite Score")
    ax1.set_title("Composite Score by Season")
    ax1.legend(fontsize=8)
    
    # Panel 2: Feature scatter (SST vs Chl)
    ax2 = axes[0, 1]
    df_plot = features_df.dropna(subset=['sst_z', 'chl_z'])
    normal_p = df_plot[df_plot['target'] == 0]
    atrisk_p = df_plot[df_plot['target'] == 1]
    ax2.scatter(normal_p['sst_z'], normal_p['chl_z'], c='forestgreen', s=60, label='Normal', edgecolors='black', alpha=0.8)
    ax2.scatter(atrisk_p['sst_z'], atrisk_p['chl_z'], c='red', s=80, marker='X', label='At-Risk', edgecolors='black')
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.axvline(0, color='gray', linewidth=0.5)
    ax2.set_xlabel("SST Z-score (positive = warm)")
    ax2.set_ylabel("Chl Z-score (negative = low productivity)")
    ax2.set_title("SST vs Chlorophyll (decision month)")
    ax2.legend(fontsize=8)
    
    # Annotate key events
    for _, r in atrisk_p.iterrows():
        if not np.isnan(r['sst_z']) and not np.isnan(r['chl_z']):
            ax2.annotate(f"{int(r['year'])}S{int(r['season'])}", 
                        (r['sst_z'], r['chl_z']), fontsize=7, alpha=0.7)
    
    # Panel 3: PR curve (if logistic regression available)
    ax3 = axes[1, 0]
    if results is not None:
        y_true = results['y_true']
        y_probs = results['y_probs']
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        ax3.plot(recall, precision, color='darkred', linewidth=2)
        ax3.fill_between(recall, precision, alpha=0.2, color='red')
        ax3.set_xlabel("Recall (sensitivity)")
        ax3.set_ylabel("Precision")
        ax3.set_title(f"Precision-Recall Curve (AUC={metrics.get('pr_auc', 0):.3f})")
        ax3.set_xlim(0, 1.05)
        ax3.set_ylim(0, 1.05)
        
        # Baseline (random)
        baseline = y_true.sum() / len(y_true)
        ax3.axhline(baseline, color='gray', linestyle='--', label=f'Random ({baseline:.2f})')
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "Install scikit-learn\nfor logistic regression",
                 transform=ax3.transAxes, ha='center', va='center', fontsize=12)
        ax3.set_title("PR Curve (unavailable)")
    
    # Panel 4: Feature importance
    ax4 = axes[1, 1]
    if results is not None:
        weights = results['rel_weights']
        labels = ['SST\nwarm', 'Chl\nfood', 'Niño1+2\nbasin', 'Season\nflag', 'Bio>23°C\nthermal']
        colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd']
        bars = ax4.bar(labels, weights, color=colors, edgecolor='black', alpha=0.8)
        ax4.set_ylabel("Relative Importance")
        ax4.set_title("Data-Driven Feature Weights (Logistic Regression)")
        ax4.set_ylim(0, 1)
        for bar, w in zip(bars, weights):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{w:.1%}', ha='center', fontsize=10, fontweight='bold')
    else:
        labels = ['SST', 'Chl', 'Niño 1+2']
        weights = [0.4, 0.4, 0.2]
        ax4.bar(labels, weights, color=['#d62728', '#2ca02c', '#1f77b4'], edgecolor='black')
        ax4.set_title("Hardcoded Weights (physics-based)")
        ax4.set_ylabel("Weight")
    
    plt.tight_layout()
    outpath = OUTPUTS / "composite_score_validation.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Dashboard saved: {outpath}", flush=True)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAEWS Composite Score Builder")
    parser.add_argument("--predict", action="store_true", help="Score current conditions")
    args = parser.parse_args()
    
    print("=" * 60, flush=True)
    print("PAEWS COMPOSITE SCORE & VALIDATION", flush=True)
    print("=" * 60, flush=True)
    
    # ---- Load everything ----
    print("\nStep 1: Loading data...", flush=True)
    ground_truth = load_ground_truth()
    nino_df = load_nino_indices()
    sst_clim = load_climatology('sst')
    chl_clim = load_climatology('chl')
    
    if ground_truth is None or sst_clim is None or chl_clim is None:
        print("  FATAL: Missing required data", flush=True)
        sys.exit(1)
    
    # ---- Static ocean mask (fixes >100% bug) ----
    print("\nStep 2: Building static ocean mask...", flush=True)
    get_static_ocean_mask(sst_clim, chl_clim)
    
    # ---- Build feature matrix ----
    print("\nStep 3: Computing features for each season...", flush=True)
    features_df = build_feature_matrix(ground_truth, sst_clim, chl_clim, nino_df)
    
    # Save feature matrix
    feat_path = DATA_EXTERNAL / "paews_feature_matrix.csv"
    features_df.to_csv(feat_path, index=False)
    print(f"\n  Feature matrix saved: {feat_path}", flush=True)
    
    # ---- Logistic regression ----
    print("\nStep 4: Running logistic regression...", flush=True)
    lr_results = run_logistic_regression(features_df)
    
    # ---- Validation ----
    print("\nStep 5: Computing validation metrics...", flush=True)
    metrics = compute_validation_metrics(lr_results, features_df)
    
    # ---- Dashboard ----
    print("\nStep 6: Generating dashboard...", flush=True)
    plot_validation_dashboard(features_df, lr_results, metrics)
    
    # ---- Summary ----
    print("\n" + "=" * 60, flush=True)
    print("COMPOSITE SCORE COMPLETE", flush=True)
    print(f"  Seasons analyzed: {len(features_df)}", flush=True)
    valid = features_df.dropna(subset=['composite_hard'])
    print(f"  Valid features: {len(valid)}", flush=True)
    if metrics:
        print(f"  ROC-AUC: {metrics.get('roc_auc', 'N/A')}", flush=True)
        print(f"  PR-AUC:  {metrics.get('pr_auc', 'N/A')}", flush=True)
    if lr_results:
        w = lr_results['rel_weights']
        print(f"  Data-driven weights: SST={w[0]:.1%} Chl={w[1]:.1%} Niño={w[2]:.1%} Season={w[3]:.1%} Bio={w[4]:.1%}", flush=True)
    print(f"  Dashboard: outputs/composite_score_validation.png", flush=True)
    print(f"  Feature matrix: data/external/paews_feature_matrix.csv", flush=True)
    print("=" * 60, flush=True)
