"""
test_prev_catch_feature.py — Test previous season catch % as 4th feature

For S1 of year Y: uses S2 of year Y-1 catch_pct
For S2 of year Y: uses S1 of year Y catch_pct
No data leakage: previous season is always known at prediction time.

Run from project root:
    python scripts/test_prev_catch_feature.py
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Load data ──
gt = pd.read_csv("data/external/imarpe_ground_truth.csv")
fm = pd.read_csv("data/external/paews_feature_matrix.csv")

# ── Build previous season catch_pct lookup ──
# Ground truth has: year, season, catch_pct, outcome
gt_lookup = {}
for _, row in gt.iterrows():
    gt_lookup[(row["year"], row["season"])] = row["catch_pct"]

prev_catch = []
for _, row in fm.iterrows():
    yr = row["year"]
    sn = row["season"]
    
    if sn == 1:
        # S1 prediction uses previous S2 (year-1)
        prev_key = (yr - 1, 2)
    else:
        # S2 prediction uses previous S1 (same year)
        prev_key = (yr, 1)
    
    pct = gt_lookup.get(prev_key, np.nan)
    prev_catch.append(pct)

fm["prev_catch_pct"] = prev_catch

# ── Handle missing (2010 S1 has no 2009 S2 in dataset) ──
missing = fm["prev_catch_pct"].isna().sum()
print(f"Previous catch % computed: {len(fm) - missing}/32 available, {missing} missing")
if missing > 0:
    print(f"  Missing rows:")
    for _, row in fm[fm["prev_catch_pct"].isna()].iterrows():
        print(f"    {int(row['year'])} S{int(row['season'])}")

# Drop rows with missing prev_catch_pct for fair comparison
fm_full = fm.dropna(subset=["prev_catch_pct"]).copy()
n = len(fm_full)
print(f"\nUsing {n} samples for LOO comparison\n")

# ── Define feature sets ──
feat_3 = ["sst_z", "chl_z", "nino12_t1"]
feat_4 = ["sst_z", "chl_z", "nino12_t1", "prev_catch_pct"]

X_3 = fm_full[feat_3].values
X_4 = fm_full[feat_4].values
y = fm_full["target"].values

# ── LOO for both models ──
def loo_evaluate(X, y, label):
    n = len(y)
    probs = np.zeros(n)
    
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i:i+1]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs"
        )
        model.fit(X_train_s, y_train)
        probs[i] = model.predict_proba(X_test_s)[0, 1]
    
    roc = roc_auc_score(y, probs)
    pr = average_precision_score(y, probs)
    
    # Tier calibration
    tiers = {"LOW": 0, "MODERATE": 0, "ELEVATED": 0, "SEVERE": 0}
    tier_correct = {"LOW": 0, "MODERATE": 0, "ELEVATED": 0, "SEVERE": 0}
    tier_total = {"LOW": 0, "MODERATE": 0, "ELEVATED": 0, "SEVERE": 0}
    
    for p, actual in zip(probs, y):
        if p >= 0.70:
            tier = "SEVERE"
        elif p >= 0.50:
            tier = "ELEVATED"
        elif p >= 0.20:
            tier = "MODERATE"
        else:
            tier = "LOW"
        tier_total[tier] += 1
        if actual == 1:
            tier_correct[tier] += 1
    
    print(f"  {label}")
    print(f"  {'='*50}")
    print(f"  ROC-AUC:  {roc:.3f}")
    print(f"  PR-AUC:   {pr:.3f}")
    print(f"  Samples:  {n}")
    print(f"  Positives: {int(y.sum())}")
    print(f"")
    print(f"  Tier Calibration:")
    for tier in ["SEVERE", "ELEVATED", "MODERATE", "LOW"]:
        t = tier_total[tier]
        c = tier_correct[tier]
        rate = f"{c}/{t} ({100*c/t:.0f}%)" if t > 0 else "0/0"
        print(f"    {tier:>10}: {rate}")
    print()
    
    # Full model coefficients (trained on all data)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
    model.fit(X_s, y)
    
    feat_names = feat_3 if X.shape[1] == 3 else feat_4
    print(f"  Coefficients (full model):")
    for name, coef in zip(feat_names, model.coef_[0]):
        print(f"    {name:>16}: {coef:+.3f}")
    print(f"    {'intercept':>16}: {model.intercept_[0]:+.3f}")
    print()
    
    return roc, pr, probs

print("=" * 60)
print("  FEATURE COMPARISON: 3-feature vs 4-feature (+ prev_catch_pct)")
print("=" * 60)
print()

roc3, pr3, probs3 = loo_evaluate(X_3, y, "Model A: 3 features (baseline)")
roc4, pr4, probs4 = loo_evaluate(X_4, y, "Model B: 4 features (+ prev_catch_pct)")

# ── Comparison ──
print("=" * 60)
print("  COMPARISON SUMMARY")
print("=" * 60)
droc = roc4 - roc3
dpr = pr4 - pr3
print(f"  ROC-AUC:  {roc3:.3f} → {roc4:.3f}  ({droc:+.3f})")
print(f"  PR-AUC:   {pr3:.3f} → {pr4:.3f}  ({dpr:+.3f})")
print()
if droc > 0.02:
    print(f"  ✓ prev_catch_pct IMPROVES the model (+{droc:.3f} ROC-AUC)")
    print(f"    Consider adding to production model.")
elif droc > -0.02:
    print(f"  ~ prev_catch_pct has NEGLIGIBLE effect ({droc:+.3f} ROC-AUC)")
    print(f"    Not worth the added complexity.")
else:
    print(f"  ✗ prev_catch_pct HURTS the model ({droc:+.3f} ROC-AUC)")
    print(f"    Do not add. Extra feature adds noise.")

# ── Show per-season differences ──
print()
print("  Per-season probability changes (4feat - 3feat):")
print(f"  {'Season':<10} {'Actual':<8} {'3-feat':>8} {'4-feat':>8} {'Δ':>8}")
print(f"  {'-'*44}")
for i, (_, row) in enumerate(fm_full.iterrows()):
    yr = int(row["year"])
    sn = int(row["season"])
    actual = int(row["target"])
    d = probs4[i] - probs3[i]
    flag = " ←" if abs(d) > 0.05 else ""
    print(f"  {yr} S{sn:<6} {actual:<8} {probs3[i]:>8.3f} {probs4[i]:>8.3f} {d:>+8.3f}{flag}")

print()
print(f"  Prev catch % values used:")
print(f"  {'Season':<10} {'prev_catch_pct':>15}")
for _, row in fm_full.iterrows():
    print(f"  {int(row['year'])} S{int(row['season']):<6} {row['prev_catch_pct']:>14.0f}%")
