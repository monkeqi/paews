"""
test_prev_feature_variants.py — Deep dive on previous-season feature formulations

Tests multiple transformations to find one that:
  1. Improves ROC-AUC over 0.629 baseline
  2. Keeps SEVERE tier at 100% (non-negotiable)
  3. Reduces false alarms on NORMAL seasons

Run from project root:
    python scripts/test_prev_feature_variants.py
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Load data ──
gt = pd.read_csv("data/external/imarpe_ground_truth.csv")
fm = pd.read_csv("data/external/paews_feature_matrix.csv")

# ── Build lookups from ground truth ──
gt_catch_pct = {}
gt_outcome = {}
gt_target = {}
gt_quota = {}
for _, row in gt.iterrows():
    key = (row["year"], row["season"])
    gt_catch_pct[key] = row["catch_pct"]
    gt_outcome[key] = row["outcome"]
    gt_quota[key] = row.get("quota_mt", 0)
    gt_target[key] = 0 if row["outcome"] == "NORMAL" else 1

# ── Compute all candidate features ──
variants = {}

# Variant A: raw catch_pct (baseline from last test)
prev_raw = []
for _, row in fm.iterrows():
    yr, sn = row["year"], row["season"]
    prev_key = (yr - 1, 2) if sn == 1 else (yr, 1)
    prev_raw.append(gt_catch_pct.get(prev_key, np.nan))
variants["A_raw_catch_pct"] = prev_raw

# Variant B: binary "was previous season disrupted" (target=1)
prev_disrupted = []
for _, row in fm.iterrows():
    yr, sn = row["year"], row["season"]
    prev_key = (yr - 1, 2) if sn == 1 else (yr, 1)
    t = gt_target.get(prev_key, np.nan)
    prev_disrupted.append(t)
variants["B_prev_disrupted"] = prev_disrupted

# Variant C: binary "catch_pct < 80"
prev_low_catch = []
for _, row in fm.iterrows():
    yr, sn = row["year"], row["season"]
    prev_key = (yr - 1, 2) if sn == 1 else (yr, 1)
    pct = gt_catch_pct.get(prev_key, np.nan)
    if np.isnan(pct):
        prev_low_catch.append(np.nan)
    else:
        prev_low_catch.append(1.0 if pct < 80 else 0.0)
variants["C_catch_below_80"] = prev_low_catch

# Variant D: binary "catch_pct < 85"
prev_low_catch_85 = []
for _, row in fm.iterrows():
    yr, sn = row["year"], row["season"]
    prev_key = (yr - 1, 2) if sn == 1 else (yr, 1)
    pct = gt_catch_pct.get(prev_key, np.nan)
    if np.isnan(pct):
        prev_low_catch_85.append(np.nan)
    else:
        prev_low_catch_85.append(1.0 if pct < 85 else 0.0)
variants["D_catch_below_85"] = prev_low_catch_85

# Variant E: capped — min(catch_pct, 85) then invert so low = high risk
prev_capped = []
for _, row in fm.iterrows():
    yr, sn = row["year"], row["season"]
    prev_key = (yr - 1, 2) if sn == 1 else (yr, 1)
    pct = gt_catch_pct.get(prev_key, np.nan)
    if np.isnan(pct):
        prev_capped.append(np.nan)
    else:
        # Invert: 100 - min(pct, 85) so higher = more risk
        prev_capped.append(100.0 - min(pct, 85.0))
variants["E_capped_inverted"] = prev_capped

# Variant F: "consecutive stress" — prev disrupted AND current Niño positive
# This is an interaction: only fire when both biological AND oceanic stress present
prev_stress_x_nino = []
for _, row in fm.iterrows():
    yr, sn = row["year"], row["season"]
    prev_key = (yr - 1, 2) if sn == 1 else (yr, 1)
    t = gt_target.get(prev_key, np.nan)
    nino = row["nino12_t1"]
    if np.isnan(t):
        prev_stress_x_nino.append(np.nan)
    else:
        # 1 only if previous season was disrupted AND current Niño is warm (>0)
        prev_stress_x_nino.append(1.0 if (t == 1 and nino > 0) else 0.0)
variants["F_disrupted_x_warm_nino"] = prev_stress_x_nino

# Variant G: "quota signal" — was previous quota below 2M MT (low confidence by IMARPE)
prev_low_quota = []
for _, row in fm.iterrows():
    yr, sn = row["year"], row["season"]
    prev_key = (yr - 1, 2) if sn == 1 else (yr, 1)
    q = gt_quota.get(prev_key, np.nan)
    if q is np.nan or q == 0:
        # 2023 S1 had quota 0 (cancelled) - that's extreme low
        if gt_outcome.get(prev_key) == "CANCELLED":
            prev_low_quota.append(1.0)
        elif prev_key not in gt_quota:
            prev_low_quota.append(np.nan)
        else:
            prev_low_quota.append(1.0)  # zero quota = extreme
    else:
        prev_low_quota.append(1.0 if q < 2000000 else 0.0)
variants["G_prev_quota_below_2M"] = prev_low_quota

# Variant H: two-season lookback — either of the two previous seasons disrupted
prev_2season = []
for _, row in fm.iterrows():
    yr, sn = row["year"], row["season"]
    if sn == 1:
        keys = [(yr - 1, 2), (yr - 1, 1)]
    else:
        keys = [(yr, 1), (yr - 1, 2)]
    
    vals = [gt_target.get(k, np.nan) for k in keys]
    if any(np.isnan(v) for v in vals):
        prev_2season.append(np.nan)
    else:
        prev_2season.append(1.0 if any(v == 1 for v in vals) else 0.0)
variants["H_any_of_prev_2_disrupted"] = prev_2season

# ══════════════════════════════════════════════════════════════
# LOO EVALUATION
# ══════════════════════════════════════════════════════════════

feat_base = ["sst_z", "chl_z", "nino12_t1"]

def loo_evaluate(X, y, return_details=False):
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
    
    # Tier stats
    severe_total = 0
    severe_correct = 0
    false_alarms_elevated_plus = 0  # NORMAL seasons scored ELEVATED or SEVERE
    
    for p, actual in zip(probs, y):
        if p >= 0.70:
            severe_total += 1
            if actual == 1:
                severe_correct += 1
            else:
                false_alarms_elevated_plus += 1
        elif p >= 0.50:
            if actual == 0:
                false_alarms_elevated_plus += 1
    
    severe_str = f"{severe_correct}/{severe_total}" if severe_total > 0 else "0/0"
    severe_pct = severe_correct / severe_total * 100 if severe_total > 0 else 0
    
    if return_details:
        # Get coefficients
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
        model.fit(X_s, y)
        return roc, pr, probs, severe_str, severe_pct, false_alarms_elevated_plus, model.coef_[0], model.intercept_[0]
    
    return roc, pr, severe_str, severe_pct, false_alarms_elevated_plus

# ── Run all variants ──
print("=" * 80)
print("  DEEP DIVE: Previous-Season Feature Variants")
print("  Non-negotiable constraint: SEVERE tier must stay at 100%")
print("=" * 80)
print()

# First run baseline on same 31 samples (drop 2010 S1)
results = []

for name, values in variants.items():
    fm_test = fm.copy()
    fm_test["new_feat"] = values
    fm_test = fm_test.dropna(subset=["new_feat"])
    
    n = len(fm_test)
    X_base = fm_test[feat_base].values
    X_new = fm_test[feat_base + ["new_feat"]].values if "new_feat" in fm_test.columns else None
    y = fm_test["target"].values
    
    # Baseline (3-feature) on this subset
    roc_b, pr_b, sev_b, sev_pct_b, fa_b = loo_evaluate(X_base, y)
    
    # New (4-feature)
    roc_n, pr_n, probs_n, sev_n, sev_pct_n, fa_n, coefs, intercept = loo_evaluate(X_new, y, return_details=True)
    
    results.append({
        "name": name,
        "n": n,
        "roc_base": roc_b,
        "roc_new": roc_n,
        "delta_roc": roc_n - roc_b,
        "pr_base": pr_b,
        "pr_new": pr_n,
        "severe": sev_n,
        "severe_pct": sev_pct_n,
        "false_alarms": fa_n,
        "fa_base": fa_b,
        "coef_new": coefs[-1],
        "intercept": intercept,
        "probs": probs_n,
        "y": y,
        "fm": fm_test,
    })

# ── Print summary table ──
print(f"  {'Variant':<30} {'N':>3} {'ΔROC':>7} {'ROC':>6} {'SEVERE':>8} {'FA↑':>4} {'Coef':>7} {'PASS':>5}")
print(f"  {'-'*78}")

for r in sorted(results, key=lambda x: -x["delta_roc"]):
    severe_ok = "✓" if r["severe_pct"] == 100 else "✗"
    fa_delta = r["false_alarms"] - r["fa_base"]
    fa_str = f"{fa_delta:+d}" if fa_delta != 0 else "0"
    
    # PASS = improves AUC AND keeps SEVERE at 100%
    passes = r["delta_roc"] > 0.005 and r["severe_pct"] == 100
    pass_str = "✓✓" if passes else ("~" if r["severe_pct"] == 100 else "✗")
    
    print(f"  {r['name']:<30} {r['n']:>3} {r['delta_roc']:>+7.3f} {r['roc_new']:>6.3f} "
          f"{r['severe']:>8} {fa_str:>4} {r['coef_new']:>+7.3f} {pass_str:>5}")

# ── Detail on passing variants ──
print()
print("=" * 80)
print("  DETAILED ANALYSIS OF PASSING VARIANTS")
print("=" * 80)

for r in sorted(results, key=lambda x: -x["delta_roc"]):
    if r["severe_pct"] < 100:
        continue
    if r["delta_roc"] < 0.005:
        continue
    
    print(f"\n  ── {r['name']} ──")
    print(f"  ROC-AUC: {r['roc_base']:.3f} → {r['roc_new']:.3f} ({r['delta_roc']:+.3f})")
    print(f"  PR-AUC:  {r['pr_base']:.3f} → {r['pr_new']:.3f}")
    print(f"  SEVERE:  {r['severe']} (100%)")
    print(f"  New feature coefficient: {r['coef_new']:+.3f}")
    print()
    
    # Show problem cases
    fm_t = r["fm"]
    probs = r["probs"]
    y = r["y"]
    
    # Recompute baseline probs for comparison
    X_base = fm_t[feat_base].values
    _, _, base_probs, _, _, _, _, _ = loo_evaluate(X_base, y, return_details=True)
    
    print(f"  Per-season changes (|Δ| > 0.03):")
    print(f"  {'Season':<10} {'Actual':<8} {'3-feat':>8} {'4-feat':>8} {'Δ':>8} {'NewFeat':>8}")
    print(f"  {'-'*54}")
    
    for i, (_, row) in enumerate(fm_t.iterrows()):
        d = probs[i] - base_probs[i]
        if abs(d) > 0.03:
            yr = int(row['year'])
            sn = int(row['season'])
            actual = int(row['target'])
            nf = row['new_feat']
            good = "✓" if (d > 0 and actual == 1) or (d < 0 and actual == 0) else "✗"
            print(f"  {yr} S{sn:<6} {actual:<8} {base_probs[i]:>8.3f} {probs[i]:>8.3f} {d:>+8.3f} {nf:>8.1f} {good}")
    
    # Count improvements vs degradations
    improved = 0
    degraded = 0
    for i in range(len(probs)):
        d = probs[i] - base_probs[i]
        actual = y[i]
        if abs(d) > 0.01:
            if (d > 0 and actual == 1) or (d < 0 and actual == 0):
                improved += 1
            else:
                degraded += 1
    
    print(f"\n  Directionally correct changes: {improved}")
    print(f"  Directionally wrong changes:   {degraded}")

# ── If nothing passes, say so ──
passing = [r for r in results if r["severe_pct"] == 100 and r["delta_roc"] > 0.005]
if not passing:
    print("\n  ✗ NO VARIANT PASSES BOTH CRITERIA")
    print("    (Improves ROC-AUC AND keeps SEVERE at 100%)")
    print()
    print("  Closest variants:")
    for r in sorted(results, key=lambda x: -x["delta_roc"])[:3]:
        print(f"    {r['name']}: ΔROC={r['delta_roc']:+.3f}, SEVERE={r['severe']}")
    print()
    print("  Recommendation: prev-season features don't reliably help.")
    print("  Anchoveta recover too fast for sequential disruption logic.")
    print("  Try subsurface temperature (GODAS) instead — it addresses")
    print("  the physical mechanism, not statistical correlation.")
else:
    print()
    print("=" * 80)
    best = max(passing, key=lambda x: x["delta_roc"])
    print(f"  BEST PASSING VARIANT: {best['name']}")
    print(f"  ROC-AUC: {best['roc_base']:.3f} → {best['roc_new']:.3f} ({best['delta_roc']:+.3f})")
    print(f"  SEVERE: {best['severe']} (100%)")
    print(f"  Coefficient: {best['coef_new']:+.3f}")
    print(f"")
    print(f"  This variant can be considered for production.")
    print(f"  Run scenario_analysis.py with 4 features to verify")
    print(f"  the 2026 S1 prediction before committing.")
    print("=" * 80)
