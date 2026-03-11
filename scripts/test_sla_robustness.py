"""
SLA Robustness Checks
=====================
Is the sla_z improvement real or a 30-sample mirage?

Tests:
  1. Bootstrap stability — how often does sla_z actually help?
  2. Permutation test — could random noise do this well?
  3. VIF — how bad is the collinearity really?
  4. Residualized SLA — remove the SST overlap, test what's left
  5. Paired comparison — every LOO fold, does 4-feat beat 3-feat?
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

fm = pd.read_csv("/mnt/user-data/uploads/paews_feature_matrix.csv")
gt = pd.read_csv("/mnt/user-data/uploads/imarpe_ground_truth.csv")

feat_base = ["sst_z", "chl_z", "nino12_t1"]
feat_sla = ["sst_z", "chl_z", "nino12_t1", "sla_z"]

# Drop rows missing SLA
fm_test = fm.dropna(subset=feat_sla + ['target']).copy()
X_base = fm_test[feat_base].values
X_sla = fm_test[feat_sla].values
y = fm_test['target'].values
n = len(y)

print(f"Samples: {n}, Positives: {int(y.sum())}, Negatives: {int((1-y).sum())}")
print()

# ═══════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════

def loo_auc(X, y):
    """LOO cross-validation, return AUC and per-sample probs."""
    probs = np.zeros(len(y))
    for i in range(len(y)):
        Xtr = np.delete(X, i, axis=0)
        ytr = np.delete(y, i, axis=0)
        Xte = X[i:i+1]
        sc = StandardScaler()
        m = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
        m.fit(sc.fit_transform(Xtr), ytr)
        probs[i] = m.predict_proba(sc.transform(Xte))[0, 1]
    return roc_auc_score(y, probs), probs

# ═══════════════════════════════════════════════
# TEST 1: BOOTSTRAP STABILITY
# How often does adding sla_z improve AUC?
# ═══════════════════════════════════════════════
print("=" * 65)
print("  TEST 1: BOOTSTRAP STABILITY (500 resamples)")
print("=" * 65)

rng = np.random.RandomState(42)
n_boot = 500
improvements = 0
deltas = []

for b in range(n_boot):
    idx = rng.choice(n, size=n, replace=True)
    Xb, yb = X_base[idx], y[idx]
    Xb_sla = X_sla[idx]

    # Need both classes
    if yb.sum() == 0 or yb.sum() == len(yb):
        continue

    # Train on full bootstrap sample, evaluate on OOB
    oob = np.array([i for i in range(n) if i not in idx])
    if len(oob) < 5 or y[oob].sum() == 0 or y[oob].sum() == len(oob):
        continue

    sc3 = StandardScaler()
    m3 = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
    m3.fit(sc3.fit_transform(Xb), yb)
    p3 = m3.predict_proba(sc3.transform(X_base[oob]))[:, 1]

    sc4 = StandardScaler()
    m4 = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
    m4.fit(sc4.fit_transform(Xb_sla), yb)
    p4 = m4.predict_proba(sc4.transform(X_sla[oob]))[:, 1]

    try:
        auc3 = roc_auc_score(y[oob], p3)
        auc4 = roc_auc_score(y[oob], p4)
        d = auc4 - auc3
        deltas.append(d)
        if d > 0:
            improvements += 1
    except ValueError:
        continue

valid_boots = len(deltas)
print(f"  Valid bootstrap samples: {valid_boots}")
print(f"  SLA improved AUC: {improvements}/{valid_boots} ({improvements/valid_boots*100:.1f}%)")
print(f"  Mean ΔAUC: {np.mean(deltas):+.4f}")
print(f"  Median ΔAUC: {np.median(deltas):+.4f}")
print(f"  Std ΔAUC: {np.std(deltas):.4f}")
print(f"  95% CI: [{np.percentile(deltas, 2.5):+.4f}, {np.percentile(deltas, 97.5):+.4f}]")

if improvements / valid_boots > 0.65:
    print(f"  ✓ SLA helps more often than not ({improvements/valid_boots*100:.0f}% > 65%)")
elif improvements / valid_boots > 0.50:
    print(f"  ~ SLA helps slightly more than not ({improvements/valid_boots*100:.0f}%)")
else:
    print(f"  ✗ SLA does NOT reliably help ({improvements/valid_boots*100:.0f}%)")

# ═══════════════════════════════════════════════
# TEST 2: PERMUTATION TEST
# Shuffle sla_z labels 1000 times. How often
# does random noise beat the real improvement?
# ═══════════════════════════════════════════════
print()
print("=" * 65)
print("  TEST 2: PERMUTATION TEST (1000 shuffles)")
print("=" * 65)

# Real improvement
auc_base, _ = loo_auc(X_base, y)
auc_sla, _ = loo_auc(X_sla, y)
real_delta = auc_sla - auc_base
print(f"  Real ΔAUC: {real_delta:+.4f}")

n_perm = 1000
perm_deltas = []
rng2 = np.random.RandomState(123)

for p in range(n_perm):
    # Shuffle sla_z column only
    sla_shuffled = fm_test['sla_z'].values.copy()
    rng2.shuffle(sla_shuffled)
    X_perm = np.column_stack([X_base, sla_shuffled])

    auc_perm, _ = loo_auc(X_perm, y)
    perm_deltas.append(auc_perm - auc_base)

perm_deltas = np.array(perm_deltas)
p_value = np.mean(perm_deltas >= real_delta)
print(f"  Permutation p-value: {p_value:.4f}")
print(f"  (Fraction of random shuffles that beat real improvement)")
print(f"  Mean random ΔAUC: {np.mean(perm_deltas):+.4f}")
print(f"  Max random ΔAUC: {np.max(perm_deltas):+.4f}")

if p_value < 0.05:
    print(f"  ✓ SIGNIFICANT at p<0.05 — improvement unlikely due to chance")
elif p_value < 0.10:
    print(f"  ~ MARGINAL significance (p={p_value:.3f})")
else:
    print(f"  ✗ NOT SIGNIFICANT — random noise could produce this improvement")

# ═══════════════════════════════════════════════
# TEST 3: VARIANCE INFLATION FACTOR
# ═══════════════════════════════════════════════
print()
print("=" * 65)
print("  TEST 3: VIF (Variance Inflation Factor)")
print("=" * 65)

from numpy.linalg import inv

X_vif = fm_test[feat_sla].values
sc = StandardScaler()
X_scaled = sc.fit_transform(X_vif)
corr_matrix = np.corrcoef(X_scaled.T)

try:
    inv_corr = inv(corr_matrix)
    vifs = np.diag(inv_corr)
    for i, feat in enumerate(feat_sla):
        flag = " ⚠ HIGH" if vifs[i] > 5 else (" ~ moderate" if vifs[i] > 2.5 else "")
        print(f"  {feat:<15} VIF = {vifs[i]:.2f}{flag}")
    print()
    print(f"  Rule of thumb: VIF > 5 = serious collinearity, > 10 = crisis")
except Exception as e:
    print(f"  VIF computation failed: {e}")

# ═══════════════════════════════════════════════
# TEST 4: RESIDUALIZED SLA
# Regress sla_z on sst_z + nino12_t1, use residual
# This removes the overlap and tests pure SLA info
# ═══════════════════════════════════════════════
print()
print("=" * 65)
print("  TEST 4: RESIDUALIZED SLA (remove SST/Niño overlap)")
print("=" * 65)

from sklearn.linear_model import LinearRegression

# Regress sla_z on sst_z and nino12_t1
X_explain = fm_test[['sst_z', 'nino12_t1']].values
y_sla = fm_test['sla_z'].values
lr = LinearRegression()
lr.fit(X_explain, y_sla)
sla_predicted = lr.predict(X_explain)
sla_residual = y_sla - sla_predicted
r_squared = lr.score(X_explain, y_sla)

print(f"  SLA explained by SST+Niño: R² = {r_squared:.3f}")
print(f"  Residual SLA std: {np.std(sla_residual):.3f}")
print(f"  (This is the 'new information' SLA adds beyond SST/Niño)")

# Test residual as 4th feature
X_resid = np.column_stack([X_base, sla_residual])
auc_resid, probs_resid = loo_auc(X_resid, y)
delta_resid = auc_resid - auc_base

print(f"\n  3-feature baseline AUC: {auc_base:.3f}")
print(f"  + raw sla_z AUC:       {auc_sla:.3f} ({auc_sla - auc_base:+.3f})")
print(f"  + residual sla_z AUC:  {auc_resid:.3f} ({delta_resid:+.3f})")

if delta_resid > 0.01:
    print(f"\n  ✓ Residual SLA still helps — SLA carries genuine new information")
elif delta_resid > -0.01:
    print(f"\n  ~ Residual SLA is neutral — improvement was partly from overlap")
else:
    print(f"\n  ✗ Residual SLA hurts — raw SLA improvement was collinearity artifact")

# Check SEVERE on residual model
severe_total = severe_correct = 0
for p, actual in zip(probs_resid, y):
    if p >= 0.70:
        severe_total += 1
        if actual == 1:
            severe_correct += 1
sev_resid = f"{severe_correct}/{severe_total}" if severe_total > 0 else "0/0"
sev_pct = severe_correct / severe_total * 100 if severe_total > 0 else 0
print(f"  SEVERE accuracy (residual): {sev_resid} ({'100%' if sev_pct == 100 else f'{sev_pct:.0f}%'})")

# ═══════════════════════════════════════════════
# TEST 5: PAIRED LOO COMPARISON
# For each left-out sample, does 4-feat or 3-feat
# give a better probability?
# ═══════════════════════════════════════════════
print()
print("=" * 65)
print("  TEST 5: PAIRED LOO — Per-sample comparison")
print("=" * 65)

_, probs_3 = loo_auc(X_base, y)
_, probs_4 = loo_auc(X_sla, y)

# For each sample: which model gave a "better" probability?
# Better = higher prob for positives, lower for negatives
four_better = 0
three_better = 0
tied = 0

for i in range(n):
    if y[i] == 1:
        # Want higher probability
        if probs_4[i] > probs_3[i] + 0.01:
            four_better += 1
        elif probs_3[i] > probs_4[i] + 0.01:
            three_better += 1
        else:
            tied += 1
    else:
        # Want lower probability
        if probs_4[i] < probs_3[i] - 0.01:
            four_better += 1
        elif probs_3[i] < probs_4[i] - 0.01:
            three_better += 1
        else:
            tied += 1

print(f"  4-feature better: {four_better}/{n} ({four_better/n*100:.0f}%)")
print(f"  3-feature better: {three_better}/{n} ({three_better/n*100:.0f}%)")
print(f"  Tied (< 0.01):    {tied}/{n} ({tied/n*100:.0f}%)")

# Sign test (binomial)
from scipy.stats import binomtest
if four_better + three_better > 0:
    p_sign = binomtest(four_better, four_better + three_better, 0.5).pvalue
    print(f"  Sign test p-value: {p_sign:.4f}")
    if p_sign < 0.05:
        print(f"  ✓ 4-feature model is significantly better per-sample")
    else:
        print(f"  ~ Not significant per-sample")

# ═══════════════════════════════════════════════
# TEST 6: STABILITY ACROSS S1 vs S2 SPLITS
# Does SLA help for both season types?
# ═══════════════════════════════════════════════
print()
print("=" * 65)
print("  TEST 6: S1 vs S2 SPLIT")
print("=" * 65)

for season_val in [1, 2]:
    mask = fm_test['season'] == season_val
    X_b_s = X_base[mask]
    X_s_s = X_sla[mask]
    y_s = y[mask]
    n_s = len(y_s)

    if y_s.sum() == 0 or y_s.sum() == n_s:
        print(f"  S{season_val}: skipped (no class variation)")
        continue

    auc_b_s, _ = loo_auc(X_b_s, y_s)
    auc_s_s, _ = loo_auc(X_s_s, y_s)
    d = auc_s_s - auc_b_s
    print(f"  S{season_val} (n={n_s}, pos={int(y_s.sum())}): "
          f"AUC {auc_b_s:.3f} → {auc_s_s:.3f} ({d:+.3f})")

# ═══════════════════════════════════════════════
# OVERALL VERDICT
# ═══════════════════════════════════════════════
print()
print("=" * 65)
print("  OVERALL VERDICT")
print("=" * 65)

checks_passed = 0
checks_total = 5

# Bootstrap
if improvements / valid_boots > 0.60:
    checks_passed += 1
    print(f"  ✓ Bootstrap: helps {improvements/valid_boots*100:.0f}% of the time")
else:
    print(f"  ✗ Bootstrap: only helps {improvements/valid_boots*100:.0f}%")

# Permutation
if p_value < 0.10:
    checks_passed += 1
    print(f"  ✓ Permutation: p={p_value:.3f}")
else:
    print(f"  ✗ Permutation: p={p_value:.3f} (not significant)")

# VIF
max_vif = max(vifs) if 'vifs' in dir() else 99
if max_vif < 5:
    checks_passed += 1
    print(f"  ✓ VIF: max={max_vif:.1f} (acceptable)")
else:
    print(f"  ✗ VIF: max={max_vif:.1f} (high collinearity)")

# Residual
if delta_resid > 0:
    checks_passed += 1
    print(f"  ✓ Residual SLA: still improves ({delta_resid:+.3f})")
else:
    print(f"  ✗ Residual SLA: no improvement ({delta_resid:+.3f})")

# Paired
if four_better > three_better:
    checks_passed += 1
    print(f"  ✓ Paired: 4-feat wins {four_better} vs {three_better}")
else:
    print(f"  ✗ Paired: 3-feat wins {three_better} vs {four_better}")

print(f"\n  Score: {checks_passed}/{checks_total} checks passed")
if checks_passed >= 4:
    print(f"  → STRONG EVIDENCE: sla_z is a genuine improvement")
elif checks_passed >= 3:
    print(f"  → MODERATE EVIDENCE: sla_z likely helps but with caveats")
elif checks_passed >= 2:
    print(f"  → WEAK EVIDENCE: sla_z improvement may be noise")
else:
    print(f"  → INSUFFICIENT EVIDENCE: do not add sla_z")
print("=" * 65)
