r"""
scenario_analysis.py - PAEWS Scenario Sweep

Sweeps Niño 1+2 and Chl Z-score combinations at fixed SST,
outputs a risk matrix showing probability and tier for each combo.

Usage:
    cd C:\Users\josep\Documents\paews

    # Default: current SST (+0.837), sweep Niño and Chl
    python scripts/scenario_analysis.py

    # Override SST
    python scripts/scenario_analysis.py --sst 1.2

    # Custom Niño range (e.g., just the likely Feb values)
    python scripts/scenario_analysis.py --nino-min 0.5 --nino-max 1.5

    # Export CSV
    python scripts/scenario_analysis.py --csv outputs/scenario_matrix.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import argparse
import sys

BASE = Path(__file__).resolve().parent.parent
FEATURE_CSV = BASE / "data" / "external" / "paews_feature_matrix.csv"
FEATURE_COLS = ['sst_z', 'chl_z', 'nino12_t1']

TIERS = [
    (0.00, 0.20, "LOW"),
    (0.20, 0.50, "MOD"),
    (0.50, 0.70, "ELEV"),
    (0.70, 1.01, "SEV"),
]

TIER_FULL = {"LOW": "LOW", "MOD": "MODERATE", "ELEV": "ELEVATED", "SEV": "SEVERE"}


def get_tier(p):
    for lo, hi, name in TIERS:
        if lo <= p < hi:
            return name
    return "SEV"


def train_model():
    """Train on full dataset, return fitted model and scaler."""
    df = pd.read_csv(FEATURE_CSV).dropna(subset=FEATURE_COLS + ['target'])
    X = df[FEATURE_COLS].values
    y = df['target'].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_s, y)
    return model, scaler, len(X), int(y.sum())


def predict(model, scaler, sst_z, chl_z, nino):
    """Single prediction."""
    x = np.array([[sst_z, chl_z, nino]])
    x_s = scaler.transform(x)
    return model.predict_proba(x_s)[0, 1]


def main():
    parser = argparse.ArgumentParser(description="PAEWS Scenario Sweep")
    parser.add_argument('--sst', type=float, default=None, help="Fixed SST Z-score (default: from latest run)")
    parser.add_argument('--nino-min', type=float, default=-1.0, help="Niño 1+2 sweep min (default: -1.0)")
    parser.add_argument('--nino-max', type=float, default=2.0, help="Niño 1+2 sweep max (default: 2.0)")
    parser.add_argument('--nino-step', type=float, default=0.25, help="Niño step size (default: 0.25)")
    parser.add_argument('--chl-min', type=float, default=-1.2, help="Chl Z sweep min (default: -1.2)")
    parser.add_argument('--chl-max', type=float, default=0.8, help="Chl Z sweep max (default: 0.8)")
    parser.add_argument('--chl-step', type=float, default=0.2, help="Chl step size (default: 0.2)")
    parser.add_argument('--csv', type=str, default=None, help="Export results to CSV")
    args = parser.parse_args()

    # Default SST from latest prediction
    sst_z = args.sst if args.sst is not None else 0.837

    print("=" * 72)
    print("  PAEWS SCENARIO ANALYSIS")
    print("  Fixed SST Z-score: %+.3f" % sst_z)
    print("  Niño 1+2 range: [%.2f, %.2f] step %.2f" % (args.nino_min, args.nino_max, args.nino_step))
    print("  Chl Z range:    [%.2f, %.2f] step %.2f" % (args.chl_min, args.chl_max, args.chl_step))
    print("=" * 72)

    model, scaler, n_train, n_pos = train_model()
    print("  Trained on %d samples (%d positives)" % (n_train, n_pos))

    coefs = model.coef_[0]
    print("  Coefficients: SST=%+.3f  Chl=%+.3f  Niño=%+.3f  Int=%+.3f" % (
        coefs[0], coefs[1], coefs[2], model.intercept_[0]))

    # Build sweep arrays
    nino_vals = np.arange(args.nino_min, args.nino_max + args.nino_step / 2, args.nino_step)
    chl_vals = np.arange(args.chl_min, args.chl_max + args.chl_step / 2, args.chl_step)

    # Compute full matrix
    results = []
    matrix = np.zeros((len(chl_vals), len(nino_vals)))
    tier_matrix = []

    for i, chl in enumerate(chl_vals):
        row_tiers = []
        for j, nino in enumerate(nino_vals):
            p = predict(model, scaler, sst_z, chl, nino)
            matrix[i, j] = p
            tier = get_tier(p)
            row_tiers.append(tier)
            results.append({
                'sst_z': sst_z,
                'chl_z': round(chl, 2),
                'nino12_t1': round(nino, 2),
                'probability': round(p, 3),
                'tier': TIER_FULL[tier],
            })
        tier_matrix.append(row_tiers)

    # Print matrix
    print()
    print("  RISK MATRIX  (probability / tier)")
    print("  Rows = Chl Z-score  |  Columns = Niño 1+2")
    print("  SST Z fixed at %+.3f" % sst_z)
    print()

    # Header
    hdr = "  Chl Z  |"
    for nino in nino_vals:
        hdr += " %+5.2f" % nino
    print(hdr)
    print("  " + "-" * (9 + 6 * len(nino_vals)))

    for i, chl in enumerate(chl_vals):
        line = "  %+5.2f  |" % chl
        for j, nino in enumerate(nino_vals):
            p = matrix[i, j]
            line += "  .%03d" % int(p * 1000)
        print(line)

    # Tier view
    print()
    print("  TIER MAP")
    print()
    hdr = "  Chl Z  |"
    for nino in nino_vals:
        hdr += " %+5.2f" % nino
    print(hdr)
    print("  " + "-" * (9 + 6 * len(nino_vals)))

    for i, chl in enumerate(chl_vals):
        line = "  %+5.2f  |" % chl
        for j in range(len(nino_vals)):
            t = tier_matrix[i][j]
            line += "  %4s" % t
        print(line)

    # Highlight key scenarios
    print()
    print("  KEY SCENARIOS FOR 2026 S1:")
    print("  " + "-" * 60)
    scenarios = [
        ("Current (stale data)", 0.837, +0.166, -0.29),
        ("Updated Niño +0.7", 0.837, +0.166, +0.70),
        ("Updated Niño +1.0", 0.837, +0.166, +1.00),
        ("Niño +0.7, Feb Chl proxy", 0.837, -0.40, +0.70),
        ("Niño +1.0, Feb Chl proxy", 0.837, -0.40, +1.00),
        ("Niño +1.0, bad Chl", 0.837, -0.80, +1.00),
        ("Worst case (2017-like)", 0.837, -1.00, +1.50),
    ]

    print("  %-28s  SST_Z  Chl_Z  Niño   Prob   Tier" % "Scenario")
    print("  %-28s  %s  %s  %s  %s  %s" % ("", "-----", "-----", "-----", "-----", "----"))
    for name, s, c, n in scenarios:
        p = predict(model, scaler, s, c, n)
        t = get_tier(p)
        marker = " <<<" if t == "SEV" else ""
        print("  %-28s  %+.3f  %+.3f  %+.3f  %.3f  %s%s" % (
            name, s, c, n, p, TIER_FULL[t], marker))

    # SEVERE threshold finder
    print()
    print("  SEVERE THRESHOLD FINDER (prob >= 0.70):")
    print("  At SST Z = %+.3f, what Niño/Chl combos trigger SEVERE?" % sst_z)
    print()
    found_any = False
    for nino in np.arange(-0.5, 3.01, 0.1):
        for chl in np.arange(-1.5, 1.01, 0.05):
            p = predict(model, scaler, sst_z, chl, nino)
            if 0.698 <= p <= 0.702:
                print("    Niño %+.1f + Chl Z %+.2f = %.3f (threshold)" % (nino, chl, p))
                found_any = True
                break

    if not found_any:
        # Find the minimum Niño that can reach SEVERE at worst Chl
        for nino in np.arange(0.0, 3.01, 0.1):
            p = predict(model, scaler, sst_z, -1.25, nino)
            if p >= 0.70:
                print("    Minimum Niño for SEVERE (at Chl Z = -1.25): %+.1f (p=%.3f)" % (nino, p))
                found_any = True
                break

        if not found_any:
            print("    SEVERE not reachable at current SST with training-range features.")
            print("    SST would need to increase for SEVERE to become possible.")

    # Export
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = BASE / csv_path
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print("\n  Exported %d scenarios to %s" % (len(results), csv_path))

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
