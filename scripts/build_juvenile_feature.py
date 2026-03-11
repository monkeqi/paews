"""
PAEWS Juvenile Percentage Data Collection
==========================================
Pre-filled with data from news articles, IMARPE reports, and PRODUCE resolutions.
Josep: fill in the remaining gaps, then run test_juvenile_feature.py

Data source hierarchy (most reliable first):
  1. IMARPE cruise report (pre-season survey) — "Crucero de Evaluación Hidroacústica"
  2. PRODUCE Resolución Ministerial (cites IMARPE survey)
  3. IMARPE daily fishing reports (during-season, NOT pre-season)
  4. News articles citing IMARPE data

CRITICAL LEAKAGE NOTE:
  - PRE-SEASON juvenile % (from IMARPE survey BEFORE fishing starts) = NO LEAKAGE
  - DURING-SEASON juvenile % (from fishing catches) = LEAKAGE for prediction
  - We want the PRE-SEASON survey value, or the value cited in the resolution
    that opens the season (which references the pre-season survey)

Usage:
    1. Fill gaps marked with None below
    2. python scripts/build_juvenile_feature.py
    3. python scripts/test_juvenile_feature.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("c:/Users/josep/Documents/paews")
DATA_EXTERNAL = BASE_DIR / "data" / "external"
FEAT_PATH = DATA_EXTERNAL / "paews_feature_matrix.csv"
GT_PATH = DATA_EXTERNAL / "imarpe_ground_truth.csv"
JUVENILE_PATH = DATA_EXTERNAL / "juvenile_pct_data.csv"

# ══════════════════════════════════════════════════════════════
# JUVENILE DATA — fill in None values
# ══════════════════════════════════════════════════════════════
# juv_pct = percentage of juveniles (<12cm) in IMARPE pre-season survey
# juv_source = where the number comes from
# juv_type = "pre_season" (safe) or "during_season" (leakage risk)

juvenile_data = [
    # 2010 S1 — need to find
    {"year": 2010, "season": 1, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},
    # 2010 S2 — need to find
    {"year": 2010, "season": 2, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},

    # 2011 S1 — need to find
    {"year": 2011, "season": 1, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},
    # 2011 S2 — need to find (REDUCED season)
    {"year": 2011, "season": 2, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},

    # 2012 S1 — need to find (REDUCED season)
    {"year": 2012, "season": 1, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},
    # 2012 S2 — need to find (REDUCED season)
    {"year": 2012, "season": 2, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},

    # 2013 S1 — need to find
    {"year": 2013, "season": 1, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},
    # 2013 S2 — need to find
    {"year": 2013, "season": 2, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},

    # 2014 S1 — need to find (DISRUPTED - ghost miss)
    {"year": 2014, "season": 1, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},
    # 2014 S2 — need to find (REDUCED)
    {"year": 2014, "season": 2, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},

    # 2015 S1 — need to find (REDUCED)
    {"year": 2015, "season": 1, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},
    # 2015 S2 — need to find (DISRUPTED)
    {"year": 2015, "season": 2, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},

    # 2016 S1 — need to find (DISRUPTED)
    {"year": 2016, "season": 1, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},
    # 2016 S2 — need to find
    {"year": 2016, "season": 2, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},

    # 2017 S1 — need to find (REDUCED - Coastal El Nino)
    {"year": 2017, "season": 1, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},
    # 2017 S2 — 96% juveniles in IMARPE cruise CR1709-11
    {"year": 2017, "season": 2, "juv_pct": 96, "juv_source": "IMARPE cruise CR1709-11, Oceana Peru blog 2018-02-10", "juv_type": "pre_season"},

    # 2018 S1 — need to find
    {"year": 2018, "season": 1, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},
    # 2018 S2 — need to find
    {"year": 2018, "season": 2, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},

    # 2019 S1 — need to find
    {"year": 2019, "season": 1, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},
    # 2019 S2 — biomass 8.34 MMT, season had high juvenile closures
    # Imarpe biomass scandal (inflated from ~4 to 8.3 MMT)
    {"year": 2019, "season": 2, "juv_pct": None, "juv_source": "Gestion 2020-02-04 reports high juveniles during season", "juv_type": "pre_season"},

    # 2020 S1 — need to find
    {"year": 2020, "season": 1, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},
    # 2020 S2 — need to find
    {"year": 2020, "season": 2, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},

    # 2021 S1 — need to find
    {"year": 2021, "season": 1, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},
    # 2021 S2 — need to find
    {"year": 2021, "season": 2, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},

    # 2022 S1 — ~4% juveniles expected (Produce/IMARPE), biomass 9.78 MMT
    {"year": 2022, "season": 1, "juv_pct": 4, "juv_source": "La Republica 2022-06-24: Produce says 4% expected", "juv_type": "pre_season"},
    # 2022 S2 — 70% juveniles in catch, closed early (DISRUPTED)
    # Need pre-season survey value, not during-season catch
    {"year": 2022, "season": 2, "juv_pct": None, "juv_source": "During-season: 70% in catch per ground truth. Need pre-season survey value", "juv_type": "pre_season"},

    # 2023 S1 — 82-91% juveniles, CANCELLED
    {"year": 2023, "season": 1, "juv_pct": 86, "juv_source": "Ojo Publico: 82-91% in IMARPE surveys Feb-May 2023. Infobae: 86.3%", "juv_type": "pre_season"},
    # 2023 S2 — exploratory fishing showed enough adults to open
    {"year": 2023, "season": 2, "juv_pct": None, "juv_source": "Season opened after exploratory fishing Oct 21-25", "juv_type": "pre_season"},

    # 2024 S1 — ~36% during season (CooperAccion), need pre-season
    {"year": 2024, "season": 1, "juv_pct": None, "juv_source": "CooperAccion 2024-05-22: 36% during season. Need pre-season value", "juv_type": "pre_season"},
    # 2024 S2 — 14.5% in first 2 weeks (CooperAccion Nov 2024)
    {"year": 2024, "season": 2, "juv_pct": None, "juv_source": "CooperAccion 2024-11-20: 14.5% early season. Need pre-season", "juv_type": "pre_season"},

    # 2025 S1 — need to find
    {"year": 2025, "season": 1, "juv_pct": None, "juv_source": "Oceana Peru 2025-07-16: 24.4% accumulated during season", "juv_type": "pre_season"},
    # 2025 S2 — need to find
    {"year": 2025, "season": 2, "juv_pct": None, "juv_source": "", "juv_type": "pre_season"},
]


def build_and_save():
    """Build juvenile CSV and print collection status."""
    df = pd.DataFrame(juvenile_data)
    df.to_csv(JUVENILE_PATH, index=False)

    filled = df['juv_pct'].notna().sum()
    total = len(df)
    missing = total - filled

    print(f"{'='*60}")
    print(f"JUVENILE PERCENTAGE DATA STATUS")
    print(f"{'='*60}")
    print(f"  Filled:  {filled}/{total}")
    print(f"  Missing: {missing}/{total}")
    print()

    print(f"  FILLED:")
    for _, row in df[df['juv_pct'].notna()].iterrows():
        outcome = ""
        gt = pd.read_csv(GT_PATH)
        gt_row = gt[(gt['year'] == row['year']) & (gt['season'] == row['season'])]
        if len(gt_row) > 0:
            outcome = gt_row['outcome'].values[0]
        print(f"    {int(row['year'])} S{int(row['season'])} ({outcome:>10}): "
              f"{row['juv_pct']:.0f}% — {row['juv_source'][:60]}")

    print()
    print(f"  MISSING (need to find):")
    for _, row in df[df['juv_pct'].isna()].iterrows():
        gt = pd.read_csv(GT_PATH)
        gt_row = gt[(gt['year'] == row['year']) & (gt['season'] == row['season'])]
        outcome = gt_row['outcome'].values[0] if len(gt_row) > 0 else ""
        priority = "★★★" if outcome in ['DISRUPTED', 'CANCELLED'] else "★"
        hint = row['juv_source'][:50] if row['juv_source'] else "No leads"
        print(f"    {int(row['year'])} S{int(row['season'])} ({outcome:>10}) {priority}  {hint}")

    print()
    print(f"  Saved: {JUVENILE_PATH}")
    print()
    print("=" * 60)
    print("HOW TO FIND MISSING VALUES")
    print("=" * 60)
    print("""
  BEST SOURCE: IMARPE Cruise Reports (Boletín del IMARPE)
    Search: imarpe.gob.pe repositorio
    Look for: "Crucero de Evaluación Hidroacústica" reports
    The pre-season survey always reports juvenile %
    Cruise codes in imarpe_biomass_verified.csv

  SECOND: PRODUCE Resoluciones Ministeriales
    Search: gob.pe/produce → Normas legales
    Or: busquedas.elperuano.pe → search "anchoveta temporada"
    The resolution that OPENS each season cites IMARPE's
    pre-season survey results including juvenile %

  THIRD: News articles (Gestión, Infobae, La República)
    Search: "[year] temporada anchoveta juveniles porcentaje"
    These often quote the IMARPE survey numbers

  PRIORITY: Fill disrupted seasons first (★★★ above)
    2014 S1, 2015 S2, 2016 S1, 2022 S2 are the most important
    because these are the seasons the model gets wrong.
""")


def merge_into_feature_matrix():
    """Add juv_pct to feature matrix once data is collected."""
    if not JUVENILE_PATH.exists():
        print("Run build_and_save() first")
        return

    juv = pd.read_csv(JUVENILE_PATH)
    fm = pd.read_csv(FEAT_PATH)

    if 'juv_pct' not in fm.columns:
        fm['juv_pct'] = np.nan

    updated = 0
    for _, row in juv.iterrows():
        if pd.notna(row['juv_pct']):
            mask = (fm['year'] == row['year']) & (fm['season'] == row['season'])
            if mask.any():
                fm.loc[mask, 'juv_pct'] = row['juv_pct']
                updated += 1

    fm.to_csv(FEAT_PATH, index=False)
    print(f"Updated {updated} seasons with juv_pct in feature matrix")


if __name__ == "__main__":
    build_and_save()
