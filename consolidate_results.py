#!/usr/bin/env python3
"""
Consolidate all per-seed eval/curve CSVs into the aggregated files
required by generate_figures.py and generate_tables.py.

Outputs (in results/):
  eval_results.csv       – stacked individual evaluation episodes
  training_curves.csv    – stacked training curve data
  summary_stats.csv      – per-(method, instance): mean, std, gap_mean, gap_std
  statistical_tests.csv  – paired t-tests: HGT vs each other method
  ablation_summary.csv   – ablation variants (only if ablation data present)

Usage:
    python consolidate_results.py [--results-dir results]
"""

import argparse
import glob
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

OPTIMAL = {'FT06': 55, 'FT10': 930, 'FT20': 1165}

# ── helpers ───────────────────────────────────────────────────────────

def load_eval_csvs(results_dir: Path) -> pd.DataFrame:
    """Load all eval_{method}_{instance}_seed{seed}.csv files."""
    frames = []

    # Learning-method CSVs
    for path in sorted(results_dir.glob('eval_*_seed*.csv')):
        stem = path.stem  # e.g. eval_HGT_FT06_seed3
        m = re.match(r'^eval_(.+?)_(FT\d+)_seed(\d+)$', stem)
        if not m:
            continue
        df = pd.read_csv(path)
        frames.append(df)

    # Baselines CSV
    bl_path = results_dir / 'eval_baselines.csv'
    if bl_path.exists():
        df_bl = pd.read_csv(bl_path)
        # Keep relevant columns
        cols = [c for c in ['method', 'instance', 'seed', 'episode', 'makespan']
                if c in df_bl.columns]
        frames.append(df_bl[cols])

    if not frames:
        raise FileNotFoundError("No eval CSV files found in " + str(results_dir))

    combined = pd.concat(frames, ignore_index=True)
    # Ensure numeric types
    combined['makespan'] = combined['makespan'].astype(float)
    combined['seed']     = combined['seed'].astype(int)
    return combined


def load_curve_csvs(results_dir: Path) -> pd.DataFrame:
    """Load all curve_{method}_{instance}_seed{seed}.csv files."""
    frames = []
    for path in sorted(results_dir.glob('curve_*_seed*.csv')):
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (method, instance):
      1. Compute per-seed mean makespan.
      2. Aggregate over seeds → mean±std of those per-seed means.
      3. Compute gap statistics vs known optimal.
    """
    rows = []
    for (method, instance), grp in eval_df.groupby(['method', 'instance']):
        # Per-seed means
        seed_means = grp.groupby('seed')['makespan'].mean()
        n_seeds = len(seed_means)
        mean_ms = float(seed_means.mean())
        std_ms  = float(seed_means.std(ddof=1)) if n_seeds > 1 else 0.0

        opt = OPTIMAL.get(instance)
        if opt:
            seed_gaps = (seed_means - opt) / opt * 100
            gap_mean = float(seed_gaps.mean())
            gap_std  = float(seed_gaps.std(ddof=1)) if n_seeds > 1 else 0.0
        else:
            gap_mean = np.nan
            gap_std  = np.nan

        rows.append({
            'method':   method,
            'instance': instance,
            'n_seeds':  n_seeds,
            'mean':     mean_ms,
            'std':      std_ms,
            'gap_mean': gap_mean,
            'gap_std':  gap_std,
        })

    return pd.DataFrame(rows).sort_values(['instance', 'mean'])


def run_stat_tests(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (instance, baseline): paired t-test, HGT per-seed means vs baseline per-seed means.
    Requires both sides to have results on the same set of seeds (uses available intersection).
    """
    rows = []
    instances = eval_df['instance'].unique()

    for instance in instances:
        inst_df = eval_df[eval_df['instance'] == instance]
        hgt_df  = inst_df[inst_df['method'] == 'HGT']
        if hgt_df.empty:
            continue

        hgt_seed_means = hgt_df.groupby('seed')['makespan'].mean()

        for method in inst_df['method'].unique():
            if method == 'HGT':
                continue
            other_df = inst_df[inst_df['method'] == method]
            other_seed_means = other_df.groupby('seed')['makespan'].mean()

            # Intersect seeds
            common_seeds = hgt_seed_means.index.intersection(other_seed_means.index)
            if len(common_seeds) < 2:
                continue

            hgt_vals   = hgt_seed_means[common_seeds].values
            other_vals = other_seed_means[common_seeds].values

            t_stat, p_val = stats.ttest_rel(hgt_vals, other_vals)
            rel_imp = float(np.mean(other_vals - hgt_vals) / np.mean(other_vals) * 100)

            rows.append({
                'instance':         instance,
                'HGT_vs':           method,
                'n_seeds':          len(common_seeds),
                'HGT_mean':         float(hgt_vals.mean()),
                'other_mean':       float(other_vals.mean()),
                'rel_improvement_pct': rel_imp,
                't_statistic':      float(t_stat),
                'p_value':          float(p_val),
                'significant_005':  p_val < 0.05,
                'significant_001':  p_val < 0.01,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(['instance', 'p_value'])


# ── main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results')
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    print(f"\nConsolidating results from {results_dir} ...\n")

    # ── 1. Load & save eval_results.csv ──────────────────────────────
    eval_df = load_eval_csvs(results_dir)
    print(f"Loaded {len(eval_df):,} evaluation rows")
    print(f"  Methods:   {sorted(eval_df['method'].unique())}")
    print(f"  Instances: {sorted(eval_df['instance'].unique())}")

    out_eval = results_dir / 'eval_results.csv'
    eval_df.to_csv(out_eval, index=False)
    print(f"  → {out_eval}")

    # ── 2. Load & save training_curves.csv ───────────────────────────
    curve_df = load_curve_csvs(results_dir)
    if not curve_df.empty:
        print(f"\nLoaded {len(curve_df):,} training-curve rows")
        out_curve = results_dir / 'training_curves.csv'
        curve_df.to_csv(out_curve, index=False)
        print(f"  → {out_curve}")
    else:
        print("\nNo training curve files found; skipping training_curves.csv")

    # ── 3. Summary stats ──────────────────────────────────────────────
    summary = compute_summary(eval_df)
    out_sum = results_dir / 'summary_stats.csv'
    summary.to_csv(out_sum, index=False)
    print(f"\nSummary stats → {out_sum}")
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

    # ── 4. Statistical tests ──────────────────────────────────────────
    stat_df = run_stat_tests(eval_df)
    if not stat_df.empty:
        out_stat = results_dir / 'statistical_tests.csv'
        stat_df.to_csv(out_stat, index=False)
        print(f"\nStatistical tests → {out_stat}")
        print(stat_df[['instance','HGT_vs','n_seeds','HGT_mean','other_mean',
                        'rel_improvement_pct','p_value','significant_005']].to_string(index=False))
    else:
        print("\n[skip] Not enough data for statistical tests")

    # ── 5. Ablation summary (if ablation data present) ────────────────
    abl_path = results_dir / 'ablation_summary.csv'
    if not abl_path.exists():
        # Check for any ablation eval CSVs
        abl_files = list(results_dir.glob('eval_HGT-*_*.csv'))
        if abl_files:
            abl_frames = []
            for path in abl_files:
                df = pd.read_csv(path)
                abl_frames.append(df)
            abl_df = pd.concat(abl_frames, ignore_index=True)
            abl_sum = abl_df.groupby('method').apply(
                lambda g: pd.Series({
                    'variant': g['method'].iloc[0],
                    'mean': g.groupby('seed')['makespan'].mean().mean(),
                    'std':  g.groupby('seed')['makespan'].mean().std(),
                })
            ).reset_index(drop=True)
            abl_sum.to_csv(abl_path, index=False)
            print(f"\nAblation summary → {abl_path}")
        else:
            print("\n[skip] No ablation data found")
    else:
        print(f"\nAblation summary already exists: {abl_path}")

    print("\nConsolidation complete.")


if __name__ == '__main__':
    main()
