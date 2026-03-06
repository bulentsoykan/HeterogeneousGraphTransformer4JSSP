#!/usr/bin/env python3
"""
Publication-quality figure generator for HGT-Scheduler.

Reads CSV files from results/ directory and produces:
  Fig 1 – Learning curves (mean makespan vs training steps)
  Fig 2 – Main comparison bar chart (all methods × all instances)
  Fig 3 – Optimality gap comparison
  Fig 4 – Ablation study bar chart

Usage:
    python generate_figures.py [--results-dir results] [--output-dir figures]
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from pathlib import Path

# ── Publication style ─────────────────────────────────────────────────

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'font.size':       11,
    'axes.titlesize':  12,
    'axes.labelsize':  11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.dpi':      150,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
    'lines.linewidth':   2.0,
    'errorbar.capsize':  4,
})

# Method display names and colors
METHOD_INFO = {
    'HGT':     ('HGT-Scheduler (ours)', '#2166AC', '-',  'o'),
    'GIN':     ('GIN (L2D-style)',      '#D6604D', '--', 's'),
    'HomoHGT': ('Homo-HGT (ablation)', '#4DAC26', ':',  '^'),
    'Random':  ('Random',               '#878787', '-.',  'x'),
    'SPT':     ('SPT heuristic',        '#E08214', '-.',  'D'),
    'LPT':     ('LPT heuristic',        '#7B3294', '-.',  'P'),
}

INSTANCE_LABELS = {
    'FT06': 'FT06 (6×6)',
    'FT10': 'FT10 (10×10)',
    'FT20': 'FT20 (20×5)',
}

OPTIMAL = {'FT06': 55, 'FT10': 930, 'FT20': 1165}

ABLATION_DISPLAY = {
    'HGT-Full':   'HGT-Full\n(Ours)',
    'HGT-Homo':   'Homo-HGT\n(no edge types)',
    'GIN-Base':   'GIN\n(L2D-style)',
    'HGT-1Layer': '1 Layer',
    'HGT-2Layer': '2 Layers',
    'HGT-3Layer': 'HGT-Full\n(3 Layers)',
    'HGT-4Layer': '4 Layers',
    'HGT-1Head':  '1 Head',
    'HGT-8Heads': '8 Heads',
}

# ── Helper ────────────────────────────────────────────────────────────

def smooth(values, window=5):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window//2, window//2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(values)]


def load_data(results_dir: Path):
    eval_df  = pd.read_csv(results_dir / 'eval_results.csv')
    curve_df = pd.read_csv(results_dir / 'training_curves.csv') \
               if (results_dir / 'training_curves.csv').exists() else pd.DataFrame()
    summary  = pd.read_csv(results_dir / 'summary_stats.csv')
    stat_df  = pd.read_csv(results_dir / 'statistical_tests.csv') \
               if (results_dir / 'statistical_tests.csv').exists() else pd.DataFrame()
    abl_path = results_dir / 'ablation_summary.csv'
    abl_df   = pd.read_csv(abl_path) if abl_path.exists() else pd.DataFrame()
    return eval_df, curve_df, summary, stat_df, abl_df


# ── Figure 1: Learning Curves ─────────────────────────────────────────

def plot_learning_curves(curve_df: pd.DataFrame, out_dir: Path):
    if curve_df.empty:
        print("  [skip] No training curves data")
        return

    instances = sorted(curve_df['instance'].unique())
    n_inst = len(instances)
    fig, axes = plt.subplots(1, n_inst, figsize=(5.5 * n_inst, 4.2),
                              constrained_layout=True)
    if n_inst == 1:
        axes = [axes]

    for ax, instance in zip(axes, instances):
        inst_df = curve_df[curve_df['instance'] == instance]
        optimal = OPTIMAL.get(instance)

        for method in ['HGT', 'GIN', 'HomoHGT']:
            mdf = inst_df[inst_df['method'] == method]
            if mdf.empty:
                continue
            label, color, ls, marker = METHOD_INFO.get(
                method, (method, 'gray', '-', 'o'))

            # Aggregate across seeds: mean and std per timestep
            grp = mdf.groupby('timestep')['mean_makespan']
            ts   = grp.mean().index.values / 1000   # x-axis: k steps
            means = smooth(grp.mean().values)
            stds  = grp.std().fillna(0).values

            ax.plot(ts, means, label=label, color=color, linestyle=ls,
                    marker=marker, markersize=4, markevery=max(1, len(ts)//8))
            ax.fill_between(ts, means - stds, means + stds,
                            alpha=0.15, color=color)

        if optimal:
            ax.axhline(optimal, color='k', linestyle=':', linewidth=1.5,
                       label=f'Optimal ({optimal})')

        ax.set_title(INSTANCE_LABELS.get(instance, instance), fontweight='bold')
        ax.set_xlabel('Training Steps (×10³)')
        ax.set_ylabel('Mean Makespan')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    fig.suptitle('Learning Curves: HGT-Scheduler vs Baselines',
                 fontsize=13, fontweight='bold', y=1.04)

    path = out_dir / 'fig1_learning_curves.pdf'
    fig.savefig(path)
    fig.savefig(str(path).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 2: Main Comparison Bar Chart ──────────────────────────────

def plot_main_comparison(summary: pd.DataFrame, stat_df: pd.DataFrame,
                          out_dir: Path):
    instances = sorted(summary['instance'].unique())
    methods_order = ['Random', 'LPT', 'SPT', 'GIN', 'HomoHGT', 'HGT']
    methods_present = [m for m in methods_order if m in summary['method'].values]

    n_inst = len(instances)
    fig, axes = plt.subplots(1, n_inst, figsize=(4.8 * n_inst, 5.0),
                              constrained_layout=True)
    if n_inst == 1:
        axes = [axes]

    bar_width = 0.65 / len(methods_present)

    for ax, instance in zip(axes, instances):
        inst_sum = summary[summary['instance'] == instance]
        optimal  = OPTIMAL.get(instance)

        x_pos = np.arange(len(methods_present))
        for i, method in enumerate(methods_present):
            row = inst_sum[inst_sum['method'] == method]
            if row.empty:
                continue
            mean = float(row['mean'].values[0])
            std  = float(row['std'].values[0])
            label, color, _, marker = METHOD_INFO.get(method, (method, 'gray', '-', 'o'))

            bar = ax.bar(i, mean, bar_width * len(methods_present),
                         color=color, alpha=0.85, zorder=3)
            ax.errorbar(i, mean, yerr=std, fmt='none', color='black',
                        elinewidth=1.5, capsize=5, zorder=4)

            # Significance markers vs HGT
            if method != 'HGT' and len(stat_df) > 0:
                sub = stat_df[(stat_df['instance'] == instance) &
                               (stat_df['HGT_vs'] == method)]
                if not sub.empty:
                    pval = float(sub['p_value'].values[0])
                    if pval < 0.01:
                        sig = '**'
                    elif pval < 0.05:
                        sig = '*'
                    else:
                        sig = ''
                    if sig:
                        ax.text(i, mean + std + (optimal * 0.005 if optimal else 5),
                                sig, ha='center', va='bottom', fontsize=12,
                                color='darkred', fontweight='bold')

        if optimal:
            ax.axhline(optimal, color='k', linestyle='--', linewidth=1.5,
                       label=f'Optimal ({optimal})', zorder=2)

        ax.set_title(INSTANCE_LABELS.get(instance, instance), fontweight='bold')
        ax.set_ylabel('Makespan (lower is better)')
        ax.set_xticks(range(len(methods_present)))
        ax.set_xticklabels([METHOD_INFO.get(m, (m,))[0].replace(' (ours)', '')
                             .replace(' (L2D-style)', '').replace(' (ablation)', '')
                             for m in methods_present],
                           rotation=30, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.grid(axis='x', alpha=0)

        if optimal:
            ax.legend(loc='upper left', fontsize=9)

        # Add value labels on bars
        for i, method in enumerate(methods_present):
            row = inst_sum[inst_sum['method'] == method]
            if row.empty:
                continue
            mean = float(row['mean'].values[0])
            ax.text(i, mean * 0.995, f'{mean:.1f}',
                    ha='center', va='top', fontsize=8, color='white',
                    fontweight='bold')

    # Significance note
    fig.text(0.5, -0.04,
             '* p < 0.05, ** p < 0.01 (paired t-test vs HGT-Scheduler)',
             ha='center', fontsize=9, style='italic')

    fig.suptitle('Scheduling Performance Comparison',
                 fontsize=13, fontweight='bold')

    path = out_dir / 'fig2_main_comparison.pdf'
    fig.savefig(path)
    fig.savefig(str(path).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 3: Optimality Gap ──────────────────────────────────────────

def plot_optimality_gap(summary: pd.DataFrame, stat_df: pd.DataFrame,
                         out_dir: Path):
    # Only instances with known optimal AND at least one learning method
    learning_methods = {'HGT', 'GIN', 'HomoHGT'}
    inst_with_opt = [
        i for i in sorted(summary['instance'].unique())
        if i in OPTIMAL and
        not summary[(summary['instance'] == i) &
                    (summary['method'].isin(learning_methods))].empty
    ]
    if not inst_with_opt:
        print("  [skip] No instances with known optimal")
        return

    methods_order = ['Random', 'LPT', 'SPT', 'GIN', 'HomoHGT', 'HGT']
    methods_present = [m for m in methods_order if m in summary['method'].values]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    n_inst   = len(inst_with_opt)
    n_method = len(methods_present)
    group_w  = 0.8
    bar_w    = group_w / n_method

    x_base = np.arange(n_inst)
    offsets = np.linspace(-group_w/2 + bar_w/2, group_w/2 - bar_w/2, n_method)

    legend_handles = []
    for mi, method in enumerate(methods_present):
        label, color, ls, marker = METHOD_INFO.get(method, (method, 'gray', '-', 'o'))
        gaps, errs = [], []
        for instance in inst_with_opt:
            row = summary[(summary['method'] == method) &
                           (summary['instance'] == instance)]
            if row.empty:
                gaps.append(0); errs.append(0)
            else:
                gaps.append(float(row['gap_mean'].values[0]
                                  if 'gap_mean' in row.columns else 0))
                errs.append(float(row['gap_std'].values[0]
                                  if 'gap_std' in row.columns else 0))

        bars = ax.bar(x_base + offsets[mi], gaps, bar_w,
                      color=color, alpha=0.85, zorder=3)
        ax.errorbar(x_base + offsets[mi], gaps, errs,
                    fmt='none', color='black', elinewidth=1, capsize=3, zorder=4)
        legend_handles.append(mpatches.Patch(color=color, alpha=0.85, label=label))

    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_xticks(x_base)
    ax.set_xticklabels([INSTANCE_LABELS.get(i, i) for i in inst_with_opt])
    ax.set_ylabel('Optimality Gap (%)')
    ax.set_title('Optimality Gap vs Known Optimal Makespan',
                 fontweight='bold', fontsize=13)
    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f%%'))

    path = out_dir / 'fig3_optimality_gap.pdf'
    fig.savefig(path)
    fig.savefig(str(path).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 4: Ablation Study ──────────────────────────────────────────

def plot_ablation(abl_df: pd.DataFrame, out_dir: Path):
    if abl_df.empty:
        print("  [skip] No ablation data")
        return

    instance = 'FT06'
    optimal  = OPTIMAL.get(instance)

    # Sort by mean makespan
    abl_sorted = abl_df.sort_values('mean')
    variants = abl_sorted['variant'].tolist()
    means    = abl_sorted['mean'].values
    stds     = abl_sorted['std'].values

    # Color scheme: highlight 'HGT-Full' in blue
    colors = []
    for v in variants:
        if v == 'HGT-Full':
            colors.append('#2166AC')
        elif 'Homo' in v or 'GIN' in v:
            colors.append('#D6604D')
        else:
            colors.append('#74ADD1')

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    x = np.arange(len(variants))
    bars = ax.bar(x, means, color=colors, alpha=0.85, zorder=3, width=0.6)
    ax.errorbar(x, means, stds, fmt='none', color='black',
                elinewidth=1.5, capsize=5, zorder=4)

    if optimal:
        ax.axhline(optimal, color='k', linestyle='--', linewidth=1.5,
                   label=f'Optimal = {optimal}', zorder=2)

    # Annotate gap above each bar
    for xi, (mean, std) in enumerate(zip(means, stds)):
        if optimal:
            gap = (mean - optimal) / optimal * 100
            ax.text(xi, mean + std + 0.5, f'+{gap:.1f}%',
                    ha='center', va='bottom', fontsize=8.5,
                    color='darkred' if gap > 5 else 'navy')

    ax.set_xticks(x)
    labels = [ABLATION_DISPLAY.get(v, v) for v in variants]
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9.5)
    ax.set_ylabel(f'Mean Makespan on {INSTANCE_LABELS.get(instance, instance)}')
    ax.set_title('Ablation Study: Component Contributions',
                 fontweight='bold', fontsize=13)
    if optimal:
        ax.legend(fontsize=10)

    # Legend for colors
    from matplotlib.patches import Patch
    leg_patches = [
        Patch(facecolor='#2166AC', alpha=0.85, label='HGT-Full (ours)'),
        Patch(facecolor='#D6604D', alpha=0.85, label='Alternative methods'),
        Patch(facecolor='#74ADD1', alpha=0.85, label='HGT variants'),
    ]
    ax.legend(handles=leg_patches + ([mpatches.Patch(color='k', label=f'Optimal={optimal}')]
                                      if optimal else []),
              loc='upper right', fontsize=9)

    path = out_dir / 'fig4_ablation.pdf'
    fig.savefig(path)
    fig.savefig(str(path).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 5: Per-seed box plots ──────────────────────────────────────

def plot_boxplots(eval_df: pd.DataFrame, out_dir: Path):
    """Box plots of makespan distribution per method and instance."""
    learning_methods = {'HGT', 'GIN', 'HomoHGT'}
    instances = sorted(i for i in eval_df['instance'].unique()
                       if not eval_df[(eval_df['instance'] == i) &
                                      (eval_df['method'].isin(learning_methods))].empty)
    methods_order = ['Random', 'SPT', 'GIN', 'HomoHGT', 'HGT']
    methods_present = [m for m in methods_order if m in eval_df['method'].values]

    n_inst = len(instances)
    fig, axes = plt.subplots(1, n_inst, figsize=(5.5 * n_inst, 5.0),
                              constrained_layout=True)
    if n_inst == 1:
        axes = [axes]

    for ax, instance in zip(axes, instances):
        inst_df = eval_df[eval_df['instance'] == instance]
        data_by_method = []
        labels_ax = []
        colors_ax = []
        for method in methods_present:
            mdf = inst_df[inst_df['method'] == method]
            if mdf.empty:
                continue
            data_by_method.append(mdf['makespan'].values)
            display, color, _, _ = METHOD_INFO.get(method, (method, 'gray', '-', 'o'))
            labels_ax.append(display.replace(' (ours)', '')
                              .replace(' (L2D-style)', '').replace(' (ablation)', ''))
            colors_ax.append(color)

        bp = ax.boxplot(data_by_method, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        flierprops=dict(marker='o', markersize=3, alpha=0.5))
        for patch, color in zip(bp['boxes'], colors_ax):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        optimal = OPTIMAL.get(instance)
        if optimal:
            ax.axhline(optimal, color='k', linestyle='--', linewidth=1.5,
                       label=f'Optimal ({optimal})')
            ax.legend(fontsize=9)

        ax.set_title(INSTANCE_LABELS.get(instance, instance), fontweight='bold')
        ax.set_xticks(range(1, len(labels_ax) + 1))
        ax.set_xticklabels(labels_ax, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Makespan')

    fig.suptitle('Makespan Distribution Across Methods and Instances',
                 fontsize=13, fontweight='bold')

    path = out_dir / 'fig5_boxplots.pdf'
    fig.savefig(path)
    fig.savefig(str(path).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 6: Convergence comparison ─────────────────────────────────

def plot_convergence_comparison(curve_df: pd.DataFrame, out_dir: Path):
    """Normalized convergence: how fast each method approaches its final value."""
    if curve_df.empty:
        return

    instances = [i for i in ['FT06', 'FT10'] if i in curve_df['instance'].values]
    if not instances:
        return

    fig, axes = plt.subplots(1, len(instances), figsize=(5.5 * len(instances), 4.2),
                              constrained_layout=True)
    if len(instances) == 1:
        axes = [axes]

    for ax, instance in zip(axes, instances):
        inst_df = curve_df[curve_df['instance'] == instance]
        optimal = OPTIMAL.get(instance)

        for method in ['HGT', 'GIN', 'HomoHGT']:
            mdf = inst_df[inst_df['method'] == method]
            if mdf.empty:
                continue
            label, color, ls, marker = METHOD_INFO[method]

            grp  = mdf.groupby('timestep')['mean_makespan']
            ts   = grp.mean().index.values / 1000
            means = grp.mean().values

            # Compute optimality gap trajectory
            if optimal:
                gaps = (means - optimal) / optimal * 100
                ax.plot(ts, smooth(gaps), label=label, color=color,
                        linestyle=ls, marker=marker, markersize=4,
                        markevery=max(1, len(ts)//8))
                ax.fill_between(ts,
                                smooth(gaps) - grp.std().fillna(0).values / optimal * 100,
                                smooth(gaps) + grp.std().fillna(0).values / optimal * 100,
                                alpha=0.12, color=color)

        ax.set_title(INSTANCE_LABELS.get(instance, instance), fontweight='bold')
        ax.set_xlabel('Training Steps (×10³)')
        ax.set_ylabel('Optimality Gap (%)')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f%%'))

    fig.suptitle('Convergence: Optimality Gap During Training',
                 fontsize=13, fontweight='bold')

    path = out_dir / 'fig6_convergence_gap.pdf'
    fig.savefig(path)
    fig.savefig(str(path).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--output-dir',  default='figures')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading results from {results_dir} ...")
    eval_df, curve_df, summary, stat_df, abl_df = load_data(results_dir)

    print(f"Generating publication figures → {out_dir}\n")
    plot_learning_curves(curve_df, out_dir)
    plot_main_comparison(summary, stat_df, out_dir)
    plot_optimality_gap(summary, stat_df, out_dir)
    plot_ablation(abl_df, out_dir)
    plot_boxplots(eval_df, out_dir)
    plot_convergence_comparison(curve_df, out_dir)

    print(f"\nAll figures saved to {out_dir.resolve()}")


if __name__ == '__main__':
    main()
