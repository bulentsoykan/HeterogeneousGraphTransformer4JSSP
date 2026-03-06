#!/usr/bin/env python3
"""
LaTeX table generator for HGT-Scheduler publication.

Produces production-ready LaTeX tables:
  Table 1 – Main performance comparison
  Table 2 – Ablation study
  Table 3 – Statistical significance

Usage:
    python generate_tables.py [--results-dir results] [--output-dir tables]
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

OPTIMAL   = {'FT06': 55, 'FT10': 930, 'FT20': 1165}
INSTANCE_LABELS = {'FT06': r'FT06 ($6\times6$)',
                    'FT10': r'FT10 ($10\times10$)',
                    'FT20': r'FT20 ($20\times5$)'}

METHOD_DISPLAY = {
    'Random':  'Random',
    'LPT':     'LPT',
    'SPT':     'SPT',
    'GIN':     r'GIN (L2D-style)~\cite{zhang2020learning}',
    'HomoHGT': r'Homo-HGT (ablation)',
    'HGT':     r'\textbf{HGT-Scheduler (ours)}',
}

ABLATION_DISPLAY = {
    'HGT-Full':   r'\textbf{HGT-Full (Ours)}',
    'HGT-Homo':   r'Homo-HGT (no edge types)',
    'GIN-Base':   r'GIN (L2D-style)',
    'HGT-1Layer': r'HGT-1Layer',
    'HGT-2Layer': r'HGT-2Layer',
    'HGT-4Layer': r'HGT-4Layer',
    'HGT-1Head':  r'HGT-1Head',
    'HGT-8Heads': r'HGT-8Heads',
}

# ── Load data ─────────────────────────────────────────────────────────

def load_data(results_dir: Path):
    eval_df  = pd.read_csv(results_dir / 'eval_results.csv')
    summary  = pd.read_csv(results_dir / 'summary_stats.csv')
    stat_df  = pd.read_csv(results_dir / 'statistical_tests.csv') \
               if (results_dir / 'statistical_tests.csv').exists() else pd.DataFrame()
    abl_path = results_dir / 'ablation_summary.csv'
    abl_df   = pd.read_csv(abl_path) if abl_path.exists() else pd.DataFrame()
    return eval_df, summary, stat_df, abl_df


def bold(s): return r'\textbf{' + str(s) + r'}'
def underline(s): return r'\underline{' + str(s) + r'}'


# ── Table 1: Main Results ─────────────────────────────────────────────

def build_main_table(summary: pd.DataFrame, stat_df: pd.DataFrame,
                      instances: list) -> str:
    """
    Rows: methods  |  Columns: (instance × metric) where metrics = makespan, gap
    """
    methods_order = ['Random', 'LPT', 'SPT', 'GIN', 'HomoHGT', 'HGT']
    # Exclude ABL_ variants from main table (they appear in Table 2)
    methods_present = [m for m in methods_order
                       if m in summary['method'].values and not m.startswith('ABL_')]

    # Determine best (lowest) mean per instance — only among learning methods
    learning_methods = ['HGT', 'GIN', 'HomoHGT']
    best_means = {}
    for inst in instances:
        inst_sum = summary[(summary['instance'] == inst) &
                           (summary['method'].isin(learning_methods))]
        if not inst_sum.empty:
            best_means[inst] = float(inst_sum['mean'].min())

    # Get significance markers
    sig_map = {}  # (instance, method) -> marker
    if not stat_df.empty:
        for _, row in stat_df.iterrows():
            pval = row['p_value']
            marker = r'$^{**}$' if pval < 0.01 else (r'$^{*}$' if pval < 0.05 else '')
            sig_map[(row['instance'], row['HGT_vs'])] = marker

    # Header
    n_inst = len(instances)
    col_spec = 'l' + ('cc' * n_inst)
    lines = [
        r'\begin{table}[t]',
        r'  \centering',
        r'  \small',
        r'  \caption{%',
        r'    Performance comparison on Fisher-Thompson benchmark instances.',
        r'    Results show mean makespan $\pm$ standard deviation and',
        r'    optimality gap (\%) across 5 independent seeds.',
        r'    $^*$~$p<0.05$, $^{**}$~$p<0.01$ (paired $t$-test vs.',
        r'    HGT-Scheduler).',
        r'  }',
        r'  \label{tab:main_results}',
        f'  \\begin{{tabular}}{{{col_spec}}}',
        r'    \toprule',
    ]

    # Sub-headers
    inst_headers = ' & '.join(
        f'\\multicolumn{{2}}{{c}}{{{INSTANCE_LABELS.get(i, i)}}}' for i in instances)
    lines.append(f'    Method & {inst_headers} \\\\')

    cmidrule_parts = []
    for j, inst in enumerate(instances):
        col_start = 2 + 2 * j
        cmidrule_parts.append(f'\\cmidrule(lr){{{col_start}-{col_start + 1}}}')
    lines.append('    ' + ' '.join(cmidrule_parts))

    metric_headers = ' & '.join(['Makespan', r'Gap~(\%)'] * n_inst)
    lines.append(f'    & {metric_headers} \\\\')
    lines.append(r'    \midrule')

    # Separator rule before HGT
    separator_added = False

    for method in methods_present:
        if method == 'HGT' and not separator_added:
            lines.append(r'    \midrule')
            separator_added = True

        display = METHOD_DISPLAY.get(method, method)
        row_parts = [display]

        for inst in instances:
            opt = OPTIMAL.get(inst)
            row = summary[(summary['method'] == method) &
                           (summary['instance'] == inst)]
            if row.empty:
                row_parts += ['--', '--']
                continue

            mean = float(row['mean'].values[0])
            std  = float(row['std'].values[0])
            is_best = abs(mean - best_means.get(inst, float('inf'))) < 1e-6

            # Significance marker (from sig_map)
            sig = sig_map.get((inst, method), '')

            ms_str = f'{mean:.1f} $\\pm$ {std:.1f}{sig}'
            if is_best:
                ms_str = bold(f'{mean:.1f} $\\pm$ {std:.1f}') + sig

            if opt:
                gap = (mean - opt) / opt * 100
                gap_std = std / opt * 100
                gap_str = f'{gap:.2f} $\\pm$ {gap_std:.2f}'
                if is_best:
                    gap_str = bold(f'{gap:.2f} $\\pm$ {gap_std:.2f}')
            else:
                gap_str = '--'

            row_parts += [ms_str, gap_str]

        lines.append('    ' + ' & '.join(row_parts) + r' \\')

    # Optimal row
    lines.append(r'    \midrule')
    opt_parts = [r'\textit{Optimal}']
    for inst in instances:
        opt = OPTIMAL.get(inst)
        if opt:
            opt_parts += [f'\\textit{{{opt}}}', r'\textit{0.00}']
        else:
            opt_parts += ['--', '--']
    lines.append('    ' + ' & '.join(opt_parts) + r' \\')

    lines += [
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


# ── Table 2: Ablation Study ───────────────────────────────────────────

def build_ablation_table(abl_df: pd.DataFrame) -> str:
    if abl_df.empty:
        return "% Ablation data not available\n"

    instance = 'FT06'
    optimal  = OPTIMAL.get(instance)
    abl_sorted = abl_df.sort_values('mean')

    # Best variant
    best_mean = float(abl_df['mean'].min())

    lines = [
        r'\begin{table}[t]',
        r'  \centering',
        r'  \small',
        r'  \caption{%',
        f'    Ablation study on {INSTANCE_LABELS.get(instance, instance)}.',
        r'    We evaluate different architectural choices.',
        r'    Mean makespan $\pm$ std over 3 seeds (50 eval episodes each).',
        r'    Optimal makespan = ' + str(optimal) + r'.',
        r'  }',
        r'  \label{tab:ablation}',
        r'  \begin{tabular}{lcc}',
        r'    \toprule',
        r'    \textbf{Variant} & \textbf{Makespan} ($\downarrow$) & \textbf{Gap (\%)} \\',
        r'    \midrule',
    ]

    group_headers = {
        'HGT-Full':   '\\multicolumn{3}{l}{\\textit{Full model}} \\\\',
        'HGT-Homo':   '\\multicolumn{3}{l}{\\textit{Edge type ablation}} \\\\',
        'GIN-Base':   None,
        'HGT-1Layer': '\\multicolumn{3}{l}{\\textit{Depth ablation}} \\\\',
        'HGT-2Layer': None,
        'HGT-4Layer': None,
        'HGT-1Head':  '\\multicolumn{3}{l}{\\textit{Attention head ablation}} \\\\',
        'HGT-8Heads': None,
    }

    last_group = None
    for _, row in abl_sorted.iterrows():
        variant = row['variant']
        display = ABLATION_DISPLAY.get(variant, variant)
        mean = float(row['mean'])
        std  = float(row['std'])
        is_best = abs(mean - best_mean) < 1e-6

        mean_str = f'{mean:.1f} $\\pm$ {std:.1f}'
        if is_best:
            mean_str = bold(f'{mean:.1f} $\\pm$ {std:.1f}')

        if optimal:
            gap = (mean - optimal) / optimal * 100
            gap_str = f'{gap:.2f}\\%'
            if is_best:
                gap_str = bold(f'{gap:.2f}\\%')
        else:
            gap_str = '--'

        lines.append(f'    {display} & {mean_str} & {gap_str} \\\\')

    # Add optimal reference
    lines.append(r'    \midrule')
    lines.append(f'    \\textit{{Optimal}} & \\textit{{{optimal}}} & \\textit{{0.00\\%}} \\\\')

    lines += [
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


# ── Table 3: Statistical Tests ────────────────────────────────────────

def build_stat_table(stat_df: pd.DataFrame, instances: list) -> str:
    if stat_df.empty:
        return "% Statistical test data not available\n"

    # Exclude internal ablation variants (ABL_*) — those belong in Table 2
    stat_df = stat_df[~stat_df['HGT_vs'].str.startswith('ABL_')]
    comparisons = stat_df['HGT_vs'].unique()

    lines = [
        r'\begin{table}[t]',
        r'  \centering',
        r'  \small',
        r'  \caption{%',
        r'    Statistical significance of HGT-Scheduler improvements.',
        r'    Paired $t$-test comparing HGT-Scheduler vs.\ each baseline',
        r'    using per-seed mean makespans.',
        r'    $\Delta$: relative improvement; $p$: two-tailed $p$-value.',
        r'  }',
        r'  \label{tab:stat_tests}',
        r'  \begin{tabular}{llrrrl}',
        r'    \toprule',
        r'    \textbf{Instance} & \textbf{Comparison} & '
        r'$\Delta$ (\%) & \textbf{HGT} & \textbf{Baseline} & $p$-value \\',
        r'    \midrule',
    ]

    for inst in instances:
        sub = stat_df[stat_df['instance'] == inst]
        if sub.empty:
            continue
        first = True
        for _, row in sub.iterrows():
            pval = row['p_value']
            sig = r'$^{**}$' if pval < 0.01 else (r'$^{*}$' if pval < 0.05 else '')
            hgt_m = f"{row['HGT_mean']:.1f}"
            oth_m = f"{row['other_mean']:.1f}"
            imp   = f"{row['rel_improvement_pct']:+.2f}"
            pstr  = f'{pval:.4f}{sig}'
            baseline = METHOD_DISPLAY.get(row['HGT_vs'], row['HGT_vs']) \
                .replace(r'~\cite{zhang2020learning}', '') \
                .replace(r'\textbf{', '').replace(r'}', '')

            inst_col = INSTANCE_LABELS.get(inst, inst) if first else ''
            lines.append(f'    {inst_col} & {baseline} & {imp} & {hgt_m} & {oth_m} & {pstr} \\\\')
            first = False
        lines.append(r'    \midrule')

    # Remove last midrule
    if lines[-1] == r'    \midrule':
        lines[-1] = r'    \bottomrule'

    lines += [
        r'  \end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


# ── Table 4: Parameter Count ──────────────────────────────────────────

def build_model_size_table() -> str:
    import torch, sys
    sys.path.insert(0, '.')
    from src.Policy import HGTPolicy
    from src.baselines import GINPolicy, HomoHGTPolicy

    def count(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    hgt   = HGTPolicy()
    gin   = GINPolicy()
    homo  = HomoHGTPolicy()

    lines = [
        r'\begin{table}[t]',
        r'  \centering',
        r'  \small',
        r'  \caption{Model complexity comparison.}',
        r'  \label{tab:model_size}',
        r'  \begin{tabular}{lrr}',
        r'    \toprule',
        r'    \textbf{Model} & \textbf{Parameters} & \textbf{Layers} \\',
        r'    \midrule',
        f'    HGT-Scheduler (ours) & {count(hgt):,} & 3 HGT + 2 MLP \\\\',
        f'    Homo-HGT & {count(homo):,} & 3 HGT + 2 MLP \\\\',
        f'    GIN (L2D-style) & {count(gin):,} & 3 GIN + 2 MLP \\\\',
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


# ── Wrapper: write all tables ─────────────────────────────────────────

def write_table(path: Path, content: str, title: str):
    path.write_text(content)
    print(f"  Saved: {path}  ({len(content.splitlines())} lines)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--output-dir',  default='tables')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading results from {results_dir} ...")
    eval_df, summary, stat_df, abl_df = load_data(results_dir)

    # Only include instances where at least one core learning method has results
    core_methods = ['HGT', 'GIN', 'HomoHGT']
    instances = sorted(
        summary[summary['method'].isin(core_methods)]['instance'].unique().tolist()
    )
    print(f"Instances in data: {instances}")
    print(f"Methods in data:   {sorted(summary['method'].unique().tolist())}")
    print(f"\nGenerating LaTeX tables → {out_dir}\n")

    # Table 1: Main results
    t1 = build_main_table(summary, stat_df, instances)
    write_table(out_dir / 'table1_main_results.tex', t1, 'Main Results')

    # Table 2: Ablation
    t2 = build_ablation_table(abl_df)
    write_table(out_dir / 'table2_ablation.tex', t2, 'Ablation Study')

    # Table 3: Statistical tests
    t3 = build_stat_table(stat_df, instances)
    write_table(out_dir / 'table3_stat_tests.tex', t3, 'Statistical Tests')

    # Table 4: Model size
    t4 = build_model_size_table()
    write_table(out_dir / 'table4_model_size.tex', t4, 'Model Size')

    # Master include file
    master = '\n'.join([
        r'% Auto-generated tables for HGT-Scheduler paper',
        r'% Include these in your main LaTeX file with:',
        r'%   \input{tables/all_tables}',
        r'',
        r'\usepackage{booktabs}   % \toprule, \midrule, \bottomrule',
        r'\usepackage{multirow}   % \multicolumn',
        r'',
        r'% --- Table 1: Main Results ---',
        r'\input{tables/table1_main_results}',
        r'',
        r'% --- Table 2: Ablation Study ---',
        r'\input{tables/table2_ablation}',
        r'',
        r'% --- Table 3: Statistical Significance ---',
        r'\input{tables/table3_stat_tests}',
        r'',
        r'% --- Table 4: Model Complexity ---',
        r'\input{tables/table4_model_size}',
    ])
    (out_dir / 'all_tables.tex').write_text(master)

    print(f"\nAll tables saved to {out_dir.resolve()}")
    print("\n--- Table 1 Preview ---")
    print(t1)
    print("\n--- Table 2 Preview ---")
    print(t2)


if __name__ == '__main__':
    main()
