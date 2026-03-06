#!/usr/bin/env python3
"""
Ablation study for HGT-Scheduler on FT06 (3 seeds, 50K steps).

Variants:
  HGT-1Layer : 1-layer HGT, 4 heads
  HGT-2Layer : 2-layer HGT, 4 heads
  HGT-4Layer : 4-layer HGT, 4 heads

HGT-Full (3-layer), HomoHGT and GIN are reused from main experiments.

Outputs:
  results/eval_ABL_{variant}_FT06_seed{seed}.csv
  results/curve_ABL_{variant}_FT06_seed{seed}.csv
  results/ablation_summary.csv

Usage:
    python run_ablation.py [--seeds 0 1 2] [--results-dir results]
"""

import argparse, sys, time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.Policy import HGTPolicy
from benchmarks.instances import FT06, FT10
from run_experiments import (RolloutBuffer, ppo_update, evaluate_policy,
                              set_seed, make_env)

INSTANCES = {'FT06': FT06, 'FT10': FT10}

# Ablation variants: name → HGTPolicy kwargs override
ABLATION_VARIANTS = {
    'HGT-1Layer': dict(num_hgt_layers=1, num_heads=4),
    'HGT-2Layer': dict(num_hgt_layers=2, num_heads=4),
    'HGT-4Layer': dict(num_hgt_layers=4, num_heads=4),
}

CFG = dict(
    timesteps={'FT06': 50_000},
    eval_episodes=50,
    eval_freq=8_000,
    hidden_dim=128, embedding_dim=64, num_layers=3, num_heads=4, dropout=0.1,
    lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
    vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
    ppo_epochs=4, batch_size=32, episodes_per_update=4,
)


def train_variant(variant_name, variant_kwargs, seed, results_dir, cfg):
    """Train and evaluate one ablation variant on FT06."""
    instance_name = 'FT06'
    eval_path  = results_dir / f'eval_ABL_{variant_name}_{instance_name}_seed{seed}.csv'
    curve_path = results_dir / f'curve_ABL_{variant_name}_{instance_name}_seed{seed}.csv'

    if eval_path.exists():
        df = pd.read_csv(eval_path)
        mean_ms = df['makespan'].mean()
        opt = INSTANCES[instance_name].get('optimal_makespan', 55)
        gap = (mean_ms - opt) / opt * 100
        print(f'  ↩ SKIP {variant_name}/seed{seed}: already done '
              f'mean={mean_ms:.1f}  gap={gap:.2f}%', flush=True)
        return mean_ms

    set_seed(seed)
    device = torch.device('cpu')
    env    = make_env(instance_name)

    # Build policy with ablation kwargs
    policy = HGTPolicy(**variant_kwargs).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg['lr'])
    buffer = RolloutBuffer()

    total_steps = cfg['timesteps'][instance_name]
    ep_per_upd  = cfg['episodes_per_update']
    eval_freq   = cfg['eval_freq']

    curve_rows  = []
    global_step = 0
    next_eval   = eval_freq
    best_ms     = float('inf')
    last_state  = None

    t0 = time.time()
    while global_step < total_steps:
        policy.eval()
        for _ in range(ep_per_upd):
            state, _ = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    action, log_prob, _, value = policy.get_action_and_value(
                        state.to(device))
                next_state, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated
                buffer.add(state, action.item(), reward, value.item(), log_prob.item(), done)
                state = next_state
                global_step += 1
            last_state = state

        with torch.no_grad():
            lv = policy.get_value(last_state.to(device)).item()
        buffer.compute_returns_and_advantages(lv, cfg['gamma'], cfg['gae_lambda'])
        policy.train()
        ppo_update(policy, optimizer, buffer, cfg, device)
        buffer.clear()

        if global_step >= next_eval or global_step >= total_steps:
            ms = evaluate_policy(policy, instance_name, 10, device,
                                 deterministic=True, seed_offset=7777)
            mmean = float(np.mean(ms))
            curve_rows.append({
                'method': f'ABL_{variant_name}',
                'instance': instance_name,
                'seed': seed,
                'timestep': global_step,
                'mean_makespan': mmean,
            })
            if mmean < best_ms:
                best_ms = mmean
                torch.save(policy.state_dict(),
                           results_dir / f'ABL_{variant_name}_seed{seed}_best.pt')
            next_eval += eval_freq

    # Load best checkpoint and final evaluation
    best_ckpt = results_dir / f'ABL_{variant_name}_seed{seed}_best.pt'
    if best_ckpt.exists():
        policy.load_state_dict(torch.load(best_ckpt, map_location='cpu'))

    final_ms = evaluate_policy(policy, instance_name, cfg['eval_episodes'],
                               device, deterministic=True,
                               seed_offset=seed * 1000)
    mean_ms  = float(np.mean(final_ms))
    opt      = INSTANCES[instance_name].get('optimal_makespan', 55)
    gap      = (mean_ms - opt) / opt * 100
    elapsed  = (time.time() - t0) / 60

    print(f'  ✓ {variant_name}/seed{seed}: '
          f'mean={mean_ms:.1f}±{np.std(final_ms):.1f}  gap={gap:.2f}%  '
          f'({elapsed:.1f} min)', flush=True)

    # Save CSVs
    pd.DataFrame(curve_rows).to_csv(curve_path, index=False)
    rows = [{'method': f'ABL_{variant_name}', 'instance': instance_name,
             'seed': seed, 'episode': i, 'makespan': ms}
            for i, ms in enumerate(final_ms)]
    pd.DataFrame(rows).to_csv(eval_path, index=False)

    return mean_ms


def build_ablation_summary(variant_names, seeds, results_dir):
    """Aggregate per-variant mean±std, also pulling main experiment results."""
    rows = []

    # Ablation variants
    for variant in variant_names:
        seed_means = []
        for seed in seeds:
            path = results_dir / f'eval_ABL_{variant}_FT06_seed{seed}.csv'
            if path.exists():
                df = pd.read_csv(path)
                seed_means.append(df['makespan'].mean())
        if seed_means:
            rows.append({
                'variant': variant,
                'n_seeds': len(seed_means),
                'mean': float(np.mean(seed_means)),
                'std':  float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
            })

    # HGT-Full from main experiments (5 seeds available)
    hgt_means = []
    for seed in range(5):
        path = results_dir / f'eval_HGT_FT06_seed{seed}.csv'
        if path.exists():
            hgt_means.append(pd.read_csv(path)['makespan'].mean())
    if hgt_means:
        rows.append({'variant': 'HGT-Full', 'n_seeds': len(hgt_means),
                     'mean': float(np.mean(hgt_means)),
                     'std':  float(np.std(hgt_means, ddof=1))})

    # HomoHGT from main experiments (5 seeds)
    homo_means = []
    for seed in range(5):
        path = results_dir / f'eval_HomoHGT_FT06_seed{seed}.csv'
        if path.exists():
            homo_means.append(pd.read_csv(path)['makespan'].mean())
    if homo_means:
        rows.append({'variant': 'HGT-Homo', 'n_seeds': len(homo_means),
                     'mean': float(np.mean(homo_means)),
                     'std':  float(np.std(homo_means, ddof=1))})

    # GIN from main experiments (5 seeds)
    gin_means = []
    for seed in range(5):
        path = results_dir / f'eval_GIN_FT06_seed{seed}.csv'
        if path.exists():
            gin_means.append(pd.read_csv(path)['makespan'].mean())
    if gin_means:
        rows.append({'variant': 'GIN-Base', 'n_seeds': len(gin_means),
                     'mean': float(np.mean(gin_means)),
                     'std':  float(np.std(gin_means, ddof=1))})

    df = pd.DataFrame(rows).sort_values('mean')
    path = results_dir / 'ablation_summary.csv'
    df.to_csv(path, index=False)
    print(f'\nAblation summary → {path}')
    print(df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--variants', nargs='+',
                        default=list(ABLATION_VARIANTS.keys()))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    variants = {k: v for k, v in ABLATION_VARIANTS.items() if k in args.variants}
    seeds = args.seeds

    print(f'\n=== Ablation Study on FT06: variants={list(variants)}, seeds={seeds} ===\n')

    for variant_name, variant_kwargs in variants.items():
        print(f'\n  -- Variant: {variant_name} --')
        for seed in seeds:
            train_variant(variant_name, variant_kwargs, seed, results_dir, CFG)

    build_ablation_summary(list(variants.keys()), seeds, results_dir)
    print('\n=== Ablation DONE ===\n')


if __name__ == '__main__':
    main()
