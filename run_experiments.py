#!/usr/bin/env python3
"""
Comprehensive Experiment Runner for HGT-Scheduler.

Trains and evaluates:
  - HGT-Scheduler (our method)
  - GIN baseline (L2D-style)
  - Homo-HGT baseline (ablation: no edge-type distinction)
  - Random policy
  - Greedy-SPT heuristic

Benchmarks: FT06, FT10, FT20
Seeds:      5 independent runs for statistical significance
Results:    saved to results/ directory as CSV files

Usage:
    python run_experiments.py                      # Full suite
    python run_experiments.py --quick              # FT06 only, 3 seeds
    python run_experiments.py --instance FT06      # Single instance
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import HeteroData, Batch

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.JSSP_Env import JSSPEnv
from src.Policy import HGTPolicy
from src.baselines import GINPolicy, HomoHGTPolicy
from benchmarks.instances import FT06, FT10, FT20

# =====================================================================
# Configuration
# =====================================================================

INSTANCES = {
    'FT06': FT06,
    'FT10': FT10,
    'FT20': FT20,
}

DEFAULT_CONFIG = dict(
    seeds           = [0, 1, 2, 3, 4],
    instances       = ['FT06', 'FT10', 'FT20'],
    timesteps       = {'FT06': 80_000, 'FT10': 200_000, 'FT20': 300_000},
    eval_episodes   = 100,
    eval_freq       = 4_000,    # evaluate every N timesteps
    # model
    hidden_dim      = 128,
    embedding_dim   = 64,
    num_layers      = 3,
    num_heads       = 4,
    dropout         = 0.1,
    # ppo
    lr              = 3e-4,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_eps        = 0.2,
    vf_coef         = 0.5,
    ent_coef        = 0.01,
    max_grad_norm   = 0.5,
    ppo_epochs      = 4,
    batch_size      = 32,
    episodes_per_update = 4,
)

QUICK_CONFIG = dict(
    seeds           = [0, 1, 2],
    instances       = ['FT06', 'FT10'],
    timesteps       = {'FT06': 60_000, 'FT10': 120_000, 'FT20': 200_000},
    eval_episodes   = 50,
    eval_freq       = 3_000,
    hidden_dim      = 128,
    embedding_dim   = 64,
    num_layers      = 3,
    num_heads       = 4,
    dropout         = 0.1,
    lr              = 3e-4,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_eps        = 0.2,
    vf_coef         = 0.5,
    ent_coef        = 0.01,
    max_grad_norm   = 0.5,
    ppo_epochs      = 4,
    batch_size      = 32,
    episodes_per_update = 4,
)


# =====================================================================
# Utilities
# =====================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def make_env(instance_name: str) -> JSSPEnv:
    inst = INSTANCES[instance_name]
    return JSSPEnv(
        num_jobs        = inst['num_jobs'],
        num_machines    = inst['num_machines'],
        processing_times  = inst['processing_times'],
        machine_assignment= inst['machine_assignment'],
        reward_type     = 'dense',
    )


def make_policy(method: str, cfg: dict) -> nn.Module:
    kwargs = dict(
        node_feature_dim = 3,
        hidden_dim       = cfg['hidden_dim'],
        embedding_dim    = cfg['embedding_dim'],
        dropout          = cfg['dropout'],
    )
    if method == 'HGT':
        return HGTPolicy(num_hgt_layers=cfg['num_layers'],
                         num_heads=cfg['num_heads'], **kwargs)
    elif method == 'HomoHGT':
        return HomoHGTPolicy(num_hgt_layers=cfg['num_layers'],
                              num_heads=cfg['num_heads'], **kwargs)
    elif method == 'GIN':
        return GINPolicy(num_gin_layers=cfg['num_layers'], **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


# =====================================================================
# Rollout Buffer
# =====================================================================

class RolloutBuffer:
    def __init__(self):
        self.states:     List[HeteroData] = []
        self.actions:    List[int]        = []
        self.rewards:    List[float]      = []
        self.values:     List[float]      = []
        self.log_probs:  List[float]      = []
        self.dones:      List[bool]       = []
        self.returns:    List[float]      = []
        self.advantages: List[float]      = []

    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state.cpu())
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        rewards   = np.array(self.rewards)
        values    = np.array(self.values + [last_value])
        dones     = np.array(self.dones)
        advantages = np.zeros_like(rewards)
        last_gae  = 0
        for t in reversed(range(len(rewards))):
            next_val = 0 if dones[t] else values[t + 1]
            delta    = rewards[t] + gamma * next_val - values[t]
            last_gae = delta + gamma * gae_lambda * last_gae * (1 - dones[t])
            advantages[t] = last_gae
        self.returns    = (advantages + values[:-1]).tolist()
        self.advantages = advantages.tolist()

    def clear(self): self.__init__()
    def __len__(self):  return len(self.states)


# =====================================================================
# PPO Trainer (works for any actor-critic policy)
# =====================================================================

def ppo_update(policy, optimizer, buffer: RolloutBuffer, cfg: dict, device):
    """Single PPO update step."""
    states        = buffer.states
    actions       = torch.tensor(buffer.actions,    dtype=torch.long,    device=device)
    returns       = torch.tensor(buffer.returns,    dtype=torch.float32, device=device)
    advantages    = torch.tensor(buffer.advantages, dtype=torch.float32, device=device)
    old_log_probs = torch.tensor(buffer.log_probs,  dtype=torch.float32, device=device)
    old_values    = torch.tensor(buffer.values,     dtype=torch.float32, device=device)

    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    epoch_stats = dict(policy_loss=[], value_loss=[], entropy=[], clip_frac=[])

    for _ in range(cfg['ppo_epochs']):
        indices = np.random.permutation(len(states))
        for start in range(0, len(states), cfg['batch_size']):
            idx = indices[start:start + cfg['batch_size']]
            batch_states = Batch.from_data_list([states[i] for i in idx]).to(device)
            b_actions  = actions[idx]
            b_returns  = returns[idx]
            b_adv      = advantages[idx]
            b_old_lp   = old_log_probs[idx]
            b_old_val  = old_values[idx]

            log_probs, values, entropy = policy.evaluate_actions(batch_states, b_actions)
            values = values.squeeze()

            ratio  = torch.exp(log_probs - b_old_lp)
            surr1  = ratio * b_adv
            surr2  = torch.clamp(ratio, 1 - cfg['clip_eps'], 1 + cfg['clip_eps']) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Clipped value loss
            v_clipped = b_old_val + torch.clamp(
                values - b_old_val, -cfg['clip_eps'], cfg['clip_eps'])
            value_loss = 0.5 * torch.max(
                (values - b_returns).pow(2),
                (v_clipped - b_returns).pow(2)).mean()

            entropy_loss = -entropy.mean()
            loss = (policy_loss
                    + cfg['vf_coef'] * value_loss
                    + cfg['ent_coef'] * entropy_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg['max_grad_norm'])
            optimizer.step()

            clip_frac = ((ratio - 1).abs() > cfg['clip_eps']).float().mean().item()
            epoch_stats['policy_loss'].append(policy_loss.item())
            epoch_stats['value_loss'].append(value_loss.item())
            epoch_stats['entropy'].append(-entropy_loss.item())
            epoch_stats['clip_frac'].append(clip_frac)

    return {k: np.mean(v) for k, v in epoch_stats.items()}


# =====================================================================
# Training
# =====================================================================

def train_policy(method: str, instance_name: str, seed: int, cfg: dict,
                 device: torch.device, results_dir: Path) -> Tuple[nn.Module, pd.DataFrame]:
    """
    Train a policy with PPO and return (policy, training_curve).
    """
    set_seed(seed)
    env = make_env(instance_name)
    policy = make_policy(method, cfg).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg['lr'])
    buffer = RolloutBuffer()

    total_steps = cfg['timesteps'][instance_name]
    ep_per_upd  = cfg['episodes_per_update']
    eval_freq   = cfg['eval_freq']
    eval_eps    = cfg['eval_episodes']

    curve_rows = []          # (timestep, mean_makespan)
    best_makespan = float('inf')

    global_step   = 0
    next_eval_at  = eval_freq
    last_ep_steps = 0

    pbar = tqdm(total=total_steps,
                desc=f"{method}/{instance_name}/seed{seed}",
                unit='step', leave=False)

    episode_makespans = deque(maxlen=20)

    while global_step < total_steps:
        # ── collect episodes ─────────────────────────────────────────
        policy.eval()
        for _ in range(ep_per_upd):
            state, _ = env.reset()
            done = False
            while not done:
                state_t = state.to(device)
                with torch.no_grad():
                    action, log_prob, _, value = policy.get_action_and_value(state_t)
                next_state, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated
                buffer.add(state, action.item(), reward,
                           value.item(), log_prob.item(), done)
                state = next_state
                global_step += 1
                pbar.update(1)
            episode_makespans.append(info['makespan'])

        # compute advantages
        with torch.no_grad():
            last_val = policy.get_value(state.to(device)).item()
        buffer.compute_returns_and_advantages(last_val, cfg['gamma'], cfg['gae_lambda'])

        # ── PPO update ───────────────────────────────────────────────
        policy.train()
        ppo_update(policy, optimizer, buffer, cfg, device)
        buffer.clear()

        # ── periodic evaluation ──────────────────────────────────────
        if global_step >= next_eval_at or global_step >= total_steps:
            eval_ms = evaluate_policy(policy, instance_name, eval_eps // 5, device, deterministic=True)
            mean_ms = float(np.mean(eval_ms))
            curve_rows.append({'method': method, 'instance': instance_name,
                                'seed': seed, 'timestep': global_step,
                                'mean_makespan': mean_ms})
            if mean_ms < best_makespan:
                best_makespan = mean_ms
                ckpt_path = results_dir / 'checkpoints' / f"{method}_{instance_name}_seed{seed}_best.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(policy.state_dict(), ckpt_path)
            next_eval_at += eval_freq

    pbar.close()
    curve_df = pd.DataFrame(curve_rows)
    return policy, curve_df


# =====================================================================
# Evaluation helpers
# =====================================================================

def evaluate_policy(policy, instance_name: str, num_episodes: int,
                    device: torch.device, deterministic: bool = True,
                    seed_offset: int = 0) -> List[float]:
    """Run deterministic evaluation. Returns list of makespans."""
    env = make_env(instance_name)
    policy.eval()
    makespans = []
    for ep in range(num_episodes):
        state, _ = env.reset(seed=ep + seed_offset)
        done = False
        while not done:
            with torch.no_grad():
                action, _, _, _ = policy.get_action_and_value(
                    state.to(device), deterministic=deterministic)
            state, _, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
        makespans.append(info['makespan'])
    return makespans


def evaluate_random(instance_name: str, num_episodes: int, seed_offset: int = 0) -> List[float]:
    """Random scheduling baseline."""
    env = make_env(instance_name)
    makespans = []
    for ep in range(num_episodes):
        state, _ = env.reset(seed=ep + seed_offset)
        done = False
        while not done:
            mask = state['op'].action_mask.numpy()
            valid = np.where(mask)[0]
            action = int(np.random.choice(valid))
            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        makespans.append(info['makespan'])
    return makespans


def evaluate_spt(instance_name: str, num_episodes: int, seed_offset: int = 0) -> List[float]:
    """Shortest Processing Time (SPT) heuristic."""
    env = make_env(instance_name)
    makespans = []
    for ep in range(num_episodes):
        state, _ = env.reset(seed=ep + seed_offset)
        done = False
        while not done:
            mask = state['op'].action_mask.numpy()
            valid = np.where(mask)[0]
            # Select operation with shortest processing time
            proc_times = state['op'].x[:, 0].numpy()  # normalized processing times
            spt_action = valid[np.argmin(proc_times[valid])]
            state, _, terminated, truncated, info = env.step(int(spt_action))
            done = terminated or truncated
        makespans.append(info['makespan'])
    return makespans


def evaluate_lpt(instance_name: str, num_episodes: int, seed_offset: int = 0) -> List[float]:
    """Longest Processing Time (LPT) heuristic."""
    env = make_env(instance_name)
    makespans = []
    for ep in range(num_episodes):
        state, _ = env.reset(seed=ep + seed_offset)
        done = False
        while not done:
            mask = state['op'].action_mask.numpy()
            valid = np.where(mask)[0]
            proc_times = state['op'].x[:, 0].numpy()
            lpt_action = valid[np.argmax(proc_times[valid])]
            state, _, terminated, truncated, info = env.step(int(lpt_action))
            done = terminated or truncated
        makespans.append(info['makespan'])
    return makespans


# =====================================================================
# Main experiment loop
# =====================================================================

def run_all_experiments(cfg: dict, results_dir: Path):
    device = get_device()
    print(f"\n{'='*60}")
    print(f"HGT-Scheduler Experiments")
    print(f"Device: {device}")
    print(f"Instances: {cfg['instances']}")
    print(f"Seeds: {cfg['seeds']}")
    print(f"Methods: HGT, GIN, HomoHGT")
    print(f"{'='*60}\n")

    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Storage ───────────────────────────────────────────────────────
    all_eval_rows  = []   # evaluation results
    all_curve_rows = []   # training curves
    trained_policies = {}  # (method, instance, seed) -> policy

    # ── 1. Train learnable methods ─────────────────────────────────────
    learnable_methods = ['HGT', 'GIN', 'HomoHGT']

    for method in learnable_methods:
        for instance_name in cfg['instances']:
            print(f"\n── Training {method} on {instance_name} ──")
            for seed in cfg['seeds']:
                policy, curve_df = train_policy(
                    method, instance_name, seed, cfg, device, results_dir)

                all_curve_rows.append(curve_df)
                trained_policies[(method, instance_name, seed)] = policy

                # Final evaluation
                set_seed(seed + 1000)
                makespans = evaluate_policy(
                    policy, instance_name, cfg['eval_episodes'], device,
                    deterministic=True, seed_offset=seed * 1000)

                inst_data = INSTANCES[instance_name]
                optimal   = inst_data.get('optimal_makespan', None)

                for ep, ms in enumerate(makespans):
                    all_eval_rows.append({
                        'method': method, 'instance': instance_name,
                        'seed': seed, 'episode': ep, 'makespan': ms,
                        'optimal': optimal,
                        'gap': (ms - optimal) / optimal * 100 if optimal else None,
                    })
                mean_ms = np.mean(makespans)
                gap_str = (f"  gap={((mean_ms-optimal)/optimal*100):.2f}%"
                           if optimal else "")
                print(f"  seed {seed}: mean={mean_ms:.2f}{gap_str}")

    # ── 2. Evaluate non-learning baselines ─────────────────────────────
    print("\n── Evaluating non-learning baselines ──")
    for instance_name in cfg['instances']:
        inst_data = INSTANCES[instance_name]
        optimal   = inst_data.get('optimal_makespan', None)

        for method_fn, method_name in [
            (evaluate_random, 'Random'),
            (evaluate_spt,    'SPT'),
            (evaluate_lpt,    'LPT'),
        ]:
            set_seed(999)
            # Run multiple seeds worth of episodes for fair comparison
            all_ms = method_fn(instance_name,
                               cfg['eval_episodes'] * len(cfg['seeds']),
                               seed_offset=0)
            # Split into seed-sized chunks for consistent reporting
            chunk = cfg['eval_episodes']
            for s, seed in enumerate(cfg['seeds']):
                ms_chunk = all_ms[s * chunk: (s + 1) * chunk]
                for ep, ms in enumerate(ms_chunk):
                    all_eval_rows.append({
                        'method': method_name, 'instance': instance_name,
                        'seed': seed, 'episode': ep, 'makespan': ms,
                        'optimal': optimal,
                        'gap': (ms - optimal) / optimal * 100 if optimal else None,
                    })
            mean_ms = np.mean(all_ms)
            gap_str = (f"  gap={((mean_ms-optimal)/optimal*100):.2f}%"
                       if optimal else "")
            print(f"  {method_name}/{instance_name}: mean={mean_ms:.2f}{gap_str}")

    # ── 3. Save results ────────────────────────────────────────────────
    eval_df  = pd.DataFrame(all_eval_rows)
    curve_df = pd.concat(all_curve_rows, ignore_index=True) if all_curve_rows else pd.DataFrame()

    eval_df.to_csv(results_dir / 'eval_results.csv', index=False)
    curve_df.to_csv(results_dir / 'training_curves.csv', index=False)
    print(f"\nResults saved to {results_dir}")

    return eval_df, curve_df, trained_policies


# =====================================================================
# Ablation Study
# =====================================================================

def run_ablation_study(cfg: dict, results_dir: Path):
    """
    Ablation on FT06 (3 seeds for speed).
    Variants: Full HGT, HomoHGT (no edge type), Mean Pool, 1/2/4 layers.
    """
    device = get_device()
    instance_name = 'FT06'
    seeds = cfg['seeds'][:3]
    print(f"\n{'='*60}")
    print(f"Ablation Study on {instance_name}")
    print(f"{'='*60}")

    ablation_variants = {
        'HGT-Full':    lambda c: HGTPolicy(node_feature_dim=3, hidden_dim=c['hidden_dim'],
                                            embedding_dim=c['embedding_dim'],
                                            num_hgt_layers=3, num_heads=4, dropout=c['dropout']),
        'HGT-Homo':    lambda c: HomoHGTPolicy(node_feature_dim=3, hidden_dim=c['hidden_dim'],
                                                embedding_dim=c['embedding_dim'],
                                                num_hgt_layers=3, num_heads=4, dropout=c['dropout']),
        'HGT-1Layer':  lambda c: HGTPolicy(node_feature_dim=3, hidden_dim=c['hidden_dim'],
                                            embedding_dim=c['embedding_dim'],
                                            num_hgt_layers=1, num_heads=4, dropout=c['dropout']),
        'HGT-2Layer':  lambda c: HGTPolicy(node_feature_dim=3, hidden_dim=c['hidden_dim'],
                                            embedding_dim=c['embedding_dim'],
                                            num_hgt_layers=2, num_heads=4, dropout=c['dropout']),
        'HGT-4Layer':  lambda c: HGTPolicy(node_feature_dim=3, hidden_dim=c['hidden_dim'],
                                            embedding_dim=c['embedding_dim'],
                                            num_hgt_layers=4, num_heads=4, dropout=c['dropout']),
        'HGT-1Head':   lambda c: HGTPolicy(node_feature_dim=3, hidden_dim=c['hidden_dim'],
                                            embedding_dim=c['embedding_dim'],
                                            num_hgt_layers=3, num_heads=1, dropout=c['dropout']),
        'HGT-8Heads':  lambda c: HGTPolicy(node_feature_dim=3, hidden_dim=c['hidden_dim'],
                                            embedding_dim=c['embedding_dim'],
                                            num_hgt_layers=3, num_heads=8, dropout=c['dropout']),
    }

    ablation_rows = []
    abl_cfg = dict(cfg)
    abl_cfg['instances'] = [instance_name]
    abl_cfg['timesteps']  = {instance_name: cfg['timesteps'][instance_name]}

    inst_data = INSTANCES[instance_name]
    optimal   = inst_data.get('optimal_makespan', None)

    for variant_name, policy_fn in ablation_variants.items():
        print(f"\n  Variant: {variant_name}")
        for seed in seeds:
            set_seed(seed)
            env = make_env(instance_name)
            policy = policy_fn(cfg).to(device)
            optimizer = optim.Adam(policy.parameters(), lr=cfg['lr'])
            buffer = RolloutBuffer()

            total_steps = abl_cfg['timesteps'][instance_name]
            ep_per_upd  = cfg['episodes_per_update']
            global_step = 0

            while global_step < total_steps:
                policy.eval()
                for _ in range(ep_per_upd):
                    state, _ = env.reset()
                    done = False
                    while not done and global_step < total_steps:
                        state_t = state.to(device)
                        with torch.no_grad():
                            action, log_prob, _, value = policy.get_action_and_value(state_t)
                        next_state, reward, terminated, truncated, info = env.step(action.item())
                        done = terminated or truncated
                        buffer.add(state, action.item(), reward,
                                   value.item(), log_prob.item(), done)
                        state = next_state
                        global_step += 1

                with torch.no_grad():
                    last_val = policy.get_value(state.to(device)).item()
                buffer.compute_returns_and_advantages(last_val, cfg['gamma'], cfg['gae_lambda'])
                policy.train()
                ppo_update(policy, optimizer, buffer, cfg, device)
                buffer.clear()

            # Final evaluation
            makespans = evaluate_policy(
                policy, instance_name, cfg['eval_episodes'] // 2, device,
                deterministic=True, seed_offset=seed * 500)

            for ep, ms in enumerate(makespans):
                ablation_rows.append({
                    'variant': variant_name, 'instance': instance_name,
                    'seed': seed, 'episode': ep, 'makespan': ms,
                    'optimal': optimal,
                    'gap': (ms - optimal) / optimal * 100 if optimal else None,
                })
            mean_ms = np.mean(makespans)
            gap_str = f"  gap={((mean_ms-optimal)/optimal*100):.2f}%" if optimal else ""
            print(f"    seed {seed}: mean={mean_ms:.2f}{gap_str}")

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(results_dir / 'ablation_results.csv', index=False)
    print(f"\nAblation results saved to {results_dir / 'ablation_results.csv'}")
    return ablation_df


# =====================================================================
# Statistical Tests
# =====================================================================

def compute_summary_stats(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Per-seed mean makespans, then aggregate across seeds."""
    # Per-seed mean makespan
    per_seed = (eval_df
                .groupby(['method', 'instance', 'seed'])['makespan']
                .mean()
                .reset_index()
                .rename(columns={'makespan': 'seed_mean'}))

    # Aggregate across seeds
    summary = (per_seed
               .groupby(['method', 'instance'])['seed_mean']
               .agg(['mean', 'std', 'min', 'max', 'count'])
               .reset_index())
    summary.columns = ['method', 'instance', 'mean', 'std', 'min', 'max', 'n_seeds']

    # Add optimality gap
    optimal_map = {k: v.get('optimal_makespan') for k, v in INSTANCES.items()}
    summary['optimal'] = summary['instance'].map(optimal_map)
    summary['gap_mean'] = ((summary['mean'] - summary['optimal']) /
                            summary['optimal'] * 100).where(summary['optimal'].notna())
    summary['gap_std']  = ((summary['std'] / summary['optimal']) * 100
                            ).where(summary['optimal'].notna())
    return summary


def run_statistical_tests(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Paired t-tests: HGT vs each other method, per instance."""
    test_rows = []
    for instance in eval_df['instance'].unique():
        inst_df = eval_df[eval_df['instance'] == instance]
        hgt_per_seed = (inst_df[inst_df['method'] == 'HGT']
                         .groupby('seed')['makespan'].mean())

        for method in inst_df['method'].unique():
            if method == 'HGT':
                continue
            other_per_seed = (inst_df[inst_df['method'] == method]
                               .groupby('seed')['makespan'].mean())
            # Align seeds
            common_seeds = sorted(set(hgt_per_seed.index) & set(other_per_seed.index))
            if len(common_seeds) < 2:
                continue
            hgt_vals   = hgt_per_seed.loc[common_seeds].values
            other_vals = other_per_seed.loc[common_seeds].values

            t_stat, p_val = stats.ttest_rel(hgt_vals, other_vals)
            improvement = float(np.mean(other_vals) - np.mean(hgt_vals))
            rel_improvement = improvement / np.mean(other_vals) * 100

            test_rows.append({
                'instance': instance, 'HGT_vs': method,
                'HGT_mean': float(np.mean(hgt_vals)),
                'other_mean': float(np.mean(other_vals)),
                'improvement': improvement,
                'rel_improvement_pct': rel_improvement,
                't_stat': t_stat, 'p_value': p_val,
                'significant': p_val < 0.05,
                'n': len(common_seeds),
            })

    return pd.DataFrame(test_rows)


# =====================================================================
# Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='HGT-Scheduler Experiments')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: FT06+FT10, 3 seeds, fewer steps')
    parser.add_argument('--instance', type=str, default=None,
                        help='Single instance to run (FT06/FT10/FT20)')
    parser.add_argument('--no-ablation', action='store_true',
                        help='Skip ablation study')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory for results (default: results/)')
    args = parser.parse_args()

    cfg = QUICK_CONFIG if args.quick else DEFAULT_CONFIG
    if args.instance:
        cfg = dict(cfg)
        cfg['instances'] = [args.instance]

    results_dir = Path(args.results_dir)

    t_start = time.time()

    # ── Main experiments ──────────────────────────────────────────────
    eval_df, curve_df, trained_policies = run_all_experiments(cfg, results_dir)

    # ── Summary statistics ────────────────────────────────────────────
    summary = compute_summary_stats(eval_df)
    summary.to_csv(results_dir / 'summary_stats.csv', index=False)

    # ── Statistical tests ─────────────────────────────────────────────
    stat_df = run_statistical_tests(eval_df)
    stat_df.to_csv(results_dir / 'statistical_tests.csv', index=False)

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for instance in cfg['instances']:
        print(f"\n{instance} (optimal={INSTANCES[instance].get('optimal_makespan','?')}):")
        inst_sum = summary[summary['instance'] == instance].sort_values('mean')
        for _, row in inst_sum.iterrows():
            gap_str = f"  gap={row['gap_mean']:.2f}%±{row['gap_std']:.2f}%" if pd.notna(row.get('gap_mean')) else ""
            print(f"  {row['method']:12s}: {row['mean']:.2f}±{row['std']:.2f}{gap_str}")

    print(f"\n{'='*60}")
    print("STATISTICAL TESTS (HGT vs others, paired t-test)")
    print(f"{'='*60}")
    if len(stat_df) > 0:
        for _, row in stat_df.iterrows():
            sig_marker = "**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else "  ")
            print(f"  {row['instance']} HGT vs {row['HGT_vs']:10s}: "
                  f"improvement={row['rel_improvement_pct']:+.2f}%  "
                  f"p={row['p_value']:.4f} {sig_marker}")

    # ── Ablation study ────────────────────────────────────────────────
    if not args.no_ablation:
        ablation_df = run_ablation_study(cfg, results_dir)
        abl_summary = (ablation_df
                       .groupby(['variant', 'instance', 'seed'])['makespan']
                       .mean()
                       .reset_index()
                       .groupby(['variant', 'instance'])['makespan']
                       .agg(['mean', 'std'])
                       .reset_index())
        abl_summary.to_csv(results_dir / 'ablation_summary.csv', index=False)

        print(f"\n{'='*60}")
        print("ABLATION STUDY (FT06)")
        print(f"{'='*60}")
        optimal = INSTANCES['FT06'].get('optimal_makespan', None)
        for _, row in abl_summary.sort_values('mean').iterrows():
            gap = f"  gap={((row['mean']-optimal)/optimal*100):.2f}%" if optimal else ""
            print(f"  {row['variant']:15s}: {row['mean']:.2f}±{row['std']:.2f}{gap}")

    elapsed = (time.time() - t_start) / 60
    print(f"\nTotal time: {elapsed:.1f} minutes")
    print(f"Results in: {results_dir.resolve()}")


if __name__ == '__main__':
    main()
