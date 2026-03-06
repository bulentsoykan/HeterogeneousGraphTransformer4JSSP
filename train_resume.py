#!/usr/bin/env python3
"""
Resume training - runs only missing (method, instance, seed) combinations.
Auto-detects which seeds are already done and skips them.

Usage:
    python train_resume.py --method HGT     --instances FT06 FT10 --seeds 3 4 0 1 2
    python train_resume.py --method GIN     --instances FT06 FT10 --seeds 2 3 4 0 1 2
    python train_resume.py --method HomoHGT --instances FT06      --seeds 1 2 3 4
"""
import argparse, sys, time, json
import numpy as np
import pandas as pd
import torch, torch.optim as optim
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.JSSP_Env import JSSPEnv
from src.Policy import HGTPolicy
from src.baselines import GINPolicy, HomoHGTPolicy
from benchmarks.instances import FT06, FT10, FT20
from run_experiments import (RolloutBuffer, ppo_update,
                              evaluate_policy, make_policy, set_seed, make_env)

INSTANCES = {'FT06': FT06, 'FT10': FT10, 'FT20': FT20}

CFG = dict(
    timesteps={'FT06': 50_000, 'FT10': 50_000},
    eval_episodes=50,
    eval_freq=8_000,
    hidden_dim=128, embedding_dim=64, num_layers=3, num_heads=4, dropout=0.1,
    lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
    vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
    ppo_epochs=4, batch_size=32, episodes_per_update=4,
)


def train_and_eval(method, instance_name, seed, cfg, out_dir):
    eval_path = out_dir / f'eval_{method}_{instance_name}_seed{seed}.csv'
    if eval_path.exists():
        df = pd.read_csv(eval_path)
        opt = INSTANCES[instance_name].get('optimal_makespan')
        mean = df['makespan'].mean()
        gap_str = f"  gap={(mean-opt)/opt*100:.2f}%" if opt else ""
        print(f"  ↩ SKIP {method}/{instance_name}/seed{seed}: already done "
              f"mean={mean:.1f}{gap_str}", flush=True)
        return

    set_seed(seed)
    device = torch.device('cpu')
    env    = make_env(instance_name)
    policy = make_policy(method, cfg).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg['lr'])
    buffer = RolloutBuffer()

    total_steps = cfg['timesteps'][instance_name]
    ep_per_upd  = cfg['episodes_per_update']
    eval_freq   = cfg['eval_freq']
    eval_eps    = 10

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
            ms = evaluate_policy(policy, instance_name, eval_eps, device,
                                 deterministic=True, seed_offset=7777)
            mmean = float(np.mean(ms))
            curve_rows.append({'method': method, 'instance': instance_name,
                                'seed': seed, 'timestep': global_step,
                                'mean_makespan': mmean})
            if mmean < best_ms:
                best_ms = mmean
                torch.save(policy.state_dict(),
                           out_dir / f'{method}_{instance_name}_seed{seed}_best.pt')
            next_eval += eval_freq

    best_ckpt = out_dir / f'{method}_{instance_name}_seed{seed}_best.pt'
    if best_ckpt.exists():
        policy.load_state_dict(torch.load(best_ckpt, map_location='cpu'))

    final_ms = evaluate_policy(policy, instance_name, cfg['eval_episodes'], device,
                               deterministic=True, seed_offset=seed * 1000)

    elapsed = (time.time() - t0) / 60
    opt = INSTANCES[instance_name].get('optimal_makespan')
    gap_str = f"  gap={(np.mean(final_ms)-opt)/opt*100:.2f}%" if opt else ""
    print(f"  ✓ {method:10s}/{instance_name}/seed{seed}: "
          f"mean={np.mean(final_ms):.1f}±{np.std(final_ms):.1f}{gap_str}  "
          f"({elapsed:.1f} min)", flush=True)

    pd.DataFrame(curve_rows).to_csv(
        out_dir / f'curve_{method}_{instance_name}_seed{seed}.csv', index=False)
    pd.DataFrame([{'method': method, 'instance': instance_name, 'seed': seed,
                    'episode': ep, 'makespan': ms_val}
                   for ep, ms_val in enumerate(final_ms)]).to_csv(
        eval_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--instances', nargs='+', default=['FT06'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--results-dir', default='results')
    args = parser.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {args.method}: instances={args.instances} seeds={args.seeds} ===",
          flush=True)
    t_total = time.time()

    for instance in args.instances:
        print(f"\n  Instance: {instance}", flush=True)
        for seed in args.seeds:
            train_and_eval(args.method, instance, seed, CFG, out_dir)

    print(f"\n=== {args.method} DONE in {(time.time()-t_total)/60:.1f} min ===",
          flush=True)


if __name__ == '__main__':
    main()
