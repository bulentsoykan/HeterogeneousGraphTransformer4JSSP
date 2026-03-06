# HGT-Scheduler: Heterogeneous Graph Transformer for Job Shop Scheduling

**Paper:** *Exploiting Edge Semantics in Job Shop Scheduling Problem with Heterogeneous Graph Transformers*

**Author:** Dr. Bulent Soykan
**Affiliation:** Institute for Simulation and Training, University of Central Florida
3100 Technology Pkwy, Orlando, FL 32826 USA
**Contact:** Bulent.Soykan@ucf.edu

---

A reinforcement learning scheduler for the Job Shop Scheduling Problem (JSSP) that uses a Heterogeneous Graph Transformer (HGT) to exploit the two distinct edge semantics of the disjunctive graph representation.

## Overview

JSSP is modeled as a disjunctive graph where operations are nodes connected by two edge types:
- **`precedes`** edges: job-flow precedence constraints (directed)
- **`competes`** edges: machine-sharing contention (undirected)

**HGT-Scheduler** applies type-dependent attention across these edge types, allowing the policy to learn that precedence constraints and machine contention carry structurally different information. A PPO agent selects which ready operation to schedule at each decision step.

## Results

All experiments use PPO (50K timesteps, lr=3×10⁻⁴, γ=0.99) evaluated on Fisher-Thompson benchmark instances. Results are mean ± std over 5 independent seeds (50 evaluation episodes each).

### Main Results

| Method | FT06 Makespan ↓ | FT06 Gap (%) | FT10 Makespan ↓ | FT10 Gap (%) |
|---|---|---|---|---|
| Random | 96.1 ± 2.7 | 74.74 | 1832.8 ± 17.4 | 97.07 |
| SPT | 109.0 ± 0.0 | 98.18 | 2648.0 ± 0.0 | 184.73 |
| LPT | 129.0 ± 0.0 | 134.55 | 2940.0 ± 0.0 | 216.13 |
| GIN (L2D-style) | 66.6 ± 8.8 | 21.09 | 1994.8 ± 361.5 | 114.49 |
| Homo-HGT (ablation) | 66.0 ± 2.6 | 20.00 | **1540.6 ± 163.0** | **65.66** |
| **HGT-Scheduler (ours)** | **59.6 ± 1.3** | **8.36** | 1594.2 ± 281.7 | 71.42 |
| *Optimal* | *55* | *0.00* | *930* | *0.00* |

**FT06:** HGT-Scheduler achieves 8.4% optimality gap, significantly better than Homo-HGT (p=0.011) and all heuristics (p<0.01).

**FT10:** Both HGT variants clearly outperform GIN. The HGT-Scheduler vs Homo-HGT difference (1594 vs 1541) is not statistically significant (p=0.775) at 50K training steps — edge-type awareness may require longer training to manifest on larger instances.

### Ablation: Depth on FT06 (3 seeds)

| Variant | Makespan | Gap (%) |
|---|---|---|
| **HGT-Full / 3-Layer (ours)** | **59.6 ± 1.3** | **8.36** |
| HGT-2Layer | 60.3 ± 4.2 | 9.70 |
| HGT-4Layer | 60.7 ± 0.6 | 10.30 |
| HGT-1Layer | 62.0 ± 1.0 | 12.73 |
| Homo-HGT (no edge types) | 66.0 ± 2.6 | 20.00 |
| GIN (L2D-style) | 66.6 ± 8.8 | 21.09 |
| *Optimal* | *55* | *0.00* |

3-layer HGT is significantly better than 1-layer (p=0.020).

### Model Complexity

| Model | Parameters | Architecture |
|---|---|---|
| HGT-Scheduler | 319,198 | 3 HGT + 2 MLP |
| Homo-HGT | 294,610 | 3 HGT + 2 MLP |
| GIN (L2D-style) | 271,174 | 3 GIN + 2 MLP |

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** Requires NumPy <2.0 for PyTorch 2.1.x compatibility (`numpy>=1.24.0,<2.0.0` in requirements.txt).

## Usage

### Training

```bash
# Train a single method on one or more instances
python train_resume.py --method HGT --instances FT06 FT10 --seeds 0 1 2 3 4

# Available methods: HGT, GIN, HomoHGT
# Available instances: FT06, FT10
```

The script skips already-completed seeds (checks for `results/eval_{method}_{instance}_seed{seed}.csv`), making it safe to restart.

### Ablation Study

```bash
python run_ablation.py
```

Trains HGT-1Layer, HGT-2Layer, and HGT-4Layer variants (3 seeds each) on FT06.

### Generating Figures and Tables

```bash
# Consolidate per-seed CSVs into summary files
python consolidate_results.py

# Generate all figures (saved to figures/)
python generate_figures.py

# Generate all LaTeX tables (saved to tables/)
python generate_tables.py
```

## Project Structure

```
.
├── src/
│   ├── Policy.py          # HGTPolicy (319K params, 3-layer HGT + PPO heads)
│   ├── baselines.py       # HomoHGTPolicy, GINPolicy
│   ├── JSSP_Env.py        # JSSP Gymnasium environment (disjunctive graph)
│   └── PPO.py             # PPO trainer with GAE and action masking
├── benchmarks/
│   └── instances.py       # FT06, FT10, FT20 instance definitions
├── results/               # Training outputs: .pt checkpoints, per-seed CSVs
│   ├── eval_results.csv       # Consolidated evaluation results
│   ├── summary_stats.csv      # Mean ± std per method/instance
│   ├── statistical_tests.csv  # Paired t-test results
│   ├── training_curves.csv    # Learning curves
│   └── ablation_summary.csv   # Ablation study summary
├── figures/               # Generated figures (PDF + PNG)
├── tables/                # Generated LaTeX tables
├── train_resume.py        # Main training script (resume-safe)
├── run_ablation.py        # Depth ablation training
├── consolidate_results.py # Result aggregation
├── generate_figures.py    # Figure generation
├── generate_tables.py     # LaTeX table generation
└── requirements.txt
```

## Architecture

**HGT-Scheduler** builds on `HGTConv` (PyTorch Geometric) with:
- **Node type:** single `op` type (operations)
- **Edge types:** `('op', 'precedes', 'op')` and `('op', 'competes', 'op')`
- **Encoder:** 3 × HGTConv(hidden=128, heads=4) with type-dependent projection matrices
- **Pooling:** global attention pooling over operation embeddings for graph-level value estimate
- **Head:** 2-layer MLP → logits (policy) and scalar (value)
- **Training:** PPO with clip=0.2, 4 epochs/update, batch=32, γ=0.99, λ=0.95

## Key Finding

Type-aware attention over `precedes` vs `competes` edges provides a statistically significant improvement on FT06 (p=0.011 vs Homo-HGT, which uses identical architecture but merges edge types). This validates the core hypothesis that distinguishing edge semantics in the disjunctive graph aids scheduling policy learning.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{soykan2026hgt_jssp,
  title   = {Exploiting Edge Semantics in Job Shop Scheduling Problem with Heterogeneous Graph Transformers},
  author  = {Soykan, Bulent},
  year    = {2026},
  institution = {Institute for Simulation and Training, University of Central Florida},
  address = {3100 Technology Pkwy, Orlando, FL 32826 USA},
}
```

## License

See [LICENSE](LICENSE).
