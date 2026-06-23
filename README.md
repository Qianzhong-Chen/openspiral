# OpenSpiral

**SPIRAL** (**S**elf-**P**olicy **I**mprovement via **R**eward-**A**ligned **L**earning) is an on-policy,
real-robot self-improvement framework that turns dense progress rewards into a self-improving robot
data flywheel for vision-language-action (VLA) policies.

This package implements the **SPIRAL** half of the paper
[*SARM2: Multi-Task Stage Aware Reward Modeling for Self Improving Robotic Manipulation*](https://qianzhong-chen.github.io/sarm.github.io/). It is built directly on top of [openpi](https://github.com/Physical-Intelligence/openpi)
by [Physical Intelligence](https://www.physicalintelligence.company/) — the π₀ / π₀.₅ flow-matching VLAs
serve as the frozen base policy that SPIRAL refines. The companion **SARM2** reward model (which produces
the dense rewards consumed here) lives in the parent [`sarm`](../) repository.


---

## Overview

Fine-tuning VLA policies for long-horizon manipulation still relies heavily on behavior cloning, which
requires costly high-quality demonstrations and keeps the policy pinned near the demonstration
distribution. SPIRAL breaks that ceiling: it takes a frozen, behavior-cloned π₀/π₀.₅ policy and trains a
**lightweight residual policy** that is optimized to maximize a learned value function, using **dense,
stage-aware rewards** from SARM2 over cheap autonomous rollouts — no new human demonstrations, no
backpropagation through the VLA's flow sampler, and no environment simulator.

Concretely, at every observation:

```
a_final = π_VLA(s, z)        +        s_θ(s, z)
          └── frozen base ──┘         └── learned residual ──┘
          (π₀ / π₀.₅, no grad)        (3×1024 MLP, ~MB-scale)
```

SPIRAL is a **residual reinforcement-learning** substrate built on
[DICE-RL](https://arxiv.org/abs/2603.10263) with two modifications that make it long-horizon-ready:

1. **Dense per-step rewards** from the SARM2 reward model replace sparse terminal rewards in the TD3
   bootstrap, giving informative credit assignment across multi-minute tasks.
2. **A hybrid critic target** that blends a dense-reward **TD3** objective with a sparse-episode
   **Monte-Carlo (MC)** objective, combining low-bias long-horizon return estimates with stable
   bootstrapped value learning.

Wrapped in an autonomous *rollout → reward-label → residual-RL update* loop, this turns a
sparse-reward residual-RL recipe into a self-improving data flywheel. 
### How it works (one training step)

For a batch of transitions `(s, a, s', r_dense, done, mc_return)`:

1. **Frozen π₀ forward (precompute, no grad).** Extract pooled SigLIP visual features and sample
   `κ` base action chunks `a^(j) = π_VLA(s, z_j)` from the flow prior (one latent `z_j` per sample).
   These are run in a *separate* JIT from the RL update to keep the compiled artifact small.
2. **Critic update.** A 5-network Q-ensemble is regressed onto a hybrid target:
   `α · MC_return + (1−α) · [R_chunk + γ^H · (1−done) · min_i Q_target,i(s', a'+s_θ(s'))]`,
   with target-policy smoothing and target networks as in standard TD3.
3. **Residual actor update.** The residual MLP `s_θ` is trained to maximize the ensemble-min Q of the
   edited action, averaged over the `κ` latent draws, with a BC regularizer `β‖s_θ(s,z)‖²` that keeps
   the residual close to the base policy:
   `min_θ E[ −Q_φ(s, a_base + s_θ(s,z)) + β‖s_θ(s,z)‖² ]`.
4. **Polyak update** of the target critic.

At deployment we apply **best-of-κ** action selection: draw `κ` candidates and execute the one with the
highest Q (`residual_rl_model.best_of_n_actions`), or simply use the single residual-edited action.

See `src/openpi/models/residual_rl_model.py` (model + losses),
`scripts/train_residual_rl.py` (training loop), and
`src/openpi/training/rl_data_loader.py` (reward/return construction) for the implementation.

<!-- ---

## Requirements

SPIRAL training is far lighter than full VLA fine-tuning — the trainable residual actor + critic
ensemble are small MLPs. The dominant cost is the **frozen** π₀ forward pass during precompute.

| Mode                              | Memory Required | Example GPU        |
| --------------------------------- | --------------- | ------------------ |
| Inference (π₀ + residual)         | > 8 GB          | RTX 4090           |
| Residual RL training              | > 24 GB         | RTX 4090 / L4      |
| Base π₀/π₀.₅ fine-tuning (full)   | > 70 GB         | A100 (80GB) / H100 |

Tested on Ubuntu 22.04. A SARM2-labeled LeRobot dataset (with a per-frame `reward` column, and
optionally a `progress` column) is required to train the residual policy — see
[Step 1](#step-1-convert-and-reward-label-data). -->

---

## Installation

This package uses [uv](https://docs.astral.sh/uv/) for dependency management, inheriting openpi's
environment. Make sure submodules are pulled.

```bash
cd openpi_spiral
curl -LsSf https://astral.sh/uv/install.sh | sh

# GIT_LFS_SKIP_SMUDGE=1 is required to pull LeRobot as a dependency
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

---

## End-to-End Pipeline

SPIRAL follows the four-stage self-improvement loop of Algorithm 1 in the paper. The minimal path to a
self-improved policy is:

```
Demos ──▶ (1) BC fine-tune π₀/π₀.₅ ──▶ (2) SARM2 reward-label data
                                              │
                                              ▼
                       (3) Residual RL (SPIRAL) ──▶ (4) Rollout ─┐
                                   ▲                             │
                                   └───── relabel with SARM2 ◀───┘
```

### Step 1: Convert and reward-label data

Convert your raw [YAM](https://i2rt.com/products/yam-manipulator) (or other) teleoperation data into the
LeRobot format **with a dense `reward` column** produced by the SARM2 reward model.

Open `scripts/yam_data/convert_yam_data_dense_reward.py` and set:
- `yam_data_path` — path(s) to your raw episodes
- `repo_name` — output LeRobot dataset repo id
- `language_instruction` — the task prompt (e.g. `"clean the whiteboard"`)

```bash
python scripts/yam_data/convert_yam_data_dense_reward.py
```

The dense `reward` (and optional `progress`) columns come from SARM2's MMoE goal-distance value head; see
the parent [`sarm`](../) repo for reward-model training and labeling. SPIRAL itself reads these columns
directly — no RL signals are stored beyond them; rewards, dones, and MC returns are derived on the fly
from episode boundaries (`src/openpi/training/rl_data_loader.py`).

### Step 2: Compute normalization statistics

Add your own `TrainConfig` to `src/openpi/training/config.py`, then compute norm stats:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py \
    --config-name <your_training_configuration> --epsilon 1e-2
```

Faster, video-free variants (recommended for large datasets):

```bash
python scripts/compute_norm_stats_video_free.py
python scripts/compute_norm_stats_delta_action_video_free.py
```

### Step 3: Train the base π₀ / π₀.₅ policy (behavior cloning)

This is standard openpi BC fine-tuning — it produces the frozen base policy `π₁` that SPIRAL refines.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    <your_training_configuration> --exp-name=<task_name> --overwrite
```

### Step 4: Train the residual RL (SPIRAL) policy

Point a residual-RL config (see [presets](#residual-rl-config-presets)) at the frozen base checkpoint and
the reward-labeled dataset:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python scripts/train_residual_rl.py \
    <your_residualRL_training_configuration> --exp_name <task_name>
```

This loads the frozen π₀, initializes the residual actor + critic ensemble, and runs offline
actor-critic training on the LeRobot dataset. Checkpoints (actor/critic params + `rl_config.json`) are
written to `<checkpoint_dir>/<config_name>/<exp_name>/step_xxxxxxxx/`.

### Step 5: Inference

Serve the fused frozen π₀ + trained residual actor:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run python scripts/serve_policy.py \
    policy:residual-rl-checkpoint \
    --policy.config <your_residualRL_training_configuration> \
    --policy.rl-dir   <PATH_TO_RL_CKPT/step_xxxxxxxx> \
    --policy.pi0-dir  <PATH_TO_Pi_CKPT/params>
```

A standalone inference smoke test is available at `scripts/test_residual_rl_inference.py`.

### Step 6: Autonomous self-improvement loop

To close the flywheel, alternate **rollout collection → SARM2 relabeling → SPIRAL update** (Algorithm 1,
stages 2b–4). Each round initializes the residual policy from the checkpoint that produced the rollouts,
relabels the new rollouts with the (one-time-adapted) reward model `RM₂`, and runs another SPIRAL update —
requiring no further human labels. In the paper, success improves monotonically over ~3 rounds.

---

## Residual RL config presets

Presets are defined in `_make_rl_configs()` in `src/openpi/training/config.py` and selected by name on
the CLI. They differ only in the **critic target** mode (`critic_target` / `mc_weight`), which is the core
ablation axis of the paper:

| Config name                                | `critic_target` | Description                                                            |
| ------------------------------------------ | --------------- | --------------------------------------------------------------------- |
| `residual_rl_clean_whiteboard_spiral`      | `blend`         | **SPIRAL (ours)** — hybrid TD3 (dense) + MC (sparse), `mc_weight=0.5` |
| `residual_rl_clean_whiteboard_dense_only`  | `td`            | Dense-reward TD3 bootstrap only                                       |
| `residual_rl_clean_whiteboard_sparse`      | `mc`            | Sparse Monte-Carlo terminal reward only                               |

Set `pi0_checkpoint` (path to the frozen base `params`) and `pi0_config_name` (the matching base
`TrainConfig`) in your preset before launching. Add your own presets following the same pattern for new
tasks.

Critic-target modes (`src/openpi/models/residual_rl_model.py::critic_loss`):
- **`td`** — chunk-level TD bootstrap: `R_chunk + γ^H · (1−done) · Q_target`, where `R_chunk` is the
  discounted sum of dense rewards over the action chunk and `H` is the action horizon.
- **`mc`** — pure Monte-Carlo return from the sparse end-of-episode reward.
- **`blend`** — `mc_weight · MC + (1−mc_weight) · TD` (this is SPIRAL).

---

## Key hyperparameters

Defaults reflecting the paper (Table 3) and the `*_spiral` preset:

| Symbol      | Meaning                                                   | Value      |
| ----------- | -------------------------------------------------------- | ---------- |
| `γ`         | Discount factor                                          | `0.9995`   |
| `α`         | MC objective weight in the critic target (`mc_weight`)   | `0.5`      |
| `β`         | BC regularizer weight on the residual (`beta_bc`)        | `30`       |
| `N_critic`  | Critic ensemble size (`num_critics`)                     | `5`        |
| `κ`         | Latent candidates per state / best-of-κ at deploy (`num_noise_samples`) | `4` |
| `τ_tgt`     | Target-network soft-update (Polyak) rate (`target_tau`)  | `0.005`    |
| —           | Actor/critic MLP shape (depth × width)                   | `3 × 1024` |
| —           | Residual output scale (`actor_output_scale`)             | `0.1`      |
| `lr`        | Learning rate (actor / critic)                           | `1e-4`     |
| `B`         | Batch size                                               | `16`       |
| `T`         | Training steps per round (`num_train_steps`)             | `10,000`   |
| —           | Optimizer                                                | AdamW      |

The paper additionally uses target-policy smoothing (noise std `σ=0.2`, clip `c=0.5`) in the TD3 target.

---

## Repository layout

```
openpi_spiral/
├── scripts/
│   ├── train_residual_rl.py            # SPIRAL training loop (two-JIT actor-critic)
│   ├── serve_policy.py                 # serves frozen π₀ + residual actor (policy:residual-rl-checkpoint)
│   ├── test_residual_rl_inference.py   # inference smoke test
│   ├── train.py                        # base π₀/π₀.₅ BC fine-tuning (upstream openpi)
│   ├── compute_norm_stats*.py          # normalization statistics
│   └── yam_data/
│       └── convert_yam_data_dense_reward.py   # YAM → LeRobot with SARM2 dense rewards
└── src/openpi/
    ├── models/residual_rl_model.py     # ResidualActor, CriticEnsemble, critic/actor losses, best-of-N
    ├── policies/residual_rl_policy.py  # fused π₀ + actor inference module
    └── training/
        ├── rl_data_loader.py           # RL transitions: chunk rewards, dones, MC returns from episodes
        └── config.py                   # TrainConfig + ResidualRLTrainConfig presets
```

---

## Limitations & TODO

Known limitations and planned improvements. Contributions are welcome.

- [ ] **Single-GPU training only.** The current residual-RL training script
  (`scripts/train_residual_rl.py`) runs on a single GPU — there is no data or model parallelism
  (no `pmap`/`shard_map`, no FSDP), and multi-node training is not supported. The frozen π₀ forward
  pass dominates per-step memory, so very large base models may not fit on smaller GPUs. Multi-GPU /
  sharded training is a planned improvement.

---

## Citation

If you use SPIRAL or SARM2, please cite:

```bibtex
@article{chen2026sarm2,
  title={SARM2: Multi-Task Stage Aware Reward Modeling for Self Improving Robotic Manipulation},
  author={Chen, Qianzhong and Zheng, Hau and Yu, Justin and Huang, Suning and Sun, Jiankai and Goldberg, Ken and Wen, Chuan and Abbeel, Pieter and Shentu, Yide and Wu, Philipp and others},
  journal={arXiv preprint arXiv:2606.10305},
  year={2026}
}
```

---

## Acknowledgements

OpenSpiral is built on top of [**openpi**](https://github.com/Physical-Intelligence/openpi) by
[Physical Intelligence](https://www.physicalintelligence.company/), and uses their **π₀** and **π₀.₅**
vision-language-action models as the frozen base policy. We are grateful to the openpi team for
open-sourcing the models and training infrastructure that make this work possible. Please also see and
cite the openpi project and the π₀ / π₀.₅ papers when using this code. The original openpi documentation
is preserved in [`openpi_README.md`](openpi_README.md).

The residual-RL substrate follows [DICE-RL](https://arxiv.org/abs/2603.10263) (Sun & Song), and the
actor-critic update follows [TD3](https://arxiv.org/abs/1802.09477) (Fujimoto et al.).

## License

This project inherits the license of the upstream openpi repository; see [`LICENSE`](LICENSE).
