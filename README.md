# CATPO: Critique-Augmented Tree Policy Optimization

Diagnoses uninformative trees via a fitness score, heals failing branches with step-level critique, and weights the training loss by tree informativeness. Built on top of the [TreeRPO](https://arxiv.org/abs/2506.05183) codebase.

## Installation

```bash
conda create -n rllm python=3.10 -y
conda activate rllm
pip install -e ./verl
pip install -e .
```

## Quick Start

### TreeRPO baseline (no CATPO)

```bash
bash scripts/deepscaler/train/qwen2.5-math-1.5b-base_math_1k_bs128_n8_s384_qwen25mathcot.sh
```

### CATPO: Fitness weighting only (no healing)

```bash
bash scripts/deepscaler/train/qwen2.5-math-1.5b-base_math_1k_bs128_n8_s384_qwen25mathcot.sh \
    +trainer.enable_catpo=True \
    trainer.project_name='CATPO' \
    trainer.experiment_name='catpo-fitness-only-1.5b'
```

### CATPO: Full (fitness weighting + critique healing)

```bash
bash scripts/deepscaler/train/qwen2.5-math-1.5b-base_math_1k_bs128_n8_s384_qwen25mathcot.sh \
    +trainer.enable_catpo=True \
    +trainer.enable_catpo_healing=True \
    +trainer.catpo_num_refinements=4 \
    trainer.project_name='CATPO' \
    trainer.experiment_name='catpo-full-1.5b'
```

## CATPO Config Flags

All CATPO flags are **off by default** — the code behaves identically to vanilla TreeRPO unless explicitly enabled.

| Flag | Default | Description |
|------|---------|-------------|
| `trainer.enable_catpo` | `False` | Master switch. Enables fitness computation and fitness-weighted advantages. |
| `trainer.enable_catpo_healing` | `False` | Enable critique-guided healing of dead-wrong trees. Requires `enable_catpo=True`. |
| `trainer.catpo_num_refinements` | `4` | Number of refined continuations to graft per dead-wrong tree. |

### TreeRPO / shared hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `algorithm.adv_estimator` | `grpo` | Advantage estimator. Must be `grpo` for TreeRPO/CATPO (no critic). |
| `actor_rollout_ref.rollout.n` | `8` | Branching factor at depth 1 (children per root). |
| `actor_rollout_ref.rollout.step_length` | `384` | Tokens generated per tree depth. |
| `data.max_response_length` | `1152` | Max total response length. `max_tree_depth = max_response_length / step_length`. |
| `data.max_prompt_length` | `512` | Max prompt length in tokens. |
| `data.train_batch_size` | `128` | Number of prompts per training step. |
| `actor_rollout_ref.rollout.temperature` | `0.6` | Sampling temperature for tree generation. |
| `actor_rollout_ref.actor.optim.lr` | `1e-6` | Learning rate. |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.001` | KL divergence penalty coefficient (beta). |
| `algorithm.kl_ctrl.kl_coef` | `0.001` | KL controller coefficient. |
| `trainer.total_epochs` | `30` | Number of training epochs. |
| `trainer.test_freq` | `20` | Validate every N steps. |
| `trainer.save_freq` | `20` | Save checkpoint every N steps. |

### Internal fitness thresholds (hardcoded in `evals/fitness_utils.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `tau_low` | `0.2` | Trees with F <= tau_low are "dead" (correct or wrong). |
| `tau_high` | `0.8` | Trees with F > tau_high are "informative". |
| `epsilon` | `0.05` | Healing threshold: children with E_reward < epsilon are considered failing. |

## Ablation Experiments

Run these three variants to disentangle the contributions of fitness weighting and critique healing:

```bash
# (a) TreeRPO baseline
bash scripts/deepscaler/train/qwen2.5-math-1.5b-base_math_1k_bs128_n8_s384_qwen25mathcot.sh \
    trainer.experiment_name='ablation-treerpo-baseline'

# (b) Fitness weighting only
bash scripts/deepscaler/train/qwen2.5-math-1.5b-base_math_1k_bs128_n8_s384_qwen25mathcot.sh \
    +trainer.enable_catpo=True \
    trainer.experiment_name='ablation-fitness-only'

# (c) Critique healing only (uniform weighting)
# NOTE: requires code change to skip _apply_fitness_weights when enable_catpo_healing=True but enable_catpo=False
# For now, use full CATPO and compare.

# (d) Full CATPO
bash scripts/deepscaler/train/qwen2.5-math-1.5b-base_math_1k_bs128_n8_s384_qwen25mathcot.sh \
    +trainer.enable_catpo=True \
    +trainer.enable_catpo_healing=True \
    +trainer.catpo_num_refinements=4 \
    trainer.experiment_name='ablation-catpo-full'
```

## Offline Hypothesis Validation

### H1: Fitness predicts gradient utility

```bash
python evals/h1_fitness_predicts_gradient.py \
    --model Qwen/Qwen2.5-Math-1.5B \
    --data ./data_qwen25_math_cot/train/math_cot_train.parquet \
    --num_problems 50 \
    --branching_factor 8 \
    --max_depth 3 \
    --output evals/results/h1_results.json
```

Target: Spearman r > 0.4, p < 0.01.

### H2: Critique heals dead-wrong trees

```bash
python evals/h2_critique_heals_fitness.py \
    --model Qwen/Qwen2.5-Math-1.5B \
    --data ./data_qwen25_math_cot/train/math_cot_train.parquet \
    --num_problems 50 \
    --num_refinements 4 \
    --output evals/results/h2_results.json
```

Target: mean delta-F > 0, p < 0.05 (one-sided t-test).

## WandB Metrics

When `enable_catpo=True`, the following metrics are logged per training step:

| Metric | Description |
|--------|-------------|
| `fitness/mean_F` | Mean tree fitness across batch |
| `fitness/mean_H` | Mean reward entropy (leaf diversity) |
| `fitness/mean_rho` | Mean policy-reward correlation |
| `fitness/dead_correct_pct` | Fraction of trees where all leaves are correct |
| `fitness/dead_wrong_pct` | Fraction of trees where all leaves are wrong |
| `fitness/stale_pct` | Fraction of stale trees (policy already learned) |
| `fitness/informative_pct` | Fraction of informative trees |
| `fitness/healed_count` | Number of dead-wrong trees healed this step |
| `fitness/post_heal_mean_F` | Mean fitness after healing (if healing enabled) |
| `fitness/post_heal_dead_wrong_pct` | Dead-wrong fraction after healing |

## How CATPO Works

### Tree Fitness Score

$$F(T) = \mathcal{H}(\mathbf{r}) \cdot (1 - \rho_{\pi,r})$$

| Component | What it measures |
|-----------|-----------------|
| H(r) | Binary entropy of leaf rewards. 0 when all leaves agree, 1 when half succeed and half fail. |
| rho | Pearson correlation between node log-probabilities and propagated rewards. High rho means the policy already knows which paths are good. |
| F(T) | Product of both. Range [0, 2]. High F = diverse outcomes + policy surprise = informative tree. |

### Tree Regimes

| Regime | Condition | Meaning | CATPO Action |
|--------|-----------|---------|--------------|
| Dead-correct | F <= 0.2, mostly correct | Model already solves this | Downweight |
| Dead-wrong | F <= 0.2, mostly wrong | No positive signal exists | Heal via critique |
| Stale | 0.2 < F <= 0.8 | Policy partially learned this | Moderate weight |
| Informative | F > 0.8 | High learning potential | Full weight |

### Training Pipeline (4 phases per step)

1. **Tree Construction** — Sample N-ary tree of depth D from the LLM policy (standard TreeRPO).
2. **Fitness Diagnosis** — Compute F(T) for each tree, classify into regimes.
3. **Critique Healing** — For dead-wrong trees: find shallowest failure, generate critique, graft k refined branches, re-propagate rewards.
4. **Fitness-Weighted Update** — Scale advantages by normalized fitness, apply standard PPO clipped loss. Only the LLM policy is updated (no critic).