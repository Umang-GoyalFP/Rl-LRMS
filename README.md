# CATPO: Critique-Augmented Tree Policy Optimization

Diagnoses uninformative trees via a fitness score, heals failing branches with step-level critique, and weights the training loss by tree informativeness. Built on top of the [TreeRPO](https://arxiv.org/abs/2506.05183) codebase.

## Installation
```bash
conda create -n rllm python=3.10 -y
conda activate rllm
pip install -e ./verl
pip install -e .
```

### Done

- **Tree Fitness Score** — $F(T) = \mathcal{H}(\mathbf{r}) \cdot (1 - \rho_{\pi,r})$  
  Measures how much a tree will improve the policy. Binary entropy of leaf rewards × policy surprise (Pearson correlation between node log-probs and propagated rewards). Range $[0, 2]$, zero extra compute.

- **Fitness utilities** (`evals/fitness_utils.py`)  
  Standalone implementation of `compute_tree_fitness`, `binary_entropy`, `pearson_correlation`, `classify_tree` (dead-correct / dead-wrong / stale / informative), tree walking helpers, and serialization. Unit tested.

- **H1 validation script** (`evals/h1_fitness_predicts_gradient.py`)  
  Offline experiment: build N-ary trees via HuggingFace generation, compute F(T) per tree, compute per-tree gradient norms via forward+backward, report Pearson/Spearman correlation between F and $\|g\|^2$.  
  Supports `--skip_gradient` for fast fitness-only analysis.

- **H2 validation script** (`evals/h2_critique_heals_fitness.py`)  
  Offline experiment: identify Dead-Wrong trees ($F \approx 0$, all leaves wrong), generate step-level critique at shallowest failing depth, graft $k$ refined continuations, measure $\Delta F = F_\text{after} - F_\text{before}$.  
  Supports separate `--critic_model` or self-critique.

- **CATPO framework design** (theoretical)  
  Full 4-step algorithm: tree construction → fitness diagnosis → step-level critique healing → fitness-weighted PPO loss. Dual objective $\mathcal{J}_\text{init} + \lambda \cdot \mathcal{J}_\text{healed}$ with Critique-GRPO shaping ratio $\rho_t = \pi_\theta / (\pi_\theta + \gamma)$ for grafted nodes.

### TODO

- [ ] **Run H1 experiment** — Validate that F(T) actually correlates with gradient norms on Qwen2.5-Math-1.5B. Target: Spearman $r > 0.4$, $p < 0.01$.
- [ ] **Run H2 experiment** — Validate that critique raises F of Dead-Wrong trees. Target: mean $\Delta F > 0$ with $p < 0.05$ (one-sided t-test).
- [ ] **Integrate fitness into training loop** — Add `compute_tree_fitness` call after `_traversal_reward` / `_traversal_adv` in `ray_trainer.py`. Log F per tree to wandb.
- [ ] **Implement fitness-weighted loss** — Replace uniform batch weighting with $w(T) = F(\tilde{T}) / \sum F$ in the actor update step.
- [ ] **Implement critique healing in the training loop** — For Dead-Wrong trees during training: extract failing prefix, call critic (self or external), generate refinements via vLLM, graft onto tree, re-propagate rewards.
- [ ] **Implement shaped ratio for healed nodes** — Apply Critique-GRPO's $\rho_t = \pi_\theta / (\pi_\theta + \gamma)$ instead of standard importance ratio for grafted nodes in the PPO objective.
- [ ] **Tune thresholds** — Determine $\tau_\text{low}$, $\tau_\text{high}$ for tree classification, $\gamma$ for shaping ratio, $\lambda$ for healed loss weight, $k$ for number of refinements.
- [ ] **Ablation experiments** — (a) fitness weighting only (no critique), (b) critique only (no fitness weighting), (c) full CATPO, vs. vanilla TreeRPO baseline.
- [ ] **Scaling evaluation** — Test on Qwen2.5-Math-7B and harder benchmarks (AIME, OlympiadBench).
- [ ] **Write paper** — Formalize proofs, run full experiments, produce tables/figures.