#!/usr/bin/env python3
"""
H1 Validation: Does F(T) predict gradient utility?

Hypothesis: The fitness score F(T) = H(r) * (1 - rho_{pi,r}) correlates
positively with the squared gradient norm ||g(T)||^2 that tree T would
produce during a PPO/GRPO training step.

Protocol:
  1. Load a pretrained/checkpoint model and tokenizer
  2. Load math problems from a parquet dataset
  3. For each problem, build an N-ary tree of depth D via step-level sampling
  4. Score leaves with the rule-based verifier
  5. Propagate rewards bottom-up (as in TreeRPO)
  6. Compute F(T) per tree  (H and rho)
  7. Compute per-tree gradient norms via a single forward+backward pass
  8. Report Pearson/Spearman correlation between F and ||g||^2

Usage:
  python evals/h1_fitness_predicts_gradient.py \
      --model Qwen/Qwen2.5-Math-1.5B \
      --data ./data_qwen25_math_cot/train/math_cot_train.parquet \
      --num_problems 50 \
      --branching_factor 8 \
      --max_depth 3 \
      --step_length 384 \
      --temperature 0.6 \
      --output evals/results/h1_results.json
"""

import argparse
import json
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import wandb
import sys
import math
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM


# Add project root to path so we can import rllm
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
torch.set_float32_matmul_precision('high')


from rllm.rewards.rl_reward import rllm_reward_fn
from evals.fitness_utils import (
    binary_entropy,
    pearson_correlation,
    compute_tree_fitness,
    classify_tree,
    fitness_to_row,
)


# ---------------------------------------------------------------------------
# Lightweight tree node (standalone, no verl dependency)
# ---------------------------------------------------------------------------

class EvalTreeNode:
    """Minimal tree node for offline evaluation."""

    def __init__(self, token_ids: List[int], depth: int, reward: float = 0.0):
        self.token_ids = token_ids      # full sequence: prompt + steps so far
        self.depth = depth
        self.E_reward = reward          # leaf reward or propagated mean
        self.children: List['EvalTreeNode'] = []
        self.child_rewards: List[float] = []
        self.log_prob: Optional[float] = None  # sum log-prob of this step
        self.step_token_ids: Optional[List[int]] = None  # just this step's tokens

    @property
    def is_leaf(self):
        return len(self.children) == 0


# ---------------------------------------------------------------------------
# Tree construction via HF generate
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_children(
    model,
    tokenizer,
    parent: EvalTreeNode,
    n_children: int,
    step_length: int,
    max_total_length: int,
    temperature: float,
    device: torch.device,
) -> List[EvalTreeNode]:
    """Generate n_children continuations of `parent` up to `step_length` new tokens."""

    input_ids = torch.tensor([parent.token_ids], device=device)
    attention_mask = torch.ones_like(input_ids)

    # Don't exceed total length budget
    remaining = max_total_length - len(parent.token_ids)
    gen_length = min(step_length, remaining)
    if gen_length <= 0:
        return []

    outputs = model.generate(
        input_ids=input_ids.repeat(n_children, 1),
        attention_mask=attention_mask.repeat(n_children, 1),
        max_new_tokens=gen_length,
        temperature=temperature,
        do_sample=True,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    children = []
    for j in range(n_children):
        full_ids = outputs[j].tolist()
        step_ids = full_ids[len(parent.token_ids):]
        child = EvalTreeNode(token_ids=full_ids, depth=parent.depth + 1)
        child.step_token_ids = step_ids
        if tokenizer.eos_token_id in step_ids:
            eos_pos = step_ids.index(tokenizer.eos_token_id)
            child.token_ids = parent.token_ids + step_ids[:eos_pos + 1]
            child.step_token_ids = step_ids[:eos_pos + 1]
        children.append(child)

    return children


def build_tree(
    model,
    tokenizer,
    prompt_ids: List[int],
    branching_factor: int,
    max_depth: int,
    step_length: int,
    max_total_length: int,
    temperature: float,
    device: torch.device,
) -> EvalTreeNode:
    """Build an N-ary tree of depth D rooted at the prompt."""
    root = EvalTreeNode(token_ids=prompt_ids, depth=0)
    frontier = [root]

    for d in range(max_depth):
        next_frontier = []
        for node in frontier:
            # Check if this node already ended (contains EOS)
            if (node.depth > 0 and
                    tokenizer.eos_token_id in (node.step_token_ids or [])):
                continue  # leaf, don't expand further

            children = generate_children(
                model, tokenizer, node,
                n_children=branching_factor,
                step_length=step_length,
                max_total_length=max_total_length,
                temperature=temperature,
                device=device,
            )
            node.children = children
            next_frontier.extend(children)

        frontier = next_frontier
        if not frontier:
            break

    return root


# ---------------------------------------------------------------------------
# Reward scoring
# ---------------------------------------------------------------------------

def score_leaves(root: EvalTreeNode, tokenizer, data_source: str,
                 ground_truth: str, ignore_think: bool = True):
    """Score every leaf via the rule-based verifier and set .E_reward."""
    leaves = []

    def _collect(node):
        if node.is_leaf:
            leaves.append(node)
        for c in node.children:
            _collect(c)

    _collect(root)

    for leaf in leaves:
        text = tokenizer.decode(leaf.token_ids, skip_special_tokens=False)
        score = rllm_reward_fn(
            data_source=data_source,
            llm_solution=text,
            ground_truth=ground_truth,
            ignore_think=ignore_think,
        )
        leaf.E_reward = float(score)


def propagate_rewards(node: EvalTreeNode) -> float:
    """Bottom-up reward propagation: E_reward = mean of children's E_rewards."""
    if node.is_leaf:
        return node.E_reward

    node.child_rewards = []
    total = 0.0
    for child in node.children:
        r = propagate_rewards(child)
        node.child_rewards.append(r)
        total += r
    node.E_reward = total / len(node.children) if node.children else 0.0
    return node.E_reward


# ---------------------------------------------------------------------------
# Log-prob computation (per-step)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_step_log_probs(
    model,
    root: EvalTreeNode,
    device: torch.device,
):
    """Compute sum-log-prob for each non-root node's step tokens."""

    # Collect all non-root nodes first
    all_nodes = []
    def _collect(node):
        if node.depth > 0 and node.step_token_ids and len(node.step_token_ids) > 0:
            all_nodes.append(node)
        for c in node.children:
            _collect(c)
    _collect(root)

    # Process in batches of 8
    BATCH = 8
    for i in range(0, len(all_nodes), BATCH):
        batch_nodes = all_nodes[i:i+BATCH]
        for node in batch_nodes:  # still per-node but with cache cleared
            input_ids = torch.tensor([node.token_ids], device=device)
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            step_start = len(node.token_ids) - len(node.step_token_ids)
            log_probs_all = torch.log_softmax(logits[0], dim=-1)
            step_log_prob = 0.0
            for i, tok_id in enumerate(node.step_token_ids):
                pos = step_start + i
                if pos == 0:
                    continue
                step_log_prob += log_probs_all[pos - 1, tok_id].item()
            node.log_prob = step_log_prob
            del outputs, logits, log_probs_all
        torch.cuda.empty_cache()


def collect_node_stats(root: EvalTreeNode) -> Tuple[np.ndarray, np.ndarray]:
    """Collect (log_prob, E_reward) for all non-root nodes with log_probs."""
    lps, rews = [], []

    def _collect(node):
        if node.depth > 0 and node.log_prob is not None:
            lps.append(node.log_prob)
            rews.append(node.E_reward)
        for c in node.children:
            _collect(c)

    _collect(root)
    return np.array(lps), np.array(rews)


# ---------------------------------------------------------------------------
# Gradient norm computation
# ---------------------------------------------------------------------------

def compute_tree_gradient_norm(
    model,
    root: EvalTreeNode,
    tokenizer,
    device: torch.device,
) -> float:
    """Compute ||g(T)||^2 — the squared gradient norm from a pseudo-PPO step on tree T.

    We do a simplified REINFORCE-style gradient:
      g(T) = sum_v A_v * grad log pi(s_v | prefix_v)

    where A_v is the advantage (reward - sibling mean) / (bernoulli_std + eps).
    We accumulate gradients across all non-pruned nodes, then measure ||g||^2.
    """
    model.zero_grad()

    # Collect all (node, advantage) pairs
    node_adv_pairs = []
    _collect_node_advantages(root, node_adv_pairs)

    if not node_adv_pairs:
        return 0.0

    # Accumulate gradients
    

   
    losses = []
    for node, advantage in node_adv_pairs:
        if not node.step_token_ids or len(node.step_token_ids) == 0:
            continue
        input_ids = torch.tensor([node.token_ids], device=device)
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        step_start = len(node.token_ids) - len(node.step_token_ids)
        log_probs_all = torch.log_softmax(logits[0], dim=-1)
        step_log_prob = torch.tensor(0.0, device=device)
        for i, tok_id in enumerate(node.step_token_ids):
            pos = step_start + i
            if pos == 0:
                continue
            step_log_prob = step_log_prob + log_probs_all[pos - 1, tok_id]
        losses.append(-advantage * step_log_prob)

    if losses:
        total_loss = torch.stack(losses).sum()
        total_loss.backward()
        grad_norm_sq = sum(p.grad.data.pow(2).sum().item()
                      for p in model.parameters() if p.grad is not None)
        model.zero_grad()
        return grad_norm_sq
    return 0.0




def _collect_node_advantages(node: EvalTreeNode, pairs: list):
    """Walk tree and compute TreeRPO-style advantages for non-root nodes."""
    if node.depth > 0 and hasattr(node, '_bro_rewards') and node._bro_rewards:
        mean_r = np.mean(node._bro_rewards)
        bernoulli_std = mean_r * (1.0 - mean_r)
        adv = (node.E_reward - mean_r) / (bernoulli_std + 1e-6)
        # Apply pruning: skip if reward spread is too small
        if max(node._bro_rewards) - min(node._bro_rewards) >= 0.1:
            pairs.append((node, adv))

    for child in node.children:
        _collect_node_advantages(child, pairs)


def assign_sibling_rewards(root: EvalTreeNode):
    """Post-order: for each node, set _bro_rewards = parent's child_rewards."""
    def _assign(node):
        for child in node.children:
            child._bro_rewards = list(node.child_rewards)
            _assign(child)
    _assign(root)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_problems(parquet_path: str, num_problems: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Load problems from training parquet."""
    df = pd.read_parquet(parquet_path)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(df), size=min(num_problems, len(df)), replace=False)

    problems = []
    for idx in indices:
        row = df.iloc[idx]
        # Extract ground truth from reward_model column
        rm = row.get('reward_model', {})
        if isinstance(rm, str):
            rm = json.loads(rm)
        ground_truth = rm.get('ground_truth', '')

        # Extract prompt text
        prompt = row.get('prompt', [])
        if isinstance(prompt, str):
            prompt = json.loads(prompt)

        data_source = row.get('data_source', 'math')

        problems.append({
            'prompt': prompt,
            'ground_truth': ground_truth,
            'data_source': data_source,
            'index': int(idx),
        })

    return problems


def format_prompt(prompt_messages: list, tokenizer) -> List[int]:
    """Convert chat messages to token ids using the tokenizer's chat template."""
    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback: concatenate message contents
        text = '\n'.join(m.get('content', '') for m in prompt_messages)

    return tokenizer.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_h1_experiment(args):
    print(f"=== H1: Fitness Predicts Gradient Utility ===")
    print(f"Model:            {args.model}")
    print(f"Data:             {args.data}")
    print(f"Num problems:     {args.num_problems}")
    print(f"Branching factor: {args.branching_factor}")
    print(f"Max depth:        {args.max_depth}")
    print(f"Step length:      {args.step_length}")
    print(f"Temperature:      {args.temperature}")
    print()
    wandb.init(
    project="TreeRPO-H1",
    name=f"h1-{args.model.split('/')[-1]}-b{args.branching_factor}-d{args.max_depth}",
    config={
        "model": args.model,
        "data": args.data,
        "num_problems": args.num_problems,
        "branching_factor": args.branching_factor,
        "max_depth": args.max_depth,
        "step_length": args.step_length,
        "max_prompt_length": args.max_prompt_length,
        "temperature": args.temperature,
        "seed": args.seed,
        }
    )

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Use float16 for generation, float32 for gradient computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,  # fixed deprecated arg too
        trust_remote_code=True,
        device_map="auto",
    )
    
    model.eval()
    model.gradient_checkpointing_enable()

    max_prompt_length = args.max_prompt_length
    max_response_length = args.step_length * args.max_depth
    max_total_length = max_prompt_length + max_response_length

    # Load problems
    print(f"Loading {args.num_problems} problems from {args.data}...")
    problems = load_problems(args.data, args.num_problems, seed=args.seed)
    print(f"Loaded {len(problems)} problems.\n")

    # Results accumulator
    results = []

    for i, problem in enumerate(problems):
        t0 = time.time()
        print(f"[{i+1}/{len(problems)}] Building tree for problem {problem['index']}...")

        # Tokenize prompt
        prompt_ids = format_prompt(problem['prompt'], tokenizer)
        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[:max_prompt_length]

        # --- Phase 1: Build tree via sampling ---
        tree_root = build_tree(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            branching_factor=args.branching_factor,
            max_depth=args.max_depth,
            step_length=args.step_length,
            max_total_length=max_total_length,
            temperature=args.temperature,
            device=device,
        )

        # --- Phase 2: Score leaves ---
        score_leaves(
            tree_root, tokenizer,
            data_source=problem['data_source'],
            ground_truth=problem['ground_truth'],
            ignore_think=True,
        )

        # --- Phase 3: Propagate rewards ---
        propagate_rewards(tree_root)
        assign_sibling_rewards(tree_root)

        # Collect leaf rewards
        leaf_rewards = []
        def _get_leaves(n):
            if n.is_leaf:
                leaf_rewards.append(n.E_reward)
            for c in n.children:
                _get_leaves(c)
        _get_leaves(tree_root)

        # --- Phase 4: Compute step log-probs ---
        compute_step_log_probs(model, tree_root, device)
        node_lps, node_rews = collect_node_stats(tree_root)

        # --- Phase 5: Compute F(T) ---
        fitness = compute_tree_fitness(leaf_rewards, node_lps, node_rews)
        regime = classify_tree(fitness)

        print(f"  p_hat={fitness['p_hat']:.3f}  H={fitness['H']:.3f}  "
              f"rho={fitness['rho']:.3f}  F={fitness['F']:.3f}  regime={regime}")

        # --- Phase 6: Compute gradient norm ---
        if args.skip_gradient:
            grad_norm_sq = float('nan')
            print(f"  (gradient computation skipped)")
        else:
            model.train()  # need gradients
            torch.cuda.empty_cache()
            model = model.float() 
            grad_norm_sq = compute_tree_gradient_norm(model, tree_root, tokenizer, device)
            model = model.half()
            model.eval()
            print(f"  ||g||^2 = {grad_norm_sq:.6e}")
        wandb.log({
            "problem_idx": problem['index'],
            "problem_num": i + 1,
            "p_hat": fitness['p_hat'],
            "H": fitness['H'],
            "rho": fitness['rho'],
            "F": fitness['F'],
            "regime": regime,
            "grad_norm_sq": grad_norm_sq if not math.isinf(grad_norm_sq) else None,
            "elapsed_s": time.time() - t0,
            "data_source": problem['data_source'],
        })
            

        elapsed = time.time() - t0
        print(f"  time: {elapsed:.1f}s\n")

        row = fitness_to_row(i, fitness, regime, extra={
            'grad_norm_sq': grad_norm_sq,
            'problem_idx': problem['index'],
            'data_source': problem['data_source'],
            'elapsed_s': elapsed,
        })
        results.append(row)

        # Free memory
        del tree_root
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Analysis ---
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    df = pd.DataFrame(results)

    # Filter out NaN gradient norms
    valid = df.dropna(subset=['grad_norm_sq'])
    valid = valid[valid['grad_norm_sq'] > 0]

    if len(valid) < 3:
        print("WARNING: Too few valid trees for correlation analysis.")
        print(f"  Total trees: {len(df)}, valid (non-zero grad): {len(valid)}")
        analysis = {'n_trees': len(df), 'n_valid': len(valid)}
    else:
        # Pearson correlation: F vs log(||g||^2)
        log_gnorm = np.log1p(valid['grad_norm_sq'].values)
        F_vals = valid['F'].values

        r_pearson, p_pearson = stats.pearsonr(F_vals, log_gnorm)
        r_spearman, p_spearman = stats.spearmanr(F_vals, log_gnorm)

        print(f"\nCorrelation: F(T) vs log(1 + ||g||^2)")
        print(f"  Pearson  r = {r_pearson:.4f}  (p = {p_pearson:.4e})")
        print(f"  Spearman r = {r_spearman:.4f}  (p = {p_spearman:.4e})")

        # Also correlate individual terms
        r_H, p_H = stats.spearmanr(valid['H'].values, log_gnorm)
        r_rho, p_rho = stats.spearmanr(1.0 - valid['rho'].values, log_gnorm)
        print(f"\nIndividual term correlations (Spearman):")
        print(f"  H(r) vs log(1+||g||^2):     r = {r_H:.4f}  (p = {p_H:.4e})")
        print(f"  (1-rho) vs log(1+||g||^2):   r = {r_rho:.4f}  (p = {p_rho:.4e})")

        # Regime breakdown
        print(f"\nRegime distribution:")
        for regime, group in df.groupby('regime'):
            gv = group.dropna(subset=['grad_norm_sq'])
            gv = gv[gv['grad_norm_sq'] > 0]
            mean_F = group['F'].mean()
            mean_g = gv['grad_norm_sq'].mean() if len(gv) > 0 else float('nan')
            print(f"  {regime:15s}: n={len(group):3d}  mean_F={mean_F:.3f}  mean_||g||^2={mean_g:.4e}")

        analysis = {
            'pearson_r': r_pearson,
            'pearson_p': p_pearson,
            'spearman_r': r_spearman,
            'spearman_p': p_spearman,
            'H_spearman_r': r_H,
            'rho_spearman_r': r_rho,
            'n_trees': len(df),
            'n_valid': len(valid),
        }

    # --- Save results ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'config': vars(args),
        'analysis': analysis,
        'per_tree': results,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    # Also save as CSV for easy plotting
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")
    if len(valid) >= 3:
        wandb.log({
            "final/pearson_r": r_pearson,
            "final/pearson_p": p_pearson,
            "final/spearman_r": r_spearman,
            "final/spearman_p": p_spearman,
            "final/H_spearman_r": r_H,
            "final/rho_spearman_r": r_rho,
            "final/n_trees": len(df),
            "final/n_valid": len(valid),
        })

        # Log regime breakdown as a table
        regime_data = []
        for regime, group in df.groupby('regime'):
            gv = group.dropna(subset=['grad_norm_sq'])
            gv = gv[gv['grad_norm_sq'] > 0]
            regime_data.append([
                regime,
                len(group),
                group['F'].mean(),
                gv['grad_norm_sq'].mean() if len(gv) > 0 else float('nan')
            ])
        wandb.log({
            "regime_table": wandb.Table(
                columns=["regime", "count", "mean_F", "mean_grad_norm_sq"],
                data=regime_data
            )
        })
    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="H1: Fitness predicts gradient utility")
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-Math-1.5B',
                        help='HuggingFace model name or local path')
    parser.add_argument('--data', type=str,
                        default='./data_qwen25_math_cot/train/math_cot_train.parquet',
                        help='Path to training parquet file')
    parser.add_argument('--num_problems', type=int, default=50,
                        help='Number of problems to evaluate')
    parser.add_argument('--branching_factor', type=int, default=8,
                        help='Number of children per node (N)')
    parser.add_argument('--max_depth', type=int, default=3,
                        help='Maximum tree depth (D)')
    parser.add_argument('--step_length', type=int, default=384,
                        help='Tokens per tree expansion step')
    parser.add_argument('--max_prompt_length', type=int, default=512,
                        help='Maximum prompt token length')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Sampling temperature')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for problem selection')
    parser.add_argument('--output', type=str, default='evals/results/h1_results.json',
                        help='Output file path')
    parser.add_argument('--skip_gradient', action='store_true',
                        help='Skip gradient norm computation (for quick fitness-only analysis)')
    parser.add_argument('--full_precision', action='store_true',
                        help='Use float32 instead of float16 (needed for gradient computation)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Force full precision when computing gradients
    if not args.skip_gradient:
        args.full_precision = True
    run_h1_experiment(args)
