#!/usr/bin/env python3
"""
H2 Validation: Does critique heal low-fitness trees?

Hypothesis: Applying a natural-language critique at the failing step of a
Dead-Wrong tree (F ≈ 0, all leaves wrong) and grafting refined continuations
increases the fitness score F(T_healed) > F(T_original).

Protocol:
  1. Load model, tokenizer, and optionally a separate critic model
  2. Build trees for math problems (same as H1)
  3. Compute F_before per tree and identify Dead-Wrong trees
  4. For each Dead-Wrong tree:
     a. Locate the shallowest depth where all children fail
     b. Extract the partial reasoning prefix up to that depth
     c. Generate a critique of the partial reasoning
     d. Generate k refined continuations conditioned on the critique
     e. Graft as new children, re-score, re-propagate
     f. Compute F_after
  5. Report ΔF = F_after - F_before across all healed trees

Usage:
  python evals/h2_critique_heals_fitness.py \
      --model Qwen/Qwen2.5-Math-1.5B \
      --data ./data_qwen25_math_cot/train/math_cot_train.parquet \
      --num_problems 50 \
      --branching_factor 8 \
      --max_depth 3 \
      --step_length 384 \
      --num_refinements 4 \
      --output evals/results/h2_results.json

  # Use a separate critic model:
  python evals/h2_critique_heals_fitness.py \
      --model Qwen/Qwen2.5-Math-1.5B \
      --critic_model Qwen/Qwen2.5-Math-7B-Instruct \
      --data ./data_qwen25_math_cot/train/math_cot_train.parquet \
      ...
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rllm.rewards.rl_reward import rllm_reward_fn
from evals.fitness_utils import (
    binary_entropy,
    pearson_correlation,
    compute_tree_fitness,
    classify_tree,
    fitness_to_row,
)

# Import tree + scoring from H1 to avoid duplication
from evals.h1_fitness_predicts_gradient import (
    EvalTreeNode,
    generate_children,
    build_tree,
    score_leaves,
    propagate_rewards,
    assign_sibling_rewards,
    compute_step_log_probs,
    collect_node_stats,
    load_problems,
    format_prompt,
)


# ---------------------------------------------------------------------------
# Critique prompt templates
# ---------------------------------------------------------------------------

CRITIQUE_SYSTEM_PROMPT = (
    "You are a mathematical reasoning critic. Given a student's partial "
    "solution to a math problem, identify the specific error or flawed "
    "reasoning step and explain what went wrong concisely."
)

CRITIQUE_USER_TEMPLATE = (
    "## Problem\n{problem}\n\n"
    "## Student's Partial Solution (up to the failing step)\n{partial_solution}\n\n"
    "## Task\n"
    "This partial solution leads to incorrect answers. Identify the error "
    "in the reasoning and explain what should be done differently. "
    "Be specific and concise."
)

REFINE_SYSTEM_PROMPT = (
    "You are a math problem solver. Given a problem, a partial (incorrect) "
    "solution attempt, and a critique explaining the error, generate a "
    "corrected continuation that fixes the identified issue."
)

REFINE_USER_TEMPLATE = (
    "## Problem\n{problem}\n\n"
    "## Previous Attempt (contains error)\n{partial_solution}\n\n"
    "## Critique\n{critique}\n\n"
    "## Task\n"
    "Continue solving from the point before the error, incorporating the "
    "critique's feedback. Provide the corrected continuation only."
)


# ---------------------------------------------------------------------------
# Finding the failing step in a tree
# ---------------------------------------------------------------------------

def find_failing_depth(root: EvalTreeNode) -> Tuple[Optional[EvalTreeNode], int]:
    """Find the shallowest node where ALL children have E_reward ≈ 0.

    Returns (node, depth) or (None, -1) if no such node exists.
    """
    # BFS to find shallowest all-fail node
    queue = [root]
    while queue:
        next_queue = []
        for node in queue:
            if not node.children:
                continue
            child_rewards = [c.E_reward for c in node.children]
            # All children have reward < threshold → this is the failing point
            if max(child_rewards) < 0.05:
                return node, node.depth
            next_queue.extend(node.children)
        queue = next_queue

    return None, -1


def extract_prefix_text(node: EvalTreeNode, tokenizer) -> str:
    """Decode the token sequence from root to this node."""
    return tokenizer.decode(node.token_ids, skip_special_tokens=False)


def extract_problem_text(prompt_messages: list) -> str:
    """Extract the user's question from chat messages."""
    for msg in prompt_messages:
        if msg.get('role') == 'user':
            return msg.get('content', '')
    # Fallback: last message
    return prompt_messages[-1].get('content', '') if prompt_messages else ''


# ---------------------------------------------------------------------------
# Critique generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_critique(
    model,
    tokenizer,
    problem_text: str,
    partial_solution_text: str,
    device: torch.device,
    max_critique_tokens: int = 256,
) -> str:
    """Generate a natural-language critique of the partial solution."""
    messages = [
        {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
        {"role": "user", "content": CRITIQUE_USER_TEMPLATE.format(
            problem=problem_text,
            partial_solution=partial_solution_text,
        )},
    ]

    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = f"{CRITIQUE_SYSTEM_PROMPT}\n\n{messages[1]['content']}"

    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_critique_tokens,
        temperature=0.3,  # low temp for focused critique
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    critique_ids = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(critique_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Refinement generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_refinements(
    model,
    tokenizer,
    problem_text: str,
    partial_solution_text: str,
    critique_text: str,
    num_refinements: int,
    failing_node: EvalTreeNode,
    step_length: int,
    max_total_length: int,
    temperature: float,
    device: torch.device,
) -> List[EvalTreeNode]:
    """Generate k refined continuations from the point of failure.

    The refinements branch from the parent of the failing depth,
    so they replace the bad step entirely.
    """
    # Build the refinement prompt: original prefix + critique instruction
    messages = [
        {"role": "system", "content": REFINE_SYSTEM_PROMPT},
        {"role": "user", "content": REFINE_USER_TEMPLATE.format(
            problem=problem_text,
            partial_solution=partial_solution_text,
            critique=critique_text,
        )},
    ]

    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = f"{REFINE_SYSTEM_PROMPT}\n\n{messages[1]['content']}"

    refine_prompt_ids = tokenizer.encode(text, return_tensors='pt').to(device)

    # Remaining budget: from failing node depth to max total
    remaining_tokens = max_total_length - len(failing_node.token_ids)
    gen_length = max(remaining_tokens, step_length)

    refinements = []
    for _ in range(num_refinements):
        outputs = model.generate(
            input_ids=refine_prompt_ids,
            max_new_tokens=gen_length,
            temperature=temperature,
            do_sample=True,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        new_ids = outputs[0][refine_prompt_ids.shape[1]:].tolist()

        # Create a synthetic leaf node: prefix up to the failing node + new continuation
        # The refined node replaces from failing_node's depth onward
        full_ids = failing_node.token_ids + new_ids
        if len(full_ids) > max_total_length:
            full_ids = full_ids[:max_total_length]

        child = EvalTreeNode(
            token_ids=full_ids,
            depth=failing_node.depth + 1,
        )
        child.step_token_ids = new_ids[:len(full_ids) - len(failing_node.token_ids)]
        # Mark as leaf (no further expansion)
        refinements.append(child)

    return refinements


# ---------------------------------------------------------------------------
# Tree healing (graft refinements + re-evaluate)
# ---------------------------------------------------------------------------

def heal_tree(
    root: EvalTreeNode,
    failing_node: EvalTreeNode,
    refinements: List[EvalTreeNode],
    tokenizer,
    data_source: str,
    ground_truth: str,
) -> EvalTreeNode:
    """Graft refinements onto the failing node and re-evaluate.

    We add the refined continuations as new children of the failing node,
    re-score their leaves, and re-propagate rewards through the tree.
    """
    # Score each refinement leaf
    for ref_node in refinements:
        text = tokenizer.decode(ref_node.token_ids, skip_special_tokens=False)
        score = rllm_reward_fn(
            data_source=data_source,
            llm_solution=text,
            ground_truth=ground_truth,
            ignore_think=True,
        )
        ref_node.E_reward = float(score)

    # Graft: add refinements as children of the failing node
    for ref_node in refinements:
        failing_node.children.append(ref_node)
        failing_node.child_rewards.append(ref_node.E_reward)

    # Re-propagate rewards from scratch
    propagate_rewards(root)
    assign_sibling_rewards(root)

    return root


# ---------------------------------------------------------------------------
# Collect all leaf rewards
# ---------------------------------------------------------------------------

def get_all_leaf_rewards(root: EvalTreeNode) -> List[float]:
    leaves = []
    def _collect(n):
        if n.is_leaf:
            leaves.append(n.E_reward)
        for c in n.children:
            _collect(c)
    _collect(root)
    return leaves


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_h2_experiment(args):
    print(f"=== H2: Critique Heals Low-Fitness Trees ===")
    print(f"Model:             {args.model}")
    print(f"Critic model:      {args.critic_model or '(self-critique)'}")
    print(f"Data:              {args.data}")
    print(f"Num problems:      {args.num_problems}")
    print(f"Branching factor:  {args.branching_factor}")
    print(f"Max depth:         {args.max_depth}")
    print(f"Step length:       {args.step_length}")
    print(f"Num refinements:   {args.num_refinements}")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load policy model
    print("Loading policy model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device,
    )
    model.eval()

    # Load critic model (if separate)
    if args.critic_model and args.critic_model != args.model:
        print(f"Loading critic model: {args.critic_model}...")
        critic_tokenizer = AutoTokenizer.from_pretrained(args.critic_model, trust_remote_code=True)
        if critic_tokenizer.pad_token_id is None:
            critic_tokenizer.pad_token_id = critic_tokenizer.eos_token_id
        critic_model = AutoModelForCausalLM.from_pretrained(
            args.critic_model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device,
        )
        critic_model.eval()
    else:
        critic_model = model
        critic_tokenizer = tokenizer

    max_prompt_length = args.max_prompt_length
    max_response_length = args.step_length * args.max_depth
    max_total_length = max_prompt_length + max_response_length

    # Load problems
    print(f"Loading {args.num_problems} problems...")
    problems = load_problems(args.data, args.num_problems, seed=args.seed)
    print(f"Loaded {len(problems)} problems.\n")

    results = []
    n_dead_wrong = 0
    n_healed = 0
    n_improved = 0

    for i, problem in enumerate(problems):
        t0 = time.time()
        print(f"[{i+1}/{len(problems)}] Problem {problem['index']}...")

        prompt_ids = format_prompt(problem['prompt'], tokenizer)
        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[:max_prompt_length]

        problem_text = extract_problem_text(problem['prompt'])

        # --- Phase 1: Build tree ---
        tree_root = build_tree(
            model=model, tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            branching_factor=args.branching_factor,
            max_depth=args.max_depth,
            step_length=args.step_length,
            max_total_length=max_total_length,
            temperature=args.temperature,
            device=device,
        )

        # --- Phase 2: Score + propagate ---
        score_leaves(
            tree_root, tokenizer,
            data_source=problem['data_source'],
            ground_truth=problem['ground_truth'],
        )
        propagate_rewards(tree_root)
        assign_sibling_rewards(tree_root)

        # --- Phase 3: Compute F_before ---
        leaf_rewards_before = get_all_leaf_rewards(tree_root)

        # For rho, we need log-probs
        compute_step_log_probs(model, tree_root, device)
        node_lps, node_rews = collect_node_stats(tree_root)

        fitness_before = compute_tree_fitness(leaf_rewards_before, node_lps, node_rews)
        regime = classify_tree(fitness_before)

        print(f"  BEFORE: p_hat={fitness_before['p_hat']:.3f}  H={fitness_before['H']:.3f}  "
              f"rho={fitness_before['rho']:.3f}  F={fitness_before['F']:.3f}  regime={regime}")

        row = {
            'tree_idx': i,
            'problem_idx': problem['index'],
            'data_source': problem['data_source'],
            'regime': regime,
            'F_before': fitness_before['F'],
            'H_before': fitness_before['H'],
            'rho_before': fitness_before['rho'],
            'p_hat_before': fitness_before['p_hat'],
            'n_leaves_before': fitness_before['n_leaves'],
        }

        # --- Phase 4: Critique + heal if Dead-Wrong ---
        if regime == 'dead_wrong':
            n_dead_wrong += 1

            failing_node, fail_depth = find_failing_depth(tree_root)
            if failing_node is None:
                print(f"  Dead-wrong but no clear failing node found. Skipping.")
                row.update({
                    'healed': False, 'fail_depth': -1,
                    'critique': None, 'n_refinements': 0,
                    'F_after': fitness_before['F'], 'delta_F': 0.0,
                })
                results.append(row)
                continue

            print(f"  Failing at depth {fail_depth}. Generating critique...")
            partial_text = extract_prefix_text(failing_node, tokenizer)

            # Generate critique
            critique_text = generate_critique(
                critic_model, critic_tokenizer,
                problem_text=problem_text,
                partial_solution_text=partial_text,
                device=device,
                max_critique_tokens=args.max_critique_tokens,
            )
            print(f"  Critique: {critique_text[:120]}...")

            # Generate refined continuations
            print(f"  Generating {args.num_refinements} refinements...")
            refinements = generate_refinements(
                model=model, tokenizer=tokenizer,
                problem_text=problem_text,
                partial_solution_text=partial_text,
                critique_text=critique_text,
                num_refinements=args.num_refinements,
                failing_node=failing_node,
                step_length=args.step_length,
                max_total_length=max_total_length,
                temperature=args.temperature,
                device=device,
            )

            # Graft + re-evaluate
            heal_tree(
                tree_root, failing_node, refinements,
                tokenizer=tokenizer,
                data_source=problem['data_source'],
                ground_truth=problem['ground_truth'],
            )
            n_healed += 1

            # --- Phase 5: Compute F_after ---
            leaf_rewards_after = get_all_leaf_rewards(tree_root)
            compute_step_log_probs(model, tree_root, device)
            node_lps_after, node_rews_after = collect_node_stats(tree_root)

            fitness_after = compute_tree_fitness(
                leaf_rewards_after, node_lps_after, node_rews_after
            )

            delta_F = fitness_after['F'] - fitness_before['F']
            if delta_F > 0:
                n_improved += 1

            print(f"  AFTER:  p_hat={fitness_after['p_hat']:.3f}  H={fitness_after['H']:.3f}  "
                  f"rho={fitness_after['rho']:.3f}  F={fitness_after['F']:.3f}")
            print(f"  ΔF = {delta_F:+.3f}  ({'improved' if delta_F > 0 else 'no improvement'})")

            # Count how many refinements were correct
            ref_rewards = [r.E_reward for r in refinements]
            n_ref_correct = sum(1 for r in ref_rewards if r > 0.5)

            row.update({
                'healed': True,
                'fail_depth': fail_depth,
                'critique': critique_text[:500],
                'n_refinements': len(refinements),
                'n_ref_correct': n_ref_correct,
                'F_after': fitness_after['F'],
                'H_after': fitness_after['H'],
                'rho_after': fitness_after['rho'],
                'p_hat_after': fitness_after['p_hat'],
                'n_leaves_after': fitness_after['n_leaves'],
                'delta_F': delta_F,
            })

        else:
            # Not dead_wrong — no healing needed
            row.update({
                'healed': False, 'fail_depth': -1,
                'critique': None, 'n_refinements': 0,
                'F_after': fitness_before['F'], 'delta_F': 0.0,
            })

        elapsed = time.time() - t0
        row['elapsed_s'] = elapsed
        print(f"  time: {elapsed:.1f}s\n")
        results.append(row)

        del tree_root
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Analysis ---
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    df = pd.DataFrame(results)

    print(f"\nOverall statistics:")
    print(f"  Total trees:          {len(df)}")
    print(f"  Dead-Wrong trees:     {n_dead_wrong}")
    print(f"  Healed trees:         {n_healed}")
    print(f"  Improved (ΔF > 0):    {n_improved}")
    if n_healed > 0:
        print(f"  Heal success rate:    {n_improved / n_healed:.1%}")

    print(f"\nRegime distribution:")
    for regime, group in df.groupby('regime'):
        print(f"  {regime:15s}: n={len(group)}")

    healed = df[df['healed'] == True]
    if len(healed) > 0:
        print(f"\nHealed trees analysis:")
        print(f"  Mean F_before:  {healed['F_before'].mean():.4f}")
        print(f"  Mean F_after:   {healed['F_after'].mean():.4f}")
        print(f"  Mean ΔF:        {healed['delta_F'].mean():.4f}")
        print(f"  Median ΔF:      {healed['delta_F'].median():.4f}")
        print(f"  Std ΔF:         {healed['delta_F'].std():.4f}")

        # One-sided t-test: ΔF > 0
        if len(healed) >= 3:
            t_stat, p_val = stats.ttest_1samp(healed['delta_F'].values, 0.0)
            # One-sided p-value (we want ΔF > 0)
            p_one_sided = p_val / 2 if t_stat > 0 else 1.0 - p_val / 2
            print(f"\n  One-sided t-test (H_a: mean ΔF > 0):")
            print(f"    t = {t_stat:.4f}, p = {p_one_sided:.4e}")

        # Breakdown by refinement success
        if 'n_ref_correct' in healed.columns:
            print(f"\n  Refinement accuracy:")
            total_refs = healed['n_refinements'].sum()
            total_correct = healed['n_ref_correct'].sum()
            if total_refs > 0:
                print(f"    Total refinements: {total_refs}")
                print(f"    Correct:           {total_correct} ({total_correct/total_refs:.1%})")

        # Correlation between n_ref_correct and ΔF
        if 'n_ref_correct' in healed.columns and len(healed) >= 3:
            r, p = stats.spearmanr(healed['n_ref_correct'].values, healed['delta_F'].values)
            print(f"\n  Spearman(n_ref_correct, ΔF): r={r:.4f}, p={p:.4e}")

        analysis = {
            'n_dead_wrong': n_dead_wrong,
            'n_healed': n_healed,
            'n_improved': n_improved,
            'mean_F_before': float(healed['F_before'].mean()),
            'mean_F_after': float(healed['F_after'].mean()),
            'mean_delta_F': float(healed['delta_F'].mean()),
            'median_delta_F': float(healed['delta_F'].median()),
        }
        if len(healed) >= 3:
            analysis['t_stat'] = float(t_stat)
            analysis['p_one_sided'] = float(p_one_sided)
    else:
        print("\nNo Dead-Wrong trees found — nothing to heal.")
        print("Try increasing --num_problems or using a weaker model.")
        analysis = {'n_dead_wrong': 0, 'n_healed': 0}

    # --- Save ---
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

    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="H2: Critique heals low-fitness trees")
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-Math-1.5B',
                        help='Policy model (HuggingFace name or local path)')
    parser.add_argument('--critic_model', type=str, default=None,
                        help='Critic model for generating critiques (default: self-critique with --model)')
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
                        help='Sampling temperature for policy')
    parser.add_argument('--num_refinements', type=int, default=4,
                        help='Number of refined continuations to graft (k)')
    parser.add_argument('--max_critique_tokens', type=int, default=256,
                        help='Max tokens for critique generation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default='evals/results/h2_results.json',
                        help='Output file path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_h2_experiment(args)
