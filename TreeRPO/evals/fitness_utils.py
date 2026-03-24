"""
Shared utilities for tree fitness evaluation experiments.

Provides:
  - compute_tree_fitness: F(T) = H(r) * (1 - rho_{pi,r})
  - collect_leaf_rewards: gather binary rewards from tree leaves
  - collect_node_stats: gather (log_prob, reward) pairs from all non-root nodes
  - walk_tree: generic DFS over a TreeNode
  - binary_entropy: H(p) for a Bernoulli variable
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable


# ---------------------------------------------------------------------------
# Basic math helpers
# ---------------------------------------------------------------------------

def binary_entropy(p: float) -> float:
    """Binary entropy H(p) in bits.  Returns 0 for p in {0, 1}."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r between two 1-D arrays.  Returns 0 when degenerate."""
    if len(x) < 2:
        return 0.0
    std_x = np.std(x, ddof=0)
    std_y = np.std(y, ddof=0)
    if std_x < 1e-12 or std_y < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Tree walking
# ---------------------------------------------------------------------------

def walk_tree(node, visitor_fn: Callable, depth: int = 0):
    """Pre-order DFS.  `visitor_fn(node, depth)` is called on every node."""
    visitor_fn(node, depth)
    for child in node.children:
        walk_tree(child, visitor_fn, depth + 1)


def collect_leaf_rewards(root) -> List[float]:
    """Return binary rewards of all leaf nodes in the tree."""
    leaves: List[float] = []

    def _visit(node, depth):
        if len(node.children) == 0:
            leaves.append(float(node.E_reward))

    walk_tree(root, _visit)
    return leaves


def collect_node_logprobs_rewards(root) -> Tuple[np.ndarray, np.ndarray]:
    """Collect (sum-log-prob, propagated reward) for every non-root node.

    Requires that each node has `batch_dict['old_log_probs']` populated
    (i.e. after calling `compute_log_prob` on the train batch).
    For nodes that haven't had log-probs computed yet, we skip them.

    Returns:
        log_probs: shape (N,) — sum of token log-probs per step
        rewards:   shape (N,) — propagated E_reward per node
    """
    log_probs_list: List[float] = []
    rewards_list: List[float] = []

    def _visit(node, depth):
        if depth == 0:
            return  # skip root
        if 'old_log_probs' not in node.batch_dict:
            return
        # sum token-level log probs over the response (step) tokens,
        # masked by attention
        import torch
        lp = node.batch_dict['old_log_probs']  # (1, response_length)
        mask = node.batch_dict['attention_mask'][:, -lp.shape[-1]:]
        token_lps = (lp * mask).sum().item()
        log_probs_list.append(token_lps)
        rewards_list.append(float(node.E_reward))

    walk_tree(root, _visit)
    return np.array(log_probs_list), np.array(rewards_list)


def collect_node_logprobs_rewards_from_batch(
    train_batch,
    level_node_list: List[list],
    max_tree_depth: int,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Alternative: collect (log_prob, reward) per tree from the flattened train batch.

    Since ray_trainer computes old_log_probs on the flattened train_batch
    (after `_format_train_batch`), individual node batch_dicts won't have 
    `old_log_probs`. This function reconstructs the per-tree mapping.

    We walk the level_node_list and match nodes to train_batch rows by
    checking that the node wasn't pruned (ΔR ≥ 0.1) and has bro_rewards set.

    Args:
        train_batch: DataProto with 'old_log_probs' of shape (B, resp_len)
        level_node_list: list of lists of TreeNodes per depth
        max_tree_depth: maximum depth of tree

    Returns:
        dict mapping root_index -> (log_probs_array, rewards_array)
    """
    import torch

    old_lp = train_batch.batch['old_log_probs']       # (B, resp_len)
    resp_mask = train_batch.batch['attention_mask'][:, -old_lp.shape[-1]:]

    # Per-row sum log prob
    row_lps = (old_lp * resp_mask).sum(dim=-1).cpu().numpy()  # (B,)

    # Walk tree structure and assign row indices
    row_idx = 0
    # We need to figure out which root each node belongs to.
    # level_node_list[0] = list of roots.  Each non-root node at depth d
    # was appended to level_node_list[d] in order.
    # _format_train_batch iterates depth 1..D, then within each depth
    # iterates level_node_list[depth], skipping pruned (ΔR < 0.1).

    # Build a mapping: node id -> root index
    node_to_root: Dict[int, int] = {}
    for root_idx, root in enumerate(level_node_list[0]):
        def _tag(node, root_i=root_idx):
            node_to_root[id(node)] = root_i
            for c in node.children:
                _tag(c, root_i)
        _tag(root)

    per_tree: Dict[int, Tuple[List[float], List[float]]] = {}

    for depth in range(1, max_tree_depth + 1):
        if depth >= len(level_node_list) or len(level_node_list[depth]) == 0:
            break
        for node in level_node_list[depth]:
            if not hasattr(node, 'bro_rewards'):
                continue
            # check pruning condition (mirrors _format_train_batch)
            if max(node.bro_rewards) - min(node.bro_rewards) < 0.1:
                continue
            if row_idx >= len(row_lps):
                break

            root_i = node_to_root.get(id(node), -1)
            if root_i not in per_tree:
                per_tree[root_i] = ([], [])
            per_tree[root_i][0].append(float(row_lps[row_idx]))
            per_tree[root_i][1].append(float(node.E_reward))
            row_idx += 1

    result = {}
    for root_i, (lps, rews) in per_tree.items():
        result[root_i] = (np.array(lps), np.array(rews))
    return result


# ---------------------------------------------------------------------------
# Fitness computation
# ---------------------------------------------------------------------------

def compute_tree_fitness(
    leaf_rewards: List[float],
    node_log_probs: Optional[np.ndarray] = None,
    node_rewards: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute the tree fitness score F(T) = H(r) * (1 - rho).

    Args:
        leaf_rewards: binary rewards at all leaves of the tree
        node_log_probs: per-node sum-log-probs (non-root).
                        If None, only the entropy term is returned.
        node_rewards: per-node propagated rewards (non-root).
                      Same length as node_log_probs.

    Returns:
        dict with keys:
            'p_hat':   fraction of correct leaves
            'H':       binary entropy of p_hat
            'rho':     Pearson correlation (or 0 if log_probs not given)
            'F':       fitness score H * (1 - rho)
            'n_leaves': number of leaves
            'n_nodes':  number of non-root nodes used for rho
    """
    p_hat = float(np.mean(leaf_rewards)) if len(leaf_rewards) > 0 else 0.0
    H = binary_entropy(p_hat)

    rho = 0.0
    n_nodes = 0
    if node_log_probs is not None and node_rewards is not None and len(node_log_probs) >= 2:
        rho = pearson_correlation(node_log_probs, node_rewards)
        n_nodes = len(node_log_probs)

    F = H * (1.0 - rho)

    return {
        'p_hat': p_hat,
        'H': H,
        'rho': rho,
        'F': F,
        'n_leaves': len(leaf_rewards),
        'n_nodes': n_nodes,
    }


def classify_tree(fitness: Dict[str, float],
                  tau_low: float = 0.2,
                  tau_high: float = 0.8) -> str:
    """Classify a tree into regimes based on fitness.

    Returns one of: 'dead_correct', 'dead_wrong', 'stale', 'informative'.
    """
    F = fitness['F']
    p_hat = fitness['p_hat']
    H = fitness['H']

    if F <= tau_low:
        if H < 0.2:  # low entropy → all agree
            return 'dead_correct' if p_hat > 0.5 else 'dead_wrong'
    elif F <=tau_low:
        return 'stale'
    elif F <= tau_high:
        return 'stale'
    else:
        return 'informative'


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def fitness_to_row(tree_idx: int, fitness: Dict[str, float], regime: str,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convert fitness dict to a flat row for DataFrame / JSON output."""
    row = {'tree_idx': tree_idx, 'regime': regime}
    row.update(fitness)
    if extra:
        row.update(extra)
    return row
