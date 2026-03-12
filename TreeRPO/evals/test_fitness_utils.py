"""Quick smoke test for fitness_utils."""
import sys
sys.path.insert(0, '.')

from evals.fitness_utils import compute_tree_fitness, binary_entropy
import numpy as np

# Binary entropy
assert abs(binary_entropy(0.0)) < 1e-10
assert abs(binary_entropy(1.0)) < 1e-10
assert abs(binary_entropy(0.5) - 1.0) < 1e-10

# All correct -> F=0
f = compute_tree_fitness([1.0, 1.0, 1.0, 1.0])
assert f['H'] < 0.01, f"Expected H~0, got {f['H']}"
assert f['F'] < 0.01, f"Expected F~0, got {f['F']}"

# All wrong -> F=0
f = compute_tree_fitness([0.0, 0.0, 0.0, 0.0])
assert f['H'] < 0.01

# 50/50 rewards, no log probs -> F = H*1 = 1
f = compute_tree_fitness([0.0, 1.0, 0.0, 1.0])
assert abs(f['H'] - 1.0) < 0.01
assert abs(f['F'] - 1.0) < 0.01

# With correlated log probs (rho ~ 1) -> F ~ 0
lps = np.array([0.1, 0.9, 0.1, 0.9])
rews = np.array([0.1, 0.9, 0.1, 0.9])
f = compute_tree_fitness([0.0, 1.0, 0.0, 1.0], lps, rews)
assert f['rho'] > 0.9, f"Expected high rho, got {f['rho']}"
assert f['F'] < 0.2, f"Expected low F, got {f['F']}"

# Anti-correlated -> high F
lps2 = np.array([0.9, 0.1, 0.9, 0.1])
rews2 = np.array([0.1, 0.9, 0.1, 0.9])
f2 = compute_tree_fitness([0.0, 1.0, 0.0, 1.0], lps2, rews2)
assert f2['rho'] < -0.9, f"Expected negative rho, got {f2['rho']}"
assert f2['F'] > 1.5, f"Expected high F, got {f2['F']}"

print('All fitness_utils tests passed!')
