"""Multi-seed linear latent-dynamics reference benchmark (stdlib only)."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import fmean


ENVIRONMENT_VERSION = "linear-control-v1"
MODEL_VERSION = "ridge-linear-v1"
A_TRUE = 0.92
B_TRUE = 0.35


def solve(matrix: list[list[float]], target: list[float]) -> list[float]:
    """Solve a small square linear system by pivoted Gaussian elimination."""
    n = len(target)
    rows = [list(map(float, matrix[i])) + [float(target[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(rows[row][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        if abs(rows[col][col]) < 1e-12:
            raise ValueError("singular design matrix")
        scale = rows[col][col]
        rows[col] = [value / scale for value in rows[col]]
        for row in range(n):
            if row != col:
                factor = rows[row][col]
                rows[row] = [
                    left - factor * right for left, right in zip(rows[row], rows[col])
                ]
    return [rows[i][-1] for i in range(n)]


def fit(xs: list[float], actions: list[float], next_xs: list[float]) -> list[float]:
    """Fit x[t+1] = a*x[t] + b*u[t] + c with a small ridge penalty."""
    if not xs or len(xs) != len(actions) or len(xs) != len(next_xs):
        raise ValueError("training arrays must have the same non-zero length")
    design = [[x, action, 1.0] for x, action in zip(xs, actions)]
    width = 3
    gram = [
        [
            sum(row[i] * row[j] for row in design) + (1e-6 if i == j else 0.0)
            for j in range(width)
        ]
        for i in range(width)
    ]
    rhs = [sum(row[i] * y for row, y in zip(design, next_xs)) for i in range(width)]
    return solve(gram, rhs)


def training_data(seed: int, episodes: int = 20, steps: int = 80):
    rng = random.Random(seed)
    xs: list[float] = []
    actions: list[float] = []
    next_xs: list[float] = []
    for _ in range(episodes):
        state = rng.uniform(-1.0, 1.0)
        for _ in range(steps):
            action = rng.uniform(-1.0, 1.0)
            next_state = A_TRUE * state + B_TRUE * action + rng.gauss(0.0, 0.01)
            xs.append(state)
            actions.append(action)
            next_xs.append(next_state)
            state = next_state
    return xs, actions, next_xs


def evaluate_seed(coefficients: list[float], seed: int, horizon: int = 60) -> dict:
    rng = random.Random(seed)
    actual = predicted = persistence = rng.uniform(-1.0, 1.0)
    squared_model_errors: list[float] = []
    squared_baseline_errors: list[float] = []
    for _ in range(horizon):
        action = rng.uniform(-1.0, 1.0)
        actual_next = A_TRUE * actual + B_TRUE * action
        predicted_next = (
            coefficients[0] * predicted + coefficients[1] * action + coefficients[2]
        )
        persistence_next = persistence
        squared_model_errors.append((actual_next - predicted_next) ** 2)
        squared_baseline_errors.append((actual_next - persistence_next) ** 2)
        actual, predicted, persistence = actual_next, predicted_next, persistence_next
    return {
        "seed": seed,
        "horizon": horizon,
        "rollout_rmse": math.sqrt(fmean(squared_model_errors)),
        "persistence_baseline_rmse": math.sqrt(fmean(squared_baseline_errors)),
    }


def run(seeds: list[int], training_seed: int = 2026, horizon: int = 60) -> dict:
    if not seeds:
        raise ValueError("provide at least one evaluation seed")
    xs, actions, next_xs = training_data(training_seed)
    coefficients = fit(xs, actions, next_xs)
    episodes = [evaluate_seed(coefficients, seed, horizon) for seed in seeds]
    return {
        "environment_version": ENVIRONMENT_VERSION,
        "model_version": MODEL_VERSION,
        "training_seed": training_seed,
        "evaluation_seeds": seeds,
        "horizon": horizon,
        "transition_coefficients": {
            "state": coefficients[0],
            "action": coefficients[1],
            "bias": coefficients[2],
        },
        "episodes": episodes,
        "summary": {
            "mean_rollout_rmse": fmean(item["rollout_rmse"] for item in episodes),
            "mean_persistence_baseline_rmse": fmean(
                item["persistence_baseline_rmse"] for item in episodes
            ),
        },
        "limitations": [
            "One-dimensional synthetic linear dynamics; no learned neural representation.",
            "Evaluation uses deterministic dynamics and a fixed training procedure.",
            "Results are a software reference baseline, not evidence about real environments.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--training-seed", type=int, default=2026)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run(args.seeds, args.training_seed, args.horizon)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
