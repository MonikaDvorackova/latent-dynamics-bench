# Latent Dynamics Benchmark

A reproducible baseline that learns a compact transition model from action-state trajectories, then reports rollout error. It intentionally starts with a transparent linear model so architectural comparisons have a stable reference.

Run `python3 main.py`. Extend with RSSM or transformer adapters, multiple seeds, fixed training budgets, and per-horizon error curves.
