# Latent Dynamics Benchmark

A standard-library-only, reproducible reference benchmark for fitting a compact transition model from action-state trajectories and measuring multi-step rollout error.

## Run

```bash
python3 main.py --seeds 0 1 2 3 4 --training-seed 2026 --horizon 60 \
  --output artifacts/evaluation.json
python3 -m unittest discover -s tests -v
```

The JSON report records environment/model identifiers, the fixed training seed, every evaluation seed, fitted transition coefficients, per-seed rollout RMSE, a persistence baseline, and aggregate errors. Re-running with the same inputs produces the same report.

## Experiment design

The environment is the known synthetic system `x[t+1] = 0.92*x[t] + 0.35*u[t]`. The fitted ridge-linear transition uses noisy training trajectories; evaluation compares its free-running predictions against held-out deterministic rollouts. The persistence baseline predicts that the state will not change.

This makes one narrow question reproducible: does the fitted linear transition model beat a trivial persistence forecast over the selected rollout horizon?

## Scope and limits

- This is a one-dimensional synthetic control problem, not a trained neural world model.
- The environment equations are known and fixed; the experiment does not establish generalization to real or partially observed environments.
- The training seed and evaluation seeds are recorded separately; changing either changes the corresponding trajectory data.
- RMSE summarizes state prediction error. It is not a policy-safety, planning-quality, or governance metric.

See [Model-Based Evaluation Suite](https://github.com/MonikaDvorackova/model-based-eval-suite) for a separate toy evaluation that compares predicted and environment returns and can emit an AIGov Core evidence event.
