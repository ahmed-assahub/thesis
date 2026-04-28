# TODO / Roadmap

This roadmap is ordered to minimize silent mathematical mistakes. Complete each phase with tests before moving to the next one.

## Phase 0 — Repository setup

- [ ] Create/verify package structure:
  - [ ] `option_gpr/`
  - [ ] `tests/`
  - [ ] `docs/`
  - [ ] `experiments/`
- [ ] Add `pyproject.toml` or equivalent package configuration.
- [ ] Configure `pytest`.
- [ ] Add basic dependencies:
  - [ ] `numpy`
  - [ ] `scipy`
  - [ ] `pytest`
  - [ ] optional: `matplotlib`, `pandas`
- [ ] Add `AGENTS.md`, `TODO.md`, `docs/architecture.md`, and `docs/math_notes.md`.

## Phase 1 — RBF kernel

- [ ] Implement `option_gpr/kernels/rbf.py`.
- [ ] Implement `RBFKernel.K(X, Y)`.
- [ ] Add helper methods for:
  - [ ] `tau(X, Y) = t_i - s_j`
  - [ ] `chi(X, Y) = x_i - xi_j`
  - [ ] common base terms if useful.
- [ ] Add tests:
  - [ ] kernel matrix shape `(n, m)`.
  - [ ] symmetry `K(X, Y) == K(Y, X).T`.
  - [ ] diagonal equals `sigma_f**2` when `X == Y`.
  - [ ] positive semidefinite smoke test for `K(X, X)`.

## Phase 2 — Models and payoffs

- [ ] Implement `BlackScholesModel`.
- [ ] Implement `MertonJumpDiffusionModel` parameter container.
- [ ] Implement call payoff in log-price coordinates.
- [ ] Implement put payoff in log-price coordinates if needed.
- [ ] Add tests:
  - [ ] call payoff at `S<K`, `S=K`, `S>K`.
  - [ ] terminal payoff is not discounted.
  - [ ] Merton coefficients reduce to Black--Scholes coefficients when `jump_intensity=0`.

## Phase 3 — Black--Scholes log operator

- [ ] Implement `option_gpr/operators/bs_log_operator.py`.
- [ ] Implement:
  - [ ] `Lk(X, Y)`.
  - [ ] `kLp(X, Y)`.
  - [ ] `LkLp(X, Y)`.
- [ ] Use the formulas in `docs/math_notes.md` exactly.
- [ ] Add tests:
  - [ ] shape tests for all operator blocks.
  - [ ] `Lk(X, Y) == kLp(Y, X).T`.
  - [ ] closed-form identities against manually computed small examples.
  - [ ] finite-difference smoke check for `Lk` if feasible.

## Phase 4 — Stacked operator posterior

- [ ] Implement `option_gpr/posterior/stacked_gp.py`.
- [ ] Implement construction of:
  - [ ] observation vector `y_A`.
  - [ ] block matrix `K_AA`.
  - [ ] prediction matrix `K_star_A`.
  - [ ] noise matrix with separate `noise_int` and `noise_bd`.
- [ ] Use Cholesky factorization, never explicit inverse.
- [ ] Implement:
  - [ ] `fit(X_int, X_bd)`.
  - [ ] `predict(X_star, return_cov=False, return_var=False)`.
  - [ ] `negative_log_marginal_likelihood()`.
- [ ] Add tests:
  - [ ] block shape tests.
  - [ ] Cholesky succeeds with reasonable jitter.
  - [ ] posterior mean has expected shape.
  - [ ] posterior covariance has expected shape and is symmetric up to tolerance.

## Phase 5 — Grids and boundary handling

- [ ] Implement `GridSet` dataclass.
- [ ] Implement uniform log grid.
- [ ] Implement terminal-only boundary mode.
- [ ] Implement optional terminal plus spatial boundary mode.
- [ ] Implement boundary value function:
  - [ ] terminal call payoff.
  - [ ] terminal put payoff if needed.
  - [ ] optional lower spatial boundary.
  - [ ] optional upper spatial boundary.
- [ ] Add tests:
  - [ ] grid point shapes.
  - [ ] terminal boundary has `t == T`.
  - [ ] no duplicate or malformed points in small grids.

## Phase 6 — Black--Scholes benchmark and first experiment

- [ ] Implement Black--Scholes call price benchmark in log-price input.
- [ ] Implement Black--Scholes put price benchmark if needed.
- [ ] Add tests against known values or consistency checks.
- [ ] Implement `experiments/01_bs_accuracy.py`.
- [ ] Report:
  - [ ] RMSE.
  - [ ] max absolute error.
  - [ ] relative L2 error.
  - [ ] runtime.
  - [ ] condition number.
- [ ] Save results to `results/`.

## Phase 7 — Merton jump diffusion operator

- [ ] Implement `option_gpr/operators/mjd_operator.py` using thesis appendix identities.
- [ ] Keep jump-dependent terms isolated in helper functions where possible.
- [ ] Add tests:
  - [ ] with `jump_intensity=0`, `MJDOperator.Lk == BSLogOperator.Lk`.
  - [ ] with `jump_intensity=0`, `MJDOperator.kLp == BSLogOperator.kLp`.
  - [ ] with `jump_intensity=0`, `MJDOperator.LkLp == BSLogOperator.LkLp`.
  - [ ] jump terms vanish exactly when `jump_intensity=0`.

## Phase 8 — Merton benchmark

- [ ] Implement Merton series benchmark.
- [ ] Add convergence parameter `n_terms`.
- [ ] Add tests:
  - [ ] Merton series reduces approximately to Black--Scholes when `jump_intensity=0`.
  - [ ] increasing `n_terms` stabilizes prices.
- [ ] Implement `experiments/02_merton_accuracy.py`.

## Phase 9 — Metrics

- [ ] Implement `metrics/errors.py`:
  - [ ] `rmse`.
  - [ ] `mae`.
  - [ ] `max_abs_error`.
  - [ ] `relative_l2_error`.
  - [ ] `pointwise_relative_error` with epsilon.
  - [ ] `condition_number`.
- [ ] Add tests for all metrics.

## Phase 10 — Grid and convergence experiments

- [ ] Implement strike-centered grid.
- [ ] Implement random grid.
- [ ] Optional: implement Sobol grid.
- [ ] Implement `experiments/03_grid_comparison.py`.
- [ ] Implement `experiments/04_convergence.py`.
- [ ] Track:
  - [ ] errors.
  - [ ] runtime.
  - [ ] condition numbers.
  - [ ] grid metadata.

## Phase 11 — Hyperparameter optimization

- [ ] Implement log-hyperparameter packing/unpacking.
- [ ] Implement negative log marginal likelihood objective.
- [ ] Implement optional validation RMSE objective.
- [ ] Implement Nelder--Mead wrapper.
- [ ] Implement `experiments/06_hyperparams.py`.
- [ ] Compare fixed vs optimized hyperparameters.

## Phase 12 — Monte Carlo benchmark

- [ ] Implement simple Black--Scholes Monte Carlo if useful.
- [ ] Implement Merton Monte Carlo.
- [ ] Return price, standard error, confidence interval, and runtime.
- [ ] Implement `experiments/05_runtime_mc.py`.
- [ ] Compare batch pricing fairly:
  - [ ] GPR setup time.
  - [ ] GPR evaluation time.
  - [ ] Monte Carlo pointwise pricing time.

## Phase 13 — Optional Delta extension

Do this only after price experiments are stable.

- [ ] Add derivative kernel expressions supplied later.
- [ ] Implement `build_dKdx_star_A`.
- [ ] Implement `predict_dx`.
- [ ] Implement `predict_delta`, using `Delta = exp(-x) * partial_x_mean`.
- [ ] Compare to Black--Scholes Delta.
- [ ] Optional: compare to finite-difference Merton Delta.

## Phase 14 — Thesis outputs

- [ ] Clean experiment scripts.
- [ ] Save result tables as CSV.
- [ ] Save plots in reproducible form.
- [ ] Document parameter sets.
- [ ] Ensure all experiments can be rerun from a clean checkout.
