# AGENTS.md

## Project aim

This repository implements a physics-informed Gaussian process regression solver for European option pricing under Black--Scholes and Merton finite-activity jump diffusion models.

The central mathematical object is the posterior distribution induced by the stacked information operator

\[
\mathcal A u=
\begin{pmatrix}
\mathcal L u(X^{\mathrm{int}})\\
u(X^{\mathrm{bd}})
\end{pmatrix},
\]

where:

- \(X^{\mathrm{int}}\) are interior collocation points.
- \(X^{\mathrm{bd}}\) are boundary points.
- \(\mathcal L\) is the pricing operator.
- Boundary observations are point evaluations of the option price.
- The initial baseline should use terminal payoff boundary points only.
- Artificial lower/upper spatial boundaries should be supported optionally.

The main outputs are posterior mean and posterior covariance/variance for option prices on test points.

Prioritize mathematical correctness, testability, numerical stability, and clear structure over clever abstractions.

---

## Thesis context

This code supports a mathematics bachelor thesis titled approximately:

> Gaussian Process Regression for European Option Pricing under Finite Activity Jump Diffusion Models

The thesis works in log-price coordinates

\[
x=\log S.
\]

All space-time points must be represented as NumPy arrays of shape `(n, 2)` with columns:

```text
column 0: time t
column 1: log-price x
```

Do not use stock price \(S\) as the internal state coordinate except inside benchmark formulas or payoff conversions, where \(S=e^x\).

---

## Mathematical conventions

### RBF kernel

The base kernel is the anisotropic RBF kernel

\[
k((t,x),(s,\xi))
=\sigma_f^2
\exp\left(
-\frac{(t-s)^2}{2\ell_t^2}
-\frac{(x-\xi)^2}{2\ell_x^2}
\right).
\]

Use hyperparameter names:

- `ell_t`
- `ell_x`
- `sigma_f`

### Black--Scholes log-price operator

For the Black--Scholes limit, equivalently Merton with \(\lambda=0\), the operator is

\[
\mathcal L^{\mathrm{BS}}
=\partial_t+
\left(r-\frac12\sigma^2\right)\partial_x
+\frac12\sigma^2\partial_{xx}
-r.
\]

Equivalently,

\[
\mathcal L=a+\partial_t+b\partial_x+c\partial_{xx},
\]

with

\[
a=-r,\qquad
b=r-\frac12\sigma^2,\qquad
c=\frac12\sigma^2.
\]

### Merton jump diffusion coefficients

For Merton jump diffusion in log-price coordinates, use the thesis convention

\[
a=-(r+\lambda),
\]

\[
b=r-\frac12\sigma^2-
\lambda\left(e^{\mu+\delta^2/2}-1\right),
\]

\[
c=\frac12\sigma^2.
\]

In code, use explicit names to avoid confusion:

- `jump_intensity` for \(\lambda\)
- `jump_mean` for \(\mu\)
- `jump_std` for \(\delta\)

When `jump_intensity == 0`, the Merton operator and all Merton kernel identities must reduce to Black--Scholes.

---

## Architecture references

Read these files before implementing or changing core code:

1. `docs/architecture.md` — software design and implementation phases.
2. `docs/math_notes.md` — mathematical formulas and required invariants.
3. `TODO.md` — implementation roadmap.

If these files conflict, prefer `docs/math_notes.md` for formulas and `AGENTS.md` for global coding instructions.

---

## Required project structure

Use this package layout unless there is already an equivalent structure in the repository:

```text
option_gpr/
├── models/
├── kernels/
├── operators/
├── posterior/
├── grids/
├── payoffs/
├── hyperparams/
├── benchmarks/
├── metrics/
└── experiments/

tests/
docs/
```

The central solver object should be named `StackedOperatorGP`.

The core workflow should be conceptually:

```python
model = MertonJumpDiffusionModel(...)
kernel = RBFKernel(...)
operator = MJDOperator(model=model, kernel=kernel)

gp = StackedOperatorGP(
    model=model,
    kernel=kernel,
    operator=operator,
    noise_int=noise_int,
    noise_bd=noise_bd,
    jitter=jitter,
)

gp.fit(X_int, X_bd)
mean, cov = gp.predict(X_test, return_cov=True)
```

---

## Core APIs

### Kernel API

Every kernel matrix method must accept arrays of shape `(n, 2)` and `(m, 2)` and return an array of shape `(n, m)`.

```python
class RBFKernel:
    def K(self, X, Y): ...
```

### Operator API

Operator methods must follow this convention:

```python
class KernelOperator:
    def Lk(self, X, Y): ...      # operator on first argument
    def kLp(self, X, Y): ...     # operator on second argument
    def LkLp(self, X, Y): ...    # operator on both arguments
```

Mathematically:

\[
\texttt{Lk}(X,Y)_{ij}=\mathcal L_{z_i}k(z_i,y_j),
\]

\[
\texttt{kLp}(X,Y)_{ij}=\mathcal L_{y_j}k(z_i,y_j),
\]

\[
\texttt{LkLp}(X,Y)_{ij}=\mathcal L_{z_i}\mathcal L_{y_j}k(z_i,y_j).
\]

The sign convention must be documented in every operator implementation.

### Posterior API

The posterior solver must construct

\[
K_{AA}
=
\begin{pmatrix}
LkL'(X^{\mathrm{int}},X^{\mathrm{int}})
&
Lk(X^{\mathrm{int}},X^{\mathrm{bd}})
\\
kL'(X^{\mathrm{bd}},X^{\mathrm{int}})
&
k(X^{\mathrm{bd}},X^{\mathrm{bd}})
\end{pmatrix}.
\]

For test points \(X^\ast\), construct

\[
K_{\ast A}
=
\begin{pmatrix}
kL'(X^\ast,X^{\mathrm{int}})
&
k(X^\ast,X^{\mathrm{bd}})
\end{pmatrix}.
\]

The observation vector is

\[
y_A=
\begin{pmatrix}
0_{N_{\mathrm{int}}}\\
h(X^{\mathrm{bd}})
\end{pmatrix}.
\]

---

## Numerical requirements

- Never use explicit matrix inverses for posterior computations.
- Use Cholesky factorization and triangular solves whenever possible.
- Keep `noise_int`, `noise_bd`, and `jitter` as separate parameters.
- Use small positive noise and jitter during early debugging.
- Track matrix condition numbers in experiments.
- Use log-hyperparameters for optimization.
- Do not silently clip or reshape user inputs; validate shapes clearly.

---

## Boundary condition rules

The terminal payoff is not discounted.

For a call:

\[
h(T,x)=(e^x-K)^+.
\]

For a put:

\[
h(T,x)=(K-e^x)^+.
\]

Discounting is already accounted for by the \(-ru\) term in the pricing operator.

Artificial spatial boundaries are optional numerical aids, not the primary mathematical payoff condition. If implemented for a call, a reasonable far-field upper boundary is

\[
h(t,x_{\max})\approx e^{x_{\max}}-Ke^{-r(T-t)}.
\]

---

## Testing requirements

Before large experiments, implement unit tests for:

1. RBF kernel symmetry.
2. RBF matrix shape conventions.
3. Black--Scholes log-price kernel identities.
4. `Lk(X,Y) == kLp(Y,X).T` in the constant-coefficient Black--Scholes log-price case.
5. Merton operator equals Black--Scholes operator when `jump_intensity == 0`.
6. Stacked covariance block shapes.
7. Cholesky factorization with positive noise and jitter.
8. Terminal boundary posterior sanity: with small boundary noise, predictions at terminal boundary points should be close to payoff.
9. Black--Scholes reference price smoke test.

When Merton is implemented, add tests against the Merton series benchmark.

---

## Coding style

- Write clear, boring Python.
- Prefer small pure functions.
- Use dataclasses for parameter containers when useful.
- Use type hints where reasonable.
- Add docstrings to mathematical functions.
- Keep benchmarks separate from the GPR solver.
- Keep experiment scripts separate from library code.
- Do not add Delta/Greeks until the price solver is stable.
- Avoid hidden global state.
- Avoid unnecessary object inheritance.

---

## Implementation order

Proceed in phases:

1. Package skeleton, `pyproject.toml`, test setup.
2. RBF kernel and kernel tests.
3. Black--Scholes model and Black--Scholes log operator identities.
4. Stacked operator GP posterior.
5. Black--Scholes benchmark and first price accuracy experiment.
6. Merton model and Merton operator.
7. Merton series benchmark.
8. Grid comparison and convergence experiments.
9. Hyperparameter optimization.
10. Monte Carlo comparison.
11. Optional Delta extension.

Do not implement later phases until earlier phases pass tests.
