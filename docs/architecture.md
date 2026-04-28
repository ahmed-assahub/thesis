# Option GPR Implementation Architecture

## Purpose

This document specifies the Python architecture for implementing a physics-informed Gaussian process regression solver for European option pricing under Black--Scholes and Merton jump diffusion models.

The implementation should support:

- Black--Scholes pricing as the special case \(\lambda=0\).
- Merton jump diffusion pricing for finite-activity jumps.
- The stacked information operator

\[
\mathcal A u=
\begin{pmatrix}
\mathcal L u(X^{\mathrm{int}})\\
u(X^{\mathrm{bd}})
\end{pmatrix},
\]

where \(X^{\mathrm{int}}\) are interior collocation points and \(X^{\mathrm{bd}}\) are boundary points.

- Posterior mean and posterior covariance evaluation.
- Arbitrary choices of model parameters such as \(r,\sigma,\lambda,\mu,\delta,K,T\).
- Arbitrary choices of kernel hyperparameters such as \(\ell_t,\ell_x,\sigma_f\).
- Separate noise/regularization levels for interior and boundary observations.
- Different grid designs for testing collocation quality.
- Hyperparameter optimization, probably via Nelder--Mead.
- Benchmarking against Black--Scholes closed-form prices, Merton series prices, and Monte Carlo.
- Future extension to Greeks, especially Delta, without rewriting the core solver.

The central design principle is modularity: models, kernels, operators, grids, posterior construction, benchmarks, metrics, and experiments should be separated.

The solver should not be implemented as one large function. The core workflow should look conceptually like this:

```python
model = MertonJumpDiffusionModel(
    r=r,
    sigma=sigma,
    jump_intensity=lam,
    jump_mean=mu,
    jump_std=delta,
    strike=K,
    maturity=T,
)

kernel = RBFKernel(
    ell_t=ell_t,
    ell_x=ell_x,
    sigma_f=sigma_f,
)

operator = MJDOperator(model=model, kernel=kernel)

grid = make_grid(...)

gp = StackedOperatorGP(
    model=model,
    kernel=kernel,
    operator=operator,
    noise_int=noise_int,
    noise_bd=noise_bd,
    jitter=jitter,
)

gp.fit(grid.X_int, grid.X_bd)
mean, cov = gp.predict(grid.X_test, return_cov=True)
```

---

## Mathematical Setting

We work in log-price coordinates. Let

\[
x=\log S.
\]

The unknown option price is written as

\[
u(t,x),
\]

where \(S=e^x\). The computational domain is finite:

\[
(t,x)\in[0,T]\times[x_{\min},x_{\max}],
\]

although the true log-price domain is \(\mathbb R\).

The Gaussian process prior is

\[
u\sim\mathcal{GP}(0,k),
\]

with an anisotropic RBF kernel

\[
k((t,x),(s,\xi))
=\sigma_f^2
\exp\left(
-\frac{(t-s)^2}{2\ell_t^2}
-\frac{(x-\xi)^2}{2\ell_x^2}
\right).
\]

The stacked operator is

\[
\mathcal A u=
\begin{pmatrix}
\mathcal L u(X^{\mathrm{int}})\\
u(X^{\mathrm{bd}})
\end{pmatrix}.
\]

The observation vector is

\[
y_A=
\begin{pmatrix}
0\\
h(X^{\mathrm{bd}})
\end{pmatrix}.
\]

The posterior mean at test points \(X^\ast\) is

\[
m_\ast
=K_{\ast A}
\left(K_{AA}+\Sigma_A\right)^{-1}y_A.
\]

The posterior covariance is

\[
C_\ast
=K_{\ast\ast}
-K_{\ast A}
\left(K_{AA}+\Sigma_A\right)^{-1}K_{A\ast}.
\]

No explicit matrix inverse should be used in code. Use Cholesky factorization and triangular solves.

---

## Model Operators

### Black--Scholes in log-price coordinates

For \(\lambda=0\), the log-price Black--Scholes operator is

\[
\mathcal L^{\mathrm{BS}}u
=
\partial_t u
+
\left(r-\frac12\sigma^2\right)\partial_x u
+
\frac12\sigma^2\partial_{xx}u
-ru.
\]

Equivalently,

\[
\mathcal L^{\mathrm{BS}}
=a+\partial_t+b\partial_x+c\partial_{xx},
\]

with

\[
a=-r,
\qquad
b=r-\frac12\sigma^2,
\qquad
c=\frac12\sigma^2.
\]

### Merton jump diffusion in log-price coordinates

For Merton jump diffusion, use the coefficients from the thesis:

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

The operator has the structure

\[
\mathcal L^{\mathrm{MJD}}
=a+\partial_t+b\partial_x+c\partial_{xx}+\lambda\mathcal J,
\]

where \(\mathcal J\) is the jump integral contribution. The precise analytic forms of the kernel identities for \(Lk\), \(kL'\), and \(LkL'\) should be implemented from the appendix.

The implementation must satisfy the sanity condition:

\[
\lambda=0
\quad\Longrightarrow\quad
\mathcal L^{\mathrm{MJD}}=\mathcal L^{\mathrm{BS}}.
\]

All jump-dependent terms must vanish exactly when \(\lambda=0\).

---

## Kernel Identities

The operator module must expose three matrix-valued functions:

```python
Lk(X, Y)
kLp(X, Y)
LkLp(X, Y)
```

with the following meaning:

\[
Lk(X,Y)_{ij}
=\mathcal L_{z_i}k(z_i,y_j),
\]

\[
kL'(X,Y)_{ij}
=\mathcal L_{y_j}k(z_i,y_j),
\]

\[
LkL'(X,Y)_{ij}
=\mathcal L_{z_i}\mathcal L_{y_j}k(z_i,y_j).
\]

Here each row of \(X\) and \(Y\) is a two-dimensional point:

\[
z=(t,x).
\]

The expected shapes are:

```python
X.shape == (n, 2)
Y.shape == (m, 2)
K.shape == (n, m)
```

### Black--Scholes limit check

For \(\lambda=0\), define

\[
\tau=t-s,
\qquad
\chi=x-\xi.
\]

Let

\[
k=k((t,x),(s,\xi)).
\]

Then the identities must reduce to:

\[
kL'
=
\left[
a+
\frac{\tau}{\ell_t^2}
+b\frac{\chi}{\ell_x^2}
+c\left(
\frac{\chi^2}{\ell_x^4}
-\frac1{\ell_x^2}
\right)
\right]k,
\]

\[
Lk
=
\left[
a-
\frac{\tau}{\ell_t^2}
-b\frac{\chi}{\ell_x^2}
+c\left(
\frac{\chi^2}{\ell_x^4}
-\frac1{\ell_x^2}
\right)
\right]k.
\]

Define

\[
P=
a-
\frac{\tau}{\ell_t^2}
-b\frac{\chi}{\ell_x^2}
+c\left(
\frac{\chi^2}{\ell_x^4}
-\frac1{\ell_x^2}
\right),
\]

and

\[
Q=
a+
\frac{\tau}{\ell_t^2}
+b\frac{\chi}{\ell_x^2}
+c\left(
\frac{\chi^2}{\ell_x^4}
-\frac1{\ell_x^2}
\right).
\]

Then

\[
LkL'
=
\left[
PQ
+
\frac1{\ell_t^2}
+
\frac{b^2}{\ell_x^2}
+
\frac{2c^2}{\ell_x^4}
-
\frac{4c^2\chi^2}{\ell_x^6}
\right]k.
\]

These formulas are mandatory unit-test targets for the Black--Scholes limit.

---

## Boundary Conditions

The implementation should support at least two boundary modes.

### 1. Terminal-only boundary

This is the clean baseline in log-price coordinates.

\[
X^{\mathrm{bd}}=X^T=\{(T,x_j)\}.
\]

The boundary function is the terminal payoff:

\[
h(T,x)=(e^x-K)^+
\]

for a call, and

\[
h(T,x)=(K-e^x)^+
\]

for a put.

The payoff should not include discounting. Discounting is already represented by the \(-ru\) term in the pricing operator.

### 2. Terminal plus artificial spatial boundaries

For numerical stability on a finite computational domain, optionally include

\[
X^{\mathrm{bd}}
=X^T\cup X^{x_{\min}}\cup X^{x_{\max}}.
\]

For a European call, possible approximate boundary conditions are:

Lower boundary:

\[
h(t,x_{\min})\approx 0.
\]

Upper boundary:

\[
h(t,x_{\max})\approx e^{x_{\max}}-Ke^{-r(T-t)}.
\]

This upper boundary is not the terminal payoff. It is a far-field approximation for a deep-in-the-money call. The terminal payoff remains undiscounted.

The architecture should allow switching between boundary modes without changing the posterior code.

---

## Folder Structure

Recommended project structure:

```text
option_gpr/
│
├── models/
│   ├── base.py
│   ├── black_scholes.py
│   └── merton.py
│
├── kernels/
│   ├── base.py
│   └── rbf.py
│
├── operators/
│   ├── base.py
│   ├── bs_log_operator.py
│   └── mjd_operator.py
│
├── posterior/
│   └── stacked_gp.py
│
├── grids/
│   ├── base.py
│   ├── uniform.py
│   ├── clustered.py
│   ├── random.py
│   └── sobol.py
│
├── payoffs/
│   ├── base.py
│   ├── call.py
│   └── put.py
│
├── hyperparams/
│   ├── objectives.py
│   └── optimize.py
│
├── benchmarks/
│   ├── black_scholes.py
│   ├── merton_series.py
│   └── monte_carlo.py
│
├── metrics/
│   └── errors.py
│
├── experiments/
│   ├── 01_bs_accuracy.py
│   ├── 02_merton_accuracy.py
│   ├── 03_grid_comparison.py
│   ├── 04_convergence.py
│   ├── 05_runtime_mc.py
│   └── 06_hyperparams.py
│
└── tests/
    ├── test_rbf_derivatives.py
    ├── test_kernel_identities_bs_limit.py
    ├── test_operator_symmetry.py
    ├── test_block_shapes.py
    ├── test_cholesky.py
    └── test_reference_prices.py
```

---

## Module Details

## `models/`

This module contains model parameter containers and reference model logic.

The GPR solver should not depend on reference prices. Reference formulas are only for testing and experiments.

### `models/base.py`

Define a common interface:

```python
class OptionModel:
    def coefficients(self):
        """Return operator coefficients a, b, c and jump parameters."""

    def payoff(self, x):
        """Return terminal payoff at log-price x."""

    def reference_price(self, t, x):
        """Optional analytical/semi-analytical price for tests."""
```

### `models/black_scholes.py`

Parameters:

```python
r: float
sigma: float
strike: float
maturity: float
option_type: str
```

Returns:

\[
a=-r,
\qquad
b=r-\frac12\sigma^2,
\qquad
c=\frac12\sigma^2.
\]

### `models/merton.py`

Parameters:

```python
r: float
sigma: float
jump_intensity: float   # lambda
jump_mean: float        # mu
jump_std: float         # delta
strike: float
maturity: float
option_type: str
```

Returns:

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

Use explicit names such as `jump_mean` and `jump_std` to avoid confusion with drift notation.

---

## `kernels/`

### `kernels/rbf.py`

The RBF kernel class should contain:

```python
class RBFKernel:
    def __init__(self, ell_t, ell_x, sigma_f):
        ...

    def K(self, X, Y):
        ...
```

Input arrays:

```python
X.shape == (n, 2)
Y.shape == (m, 2)
```

Output:

```python
K.shape == (n, m)
```

The implementation should be vectorized using NumPy broadcasting.

Optional helper methods:

```python
dtau(X, Y)      # t_i - s_j
dchi(X, Y)      # x_i - xi_j
base_terms(X,Y) # tau, chi, K
```

These helpers can prevent sign mistakes in the operator module.

---

## `operators/`

The operator module owns all formulas involving \(Lk\), \(kL'\), and \(LkL'\).

### `operators/base.py`

Define interface:

```python
class KernelOperator:
    def Lk(self, X, Y):
        raise NotImplementedError

    def kLp(self, X, Y):
        raise NotImplementedError

    def LkLp(self, X, Y):
        raise NotImplementedError
```

### `operators/bs_log_operator.py`

Implement the Black--Scholes log-price operator. This is useful as a separate baseline and debugging tool.

It should use the formulas from the Black--Scholes limit section above.

### `operators/mjd_operator.py`

Implement the full Merton jump diffusion kernel identities from the appendix.

Mandatory behavior:

- For \(\lambda=0\), output must exactly match `BSLogOperator` up to numerical tolerance.
- Jump terms should be isolated in helper functions where possible.
- The sign convention must be documented clearly:
  - `Lk(X,Y)` applies the operator to the first argument.
  - `kLp(X,Y)` applies the operator to the second argument.
  - `LkLp(X,Y)` applies the operator to both arguments.

---

## `posterior/`

### `posterior/stacked_gp.py`

This is the core solver.

Responsibilities:

1. Build the stacked observation vector \(y_A\).
2. Build the block covariance matrix \(K_{AA}\).
3. Add noise and jitter.
4. Compute Cholesky factorization.
5. Compute posterior mean and covariance at test points.
6. Provide access to log marginal likelihood for hyperparameter optimization.

### Block matrix

For interior points \(X^{\mathrm{int}}\) and boundary points \(X^{\mathrm{bd}}\), construct

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

The shapes must be:

```python
K_ii.shape == (N_int, N_int)
K_ib.shape == (N_int, N_bd)
K_bi.shape == (N_bd, N_int)
K_bb.shape == (N_bd, N_bd)
K_AA.shape == (N_int + N_bd, N_int + N_bd)
```

### Prediction matrix

For test points \(X^\ast\):

\[
K_{\ast A}
=
\begin{pmatrix}
kL'(X^\ast,X^{\mathrm{int}})
&
k(X^\ast,X^{\mathrm{bd}})
\end{pmatrix}.
\]

Shapes:

```python
K_star_int.shape == (N_star, N_int)
K_star_bd.shape == (N_star, N_bd)
K_star_A.shape == (N_star, N_int + N_bd)
```

### Observation vector

\[
y_A=
\begin{pmatrix}
0_{N_{\mathrm{int}}}\\
h(X^{\mathrm{bd}})
\end{pmatrix}.
\]

The function `boundary_values(X_bd)` should determine the boundary values based on whether a point lies on the terminal boundary or on artificial spatial boundaries.

### Noise matrix

Use separate noise levels:

\[
\Sigma_A
=
\begin{pmatrix}
\sigma_{\mathrm{int}}^2I_{N_{\mathrm{int}}} & 0\\
0 & \sigma_{\mathrm{bd}}^2I_{N_{\mathrm{bd}}}
\end{pmatrix}.
\]

Then add jitter:

\[
K_{\mathrm{reg}}=K_{AA}+\Sigma_A+\varepsilon I.
\]

Do not combine `noise_int`, `noise_bd`, and `jitter` into one parameter. They should remain separately configurable.

### Cholesky solves

Use:

```python
L = scipy.linalg.cho_factor(K_reg, lower=True)
alpha = scipy.linalg.cho_solve(L, y_A)
```

Prediction:

```python
mean = K_star_A @ alpha
```

Covariance:

```python
v = scipy.linalg.cho_solve(L, K_star_A.T)
cov = K_star_star - K_star_A @ v
```

The implementation should optionally return only the diagonal posterior variance if full covariance is too expensive.

---

## `grids/`

Grid generation must be independent of the solver.

All points should be returned in log-price coordinates with rows

\[
(t,x).
\]

### `GridSet`

Define a simple data container:

```python
@dataclass
class GridSet:
    X_int: np.ndarray
    X_bd: np.ndarray
    X_test: np.ndarray | None = None
    metadata: dict | None = None
```

### Grid types to support

#### Uniform log grid

Interior grid on

\[
[0,T)\times[x_{\min},x_{\max}].
\]

Terminal boundary:

\[
\{T\}\times[x_{\min},x_{\max}].
\]

#### Strike-centered grid

Cluster points around

\[
x_K=\log K.
\]

This is important because the payoff has a kink at \(S=K\), i.e. \(x=\log K\).

#### Random grid

Sample uniformly from the computational rectangle.

#### Sobol / quasi-random grid

Useful for more uniform high-dimensional sampling, though here the dimension is only two.

### Boundary mode

The grid generator should support:

```python
boundary_mode="terminal_only"
boundary_mode="terminal_and_spatial"
```

For `terminal_and_spatial`, include points at:

\[
x=x_{\min},
\qquad
x=x_{\max},
\qquad
0\le t\le T.
\]

---

## `payoffs/`

Payoff functions should be small and independent.

```python
class CallPayoff:
    def __init__(self, strike):
        self.strike = strike

    def from_x(self, x):
        S = np.exp(x)
        return np.maximum(S - self.strike, 0.0)

class PutPayoff:
    def from_x(self, x):
        S = np.exp(x)
        return np.maximum(self.strike - S, 0.0)
```

Terminal payoff is not discounted.

Boundary functions for artificial spatial boundaries can either live here or in a separate `boundary.py` module.

---

## `benchmarks/`

Benchmarks are for tests and experiments only. The GPR solver should not depend on them.

### `benchmarks/black_scholes.py`

Functions:

```python
bs_call_price(t, x, r, sigma, K, T)
bs_put_price(t, x, r, sigma, K, T)
bs_call_delta(t, x, r, sigma, K, T)  # optional
```

Internally set \(S=e^x\).

### `benchmarks/merton_series.py`

Functions:

```python
merton_call_price_series(t, x, model, n_terms=50)
merton_put_price_series(t, x, model, n_terms=50)
```

This is the main reference for Merton accuracy tests.

### `benchmarks/monte_carlo.py`

Functions:

```python
mc_price_merton(t, x, model, n_paths, seed=None)
```

Return a structured result:

```python
@dataclass
class MonteCarloResult:
    price: float
    standard_error: float
    ci_low: float
    ci_high: float
    runtime: float
```

Monte Carlo comparisons should always report standard errors or confidence intervals.

---

## `hyperparams/`

Hyperparameter optimization should wrap the solver rather than being inside the solver.

### Parameters

Optimize log-parameters:

\[
\vartheta=(
\log\ell_t,
\log\ell_x,
\log\sigma_f,
\log\sigma_{\mathrm{int}},
\log\sigma_{\mathrm{bd}}
).
\]

Then transform back with exponentials:

\[
\theta_i=e^{\vartheta_i}.
\]

This prevents negative lengthscales or variances.

### Objective 1: negative log marginal likelihood

Use

\[
\mathcal J(\theta)
=
\frac12 y_A^TK_\theta^{-1}y_A
+
\frac12\log\det K_\theta
+
\frac n2\log(2\pi).
\]

Compute using the Cholesky factorization:

```python
alpha = cho_solve(L, y_A)
quad = y_A @ alpha
logdet = 2 * np.sum(np.log(np.diag(cholesky_factor)))
nll = 0.5 * quad + 0.5 * logdet + 0.5 * n * np.log(2*np.pi)
```

### Objective 2: validation RMSE

When benchmark prices are available, optionally optimize

\[
\operatorname{RMSE}(m_\theta(X^{\mathrm{val}}),u_{\mathrm{ref}}(X^{\mathrm{val}})).
\]

This is not purely Bayesian, but useful for thesis experiments.

### Optimizer

Use SciPy Nelder--Mead:

```python
scipy.optimize.minimize(
    objective,
    x0=theta0_log,
    method="Nelder-Mead",
    options={"maxiter": ..., "xatol": ..., "fatol": ...},
)
```

The optimizer should return both log-parameters and transformed physical parameters.

---

## `metrics/`

Implement:

```python
rmse(pred, ref)
mae(pred, ref)
max_abs_error(pred, ref)
relative_l2_error(pred, ref)
pointwise_relative_error(pred, ref, eps=1e-8)
condition_number(K)
```

For option prices close to zero, relative pointwise errors can explode. Always report absolute errors as well.

Preferred thesis metrics:

\[
\mathrm{RMSE}
=\sqrt{\frac1n\sum_i(\hat u_i-u_i)^2},
\]

\[
\mathrm{MaxAE}=\max_i|\hat u_i-u_i|,
\]

\[
\mathrm{RelL2}
=\frac{\|\hat u-u\|_2}{\|u\|_2}.
\]

---

## `experiments/`

The experiments should use the core modules but not contain solver logic.

### `01_bs_accuracy.py`

Purpose:

- Verify the GPR solver in the Black--Scholes limit.
- Compare posterior mean against Black--Scholes closed-form prices.

Inputs:

- \(r,\sigma,K,T\)
- grid configuration
- kernel hyperparameters
- noise levels

Outputs:

- RMSE
- max absolute error
- relative L2 error
- runtime
- optional plots

### `02_merton_accuracy.py`

Purpose:

- Compare GPR prices against the Merton series solution.

Inputs:

- \(r,\sigma,\lambda,\mu,\delta,K,T\)

Outputs:

- RMSE
- max absolute error
- runtime
- results for several jump parameter settings

### `03_grid_comparison.py`

Purpose:

- Compare uniform log grid, strike-centered grid, random grid, and possibly Sobol grid.

Keep the same number of collocation and boundary points across grid choices.

Outputs:

- errors by grid type
- condition numbers by grid type
- runtime by grid type

### `04_convergence.py`

Purpose:

- Study error as a function of \(N_{\mathrm{int}}\) and \(N_{\mathrm{bd}}\).

Outputs:

- error vs number of collocation points
- runtime vs number of collocation points
- condition number vs number of collocation points

### `05_runtime_mc.py`

Purpose:

- Compare GPR runtime and accuracy with simple Monte Carlo.

Important framing:

- GPR has setup cost but gives a smooth pricing function over many evaluation points.
- Monte Carlo gives noisy pointwise estimates.

Monte Carlo output must include confidence intervals.

### `06_hyperparams.py`

Purpose:

- Test Nelder--Mead hyperparameter optimization.
- Compare manually chosen hyperparameters against optimized hyperparameters.

Outputs:

- initial parameters
- optimized parameters
- objective value
- validation/test error before and after optimization

### `07_delta_optional.py`

Purpose:

- Optional future Delta experiment.
- Compare GPR Delta against Black--Scholes Delta or finite-difference Merton Delta.

This experiment should only be implemented after the price solver is stable.

---

## Future Delta Extension

Design the architecture so Delta can be added later.

The posterior mean is

\[
m(z)=K_{zA}\alpha.
\]

In log-price coordinates,

\[
\frac{\partial m}{\partial x}(z)
=\frac{\partial K_{zA}}{\partial x}\alpha.
\]

The financial Delta is the derivative with respect to \(S\), not \(x\):

\[
\Delta(t,S)=\frac{\partial m}{\partial S}.
\]

Since \(S=e^x\),

\[
\frac{\partial}{\partial S}
=\frac1S\frac{\partial}{\partial x}
=e^{-x}\frac{\partial}{\partial x}.
\]

Therefore

\[
\Delta(t,x)
=e^{-x}\partial_xm(t,x).
\]

Future methods:

```python
build_dKdx_star_A(X_star, X_int, X_bd)
predict_dx(X_star)
predict_delta(X_star)
```

where

\[
\partial_xK_{\ast A}
=
\begin{pmatrix}
\partial_x kL'(X^\ast,X^{\mathrm{int}})
&
\partial_x k(X^\ast,X^{\mathrm{bd}})
\end{pmatrix}.
\]

Do not implement this until the price solver is verified.

---

## Unit Testing Plan

Unit tests are essential because the kernel identities are easy to get wrong.

### Test 1: RBF symmetry

Check:

\[
k(z,\tilde z)=k(\tilde z,z).
\]

### Test 2: Black--Scholes limit of Merton coefficients

For \(\lambda=0\), verify:

\[
a=-r,
\qquad
b=r-\frac12\sigma^2,
\qquad
c=\frac12\sigma^2.
\]

### Test 3: Merton operator equals Black--Scholes operator when \(\lambda=0\)

For random points \(X,Y\), check:

```python
MJDOperator(lambda=0).Lk(X,Y) == BSLogOperator.Lk(X,Y)
MJDOperator(lambda=0).kLp(X,Y) == BSLogOperator.kLp(X,Y)
MJDOperator(lambda=0).LkLp(X,Y) == BSLogOperator.LkLp(X,Y)
```

up to numerical tolerance.

### Test 4: One-sided operator symmetry

For constant coefficients and symmetric kernel, check:

\[
Lk(z,\tilde z)=kL'(\tilde z,z).
\]

### Test 5: finite-difference derivative check

Numerically approximate derivatives of the kernel and compare with analytic formulas.

For example:

\[
\partial_t k
\approx
\frac{k(t+h,x;s,\xi)-k(t-h,x;s,\xi)}{2h}.
\]

Then reconstruct \(Lk\) numerically and compare with closed form.

### Test 6: block matrix shapes

For

\[
N_{\mathrm{int}}=20,
\qquad
N_{\mathrm{bd}}=10,
\]

check:

```python
K_AA.shape == (30, 30)
y_A.shape == (30,)
K_star_A.shape == (N_star, 30)
```

### Test 7: Cholesky stability

Check that

\[
K_{AA}+\Sigma_A+\varepsilon I
\]

has a Cholesky factorization for reasonable noise and jitter values.

### Test 8: posterior interpolation sanity

With very small noise and enough boundary points, posterior mean at terminal boundary points should be close to the payoff.

### Test 9: Black--Scholes price accuracy smoke test

With a small grid and reasonable hyperparameters, the posterior mean should be roughly close to the Black--Scholes formula. This is not a strict unit test but a smoke test.

---

## Implementation Phases

### Phase 1: Black--Scholes baseline

Implement:

- `RBFKernel`
- `BlackScholesModel`
- `BSLogOperator`
- `StackedOperatorGP`
- terminal-only uniform log grid
- Black--Scholes benchmark

Goal:

- Produce posterior mean for Black--Scholes.
- Compare to closed-form Black--Scholes prices.
- Verify kernel identity tests.

### Phase 2: Merton jump diffusion

Implement:

- `MertonJumpDiffusionModel`
- `MJDOperator`
- Merton series benchmark

Goal:

- Verify \(\lambda=0\) reduction.
- Compare Merton GPR prices to Merton series prices.

### Phase 3: Grid experiments

Implement:

- strike-centered grids
- random grids
- optional Sobol grids
- terminal plus spatial boundary mode

Goal:

- Compare grid choices.
- Study convergence.

### Phase 4: Hyperparameter optimization

Implement:

- negative log marginal likelihood
- Nelder--Mead wrapper
- validation RMSE objective if needed

Goal:

- Compare fixed and optimized hyperparameters.

### Phase 5: Monte Carlo comparison

Implement:

- simple Monte Carlo for Black--Scholes and Merton
- runtime and confidence intervals

Goal:

- Compare GPR against a simple general-purpose method.

### Phase 6: Delta extension

Implement only if time allows:

- derivative kernel blocks for \(\partial_xK_{\ast A}\)
- posterior derivative prediction
- Delta transformation \(\Delta=e^{-x}\partial_xm\)

Goal:

- Compare against Black--Scholes Delta or finite-difference Merton Delta.

---

## Numerical Stability Guidelines

1. Always use Cholesky factorization.
2. Never use explicit matrix inverse.
3. Add jitter to the diagonal.
4. Keep interior noise and boundary noise separate.
5. Track condition numbers in experiments.
6. Use log-hyperparameters during optimization.
7. Start with small grids while debugging.
8. Verify \(\lambda=0\) before testing jumps.
9. Avoid very tiny lengthscales initially; they can produce ill-conditioned matrices.
10. Avoid zero noise in early experiments. Use small positive values.

---

## What the Architecture Should Enable

The final implementation should make it easy to answer the numerical questions from the thesis:

### Price accuracy

- Compare GPR prices against Black--Scholes closed-form prices.
- Compare GPR prices against Merton series prices.

### Black--Scholes limit

- Set \(\lambda=0\) in Merton and recover Black--Scholes kernel identities and prices.

### Collocation convergence

- Vary \(N_{\mathrm{int}}\) and \(N_{\mathrm{bd}}\).
- Measure RMSE, max error, runtime, and condition number.

### Grid dependence

- Compare uniform, strike-centered, random, and Sobol grids.

### Runtime comparison

- Compare GPR setup and evaluation time against Monte Carlo.

### Hyperparameter sensitivity

- Vary \(\ell_t,\ell_x,\sigma_f,\sigma_{\mathrm{int}},\sigma_{\mathrm{bd}}\).
- Optimize them using Nelder--Mead.

### Boundary condition choices

- Compare terminal-only boundary against terminal plus artificial spatial boundaries.

### Posterior uncertainty

- Evaluate posterior covariance or posterior variance.
- Use it descriptively, not necessarily as calibrated confidence intervals.

### Greeks

- Later add Delta without changing the posterior architecture.

---

## Summary of the Core API

The most important classes and methods should be:

```python
class RBFKernel:
    def K(self, X, Y): ...

class MertonJumpDiffusionModel:
    def coefficients(self): ...
    def payoff(self, x): ...

class MJDOperator:
    def Lk(self, X, Y): ...
    def kLp(self, X, Y): ...
    def LkLp(self, X, Y): ...

class StackedOperatorGP:
    def fit(self, X_int, X_bd): ...
    def predict(self, X_star, return_cov=False, return_var=False): ...
    def negative_log_marginal_likelihood(self): ...
```

The core solver should be model-agnostic as long as the supplied operator provides `Lk`, `kLp`, and `LkLp`.

That is the main architectural goal.
