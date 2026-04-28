# Math Notes for Option GPR Implementation

This file contains the mathematical conventions and formulas that the implementation should treat as source-of-truth. It is intentionally shorter and more formula-focused than `docs/architecture.md`.

## 1. Coordinates

The solver works in log-price coordinates:

\[
x = \log S.
\]

All points are space-time points

\[
z=(t,x), \qquad \tilde z=(s,\xi),
\]

stored as rows of NumPy arrays with shape `(n, 2)`:

```text
X[:, 0] = t
X[:, 1] = x
```

When a stock price is needed for payoff or benchmark formulas, use

\[
S=e^x.
\]

## 2. RBF kernel

The anisotropic RBF kernel is

\[
k((t,x),(s,\xi))
=
\sigma_f^2
\exp\left(
-\frac{(t-s)^2}{2\ell_t^2}
-\frac{(x-\xi)^2}{2\ell_x^2}
\right).
\]

Define

\[
\tau=t-s,
\qquad
\chi=x-\xi.
\]

Then

\[
k=\sigma_f^2\exp\left(-\frac{\tau^2}{2\ell_t^2}-\frac{\chi^2}{2\ell_x^2}\right).
\]

Useful derivatives:

\[
\partial_t k=-\frac{\tau}{\ell_t^2}k,
\qquad
\partial_s k=\frac{\tau}{\ell_t^2}k.
\]

\[
\partial_x k=-\frac{\chi}{\ell_x^2}k,
\qquad
\partial_\xi k=\frac{\chi}{\ell_x^2}k.
\]

\[
\partial_{xx} k
=
\partial_{\xi\xi} k
=
\left(\frac{\chi^2}{\ell_x^4}-\frac1{\ell_x^2}\right)k.
\]

## 3. Black--Scholes log-price operator

The stock-price Black--Scholes operator is

\[
\mathcal L_S F
=
\partial_t F+rS\partial_SF+\frac12\sigma^2S^2\partial_{SS}F-rF.
\]

Under the coordinate transform

\[
u(t,x)=F(t,e^x),
\]

this becomes

\[
\mathcal L_x u
=
\partial_tu+
\left(r-\frac12\sigma^2\right)\partial_xu+
\frac12\sigma^2\partial_{xx}u-ru.
\]

Equivalently,

\[
\mathcal L=a+\partial_t+b\partial_x+c\partial_{xx},
\]

with

\[
a=-r,
\qquad
b=r-\frac12\sigma^2,
\qquad
c=\frac12\sigma^2.
\]

## 4. Merton jump diffusion coefficients

For Merton jump diffusion, use the thesis convention

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

Here:

- \(\lambda\) is jump intensity.
- \(\mu\) is mean jump size in log-space.
- \(\delta\) is jump-size standard deviation.

Code names should be:

```text
jump_intensity = lambda
jump_mean      = mu
jump_std       = delta
```

The full Merton operator includes jump integral terms. The exact Merton kernel identities should be implemented from the thesis appendix.

Mandatory limit:

\[
\lambda=0
\quad\Rightarrow\quad
\mathcal L^{\mathrm{MJD}}=\mathcal L^{\mathrm{BS}}.
\]

All jump-dependent terms must vanish when `jump_intensity == 0`.

## 5. Black--Scholes kernel identities

These are mandatory unit-test targets. They verify the differential part of the Merton identities in the \(\lambda=0\) limit.

Let

\[
\tau=t-s,
\qquad
\chi=x-\xi.
\]

Let

\[
D(\chi)
=
\frac{\chi^2}{\ell_x^4}-\frac1{\ell_x^2}.
\]

### 5.1 Operator on the second argument: `kLp`

\[
kL'
=
\left[
 a
 +\frac{\tau}{\ell_t^2}
 +b\frac{\chi}{\ell_x^2}
 +cD(\chi)
\right]k.
\]

Define

\[
Q
=
 a
 +\frac{\tau}{\ell_t^2}
 +b\frac{\chi}{\ell_x^2}
 +cD(\chi).
\]

Then

\[
kL'=Qk.
\]

### 5.2 Operator on the first argument: `Lk`

\[
Lk
=
\left[
 a
 -\frac{\tau}{\ell_t^2}
 -b\frac{\chi}{\ell_x^2}
 +cD(\chi)
\right]k.
\]

Define

\[
P
=
 a
 -\frac{\tau}{\ell_t^2}
 -b\frac{\chi}{\ell_x^2}
 +cD(\chi).
\]

Then

\[
Lk=Pk.
\]

### 5.3 Operator on both arguments: `LkLp`

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

The terms proportional to

\[
\frac{2bc\chi}{\ell_x^4}
\]

cancel outside the product \(PQ\). If an expanded implementation contains such terms outside \(PQ\), check the algebra carefully.

## 6. Symmetry check

For the constant-coefficient Black--Scholes log-price operator and symmetric RBF kernel,

\[
Lk(z,\tilde z)=kL'(\tilde z,z).
\]

In matrix notation this means

```python
operator.Lk(X, Y) == operator.kLp(Y, X).T
```

up to numerical tolerance.

This is a critical sign-check.

## 7. Stacked operator posterior

Interior observations impose

\[
\mathcal L u(X^{\mathrm{int}})=0.
\]

Boundary observations impose

\[
u(X^{\mathrm{bd}})=h(X^{\mathrm{bd}}).
\]

The stacked observation vector is

\[
y_A=
\begin{pmatrix}
0_{N_{\mathrm{int}}}\\
h(X^{\mathrm{bd}})
\end{pmatrix}.
\]

The stacked covariance matrix is

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

For test points \(X^\ast\),

\[
K_{\ast A}
=
\begin{pmatrix}
kL'(X^\ast,X^{\mathrm{int}})
&
k(X^\ast,X^{\mathrm{bd}})
\end{pmatrix}.
\]

The posterior mean is

\[
m_\ast
=K_{\ast A}(K_{AA}+\Sigma_A)^{-1}y_A.
\]

The posterior covariance is

\[
C_\ast
=K_{\ast\ast}
-K_{\ast A}(K_{AA}+\Sigma_A)^{-1}K_{A\ast}.
\]

In code, never compute the inverse explicitly. Use Cholesky factorization.

## 8. Noise and jitter

Use

\[
\Sigma_A
=
\begin{pmatrix}
\sigma_{\mathrm{int}}^2 I_{N_{\mathrm{int}}} & 0\\
0 & \sigma_{\mathrm{bd}}^2 I_{N_{\mathrm{bd}}}
\end{pmatrix}.
\]

Then add jitter:

\[
K_{\mathrm{reg}}=K_{AA}+\Sigma_A+\varepsilon I.
\]

Keep these parameters separate:

```text
noise_int
noise_bd
jitter
```

## 9. Payoff and boundary conditions

Terminal payoff is not discounted.

Call payoff:

\[
h(T,x)=(e^x-K)^+.
\]

Put payoff:

\[
h(T,x)=(K-e^x)^+.
\]

Discounting is already represented by the \(-ru\) term in the operator.

Optional artificial spatial boundaries for a call:

Lower boundary:

\[
h(t,x_{\min})\approx 0.
\]

Upper boundary:

\[
h(t,x_{\max})\approx e^{x_{\max}}-Ke^{-r(T-t)}.
\]

These spatial boundaries are numerical far-field approximations, not the terminal payoff.

## 10. Delta extension, optional later

The posterior mean is

\[
m(z)=K_{zA}\alpha.
\]

Then

\[
\partial_x m(z)=\partial_x K_{zA}\alpha.
\]

The financial Delta is derivative with respect to stock price \(S\), not log-price \(x\):

\[
\Delta(t,S)=\partial_S m(t,S).
\]

Since

\[
\partial_S=e^{-x}\partial_x,
\]

we have

\[
\Delta(t,x)=e^{-x}\partial_xm(t,x).
\]

Do not implement Delta until price predictions and kernel identity tests are stable.
