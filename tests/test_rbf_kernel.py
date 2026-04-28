import numpy as np
import pytest

from option_gpr.kernels import RBFKernel


def test_rbf_kernel_matrix_shape() -> None:
    kernel = RBFKernel(ell_t=0.5, ell_x=1.2, sigma_f=2.0)
    X = np.array([[0.0, 3.9], [0.5, 4.0], [1.0, 4.1]])
    Y = np.array([[0.25, 3.8], [0.75, 4.2]])

    assert kernel.K(X, Y).shape == (3, 2)


def test_rbf_kernel_symmetry() -> None:
    kernel = RBFKernel(ell_t=0.7, ell_x=1.1, sigma_f=1.5)
    X = np.array([[0.0, 3.9], [0.4, 4.0], [0.8, 4.2]])
    Y = np.array([[0.2, 3.8], [0.6, 4.1]])

    np.testing.assert_allclose(kernel.K(X, Y), kernel.K(Y, X).T)


def test_rbf_kernel_diagonal_equals_signal_variance() -> None:
    sigma_f = 1.7
    kernel = RBFKernel(ell_t=0.9, ell_x=1.3, sigma_f=sigma_f)
    X = np.array([[0.0, 3.9], [0.5, 4.0], [1.0, 4.1]])

    np.testing.assert_allclose(np.diag(kernel.K(X, X)), sigma_f**2)


def test_rbf_kernel_positive_semidefinite_smoke() -> None:
    kernel = RBFKernel(ell_t=0.8, ell_x=1.4, sigma_f=1.0)
    X = np.array([[0.0, 3.8], [0.25, 3.9], [0.5, 4.0], [0.75, 4.1]])

    eigvals = np.linalg.eigvalsh(kernel.K(X, X))

    assert np.all(eigvals >= -1e-10)


def test_rbf_kernel_pairwise_helpers() -> None:
    kernel = RBFKernel(ell_t=1.0, ell_x=1.0, sigma_f=1.0)
    X = np.array([[0.0, 3.0], [1.0, 4.0]])
    Y = np.array([[0.25, 2.5], [0.75, 4.5], [1.25, 5.0]])

    expected_tau = np.array([[-0.25, -0.75, -1.25], [0.75, 0.25, -0.25]])
    expected_chi = np.array([[0.5, -1.5, -2.0], [1.5, -0.5, -1.0]])

    np.testing.assert_allclose(kernel.tau(X, Y), expected_tau)
    np.testing.assert_allclose(kernel.chi(X, Y), expected_chi)


@pytest.mark.parametrize(
    ("ell_t", "ell_x", "sigma_f"),
    [
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 1.0),
    ],
)
def test_rbf_kernel_requires_positive_hyperparameters(
    ell_t: float, ell_x: float, sigma_f: float
) -> None:
    with pytest.raises(ValueError):
        RBFKernel(ell_t=ell_t, ell_x=ell_x, sigma_f=sigma_f)


@pytest.mark.parametrize(
    "bad_points",
    [
        np.array([0.0, 1.0]),
        np.array([[0.0], [1.0]]),
        np.array([[0.0, 1.0, 2.0]]),
    ],
)
def test_rbf_kernel_rejects_invalid_point_shapes(bad_points: np.ndarray) -> None:
    kernel = RBFKernel(ell_t=1.0, ell_x=1.0, sigma_f=1.0)
    good_points = np.array([[0.0, 1.0]])

    with pytest.raises(ValueError):
        kernel.K(bad_points, good_points)

    with pytest.raises(ValueError):
        kernel.K(good_points, bad_points)
