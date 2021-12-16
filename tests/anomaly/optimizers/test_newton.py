from functools import partial
from unittest.mock import patch

import jax
from jax import numpy as jnp
import jax.core
import numpy as np
from scipy.linalg.special_matrices import toeplitz

from anomaly.optimizers.newton import newton_raphson, newton_1d, linear_solve
from jax import config


def one_dim_test(x: jnp.ndarray) -> jnp.ndarray:
    # Has roots at 1, 2, -2
    return jnp.power(2 * x - 2, 3) * jnp.power(jnp.power(x, 2) - 4, 2)


test_points = jnp.array(
    [
        [-5.1, -2],
        [-3, -2],
        [-2.1, -2],
        [-1, 2],
        [0, 1],
        [0.5, 1],
        [1.1, 1],
        [2.2, 2],
        [3, 2],
    ]
)

transform = jnp.array(toeplitz([0.9 ** i for i in range(test_points.shape[0])]))
transform_inv = jnp.linalg.inv(transform)
complex_method = lambda x: transform @ one_dim_test(transform_inv @ x)


def test_newton_1d():
    # Note: The optimization to has trouble
    # with the flatness near the root at 1,
    # resulting in lower-quality solutions.
    optimizer = lambda x0: newton_1d(one_dim_test, x0)
    for x0, y0 in test_points:
        result = optimizer(x0)
        np.testing.assert_almost_equal(result, y0, decimal=5)


def test_newton_raphson_diagonal():
    # This is a diagonal test -- mixed partials are zero
    result = newton_raphson(one_dim_test, test_points[:, 0])
    np.testing.assert_array_almost_equal(result, test_points[:, 1])


def test_newton_raphson_rotated():
    # Here, we rotate the inputs and outputs in our 1D method
    # to make a more complex optimization.

    result = newton_raphson(complex_method, test_points[:, 0])
    np.testing.assert_array_almost_equal(
        complex_method(result), jnp.zeros((test_points.shape[0],))
    )


def test_newton_raphson_vmap():
    newton_vmap = jax.vmap(
        lambda f, x0: newton_raphson(f, x0),
        in_axes=(
            None,
            0,
        ),
        out_axes=0,
    )
    result = newton_vmap(one_dim_test, test_points[:, 0])
    np.testing.assert_array_almost_equal(result, test_points[:, 1])


def sqrt_cubed_newton(x):
    implicit = lambda y: y ** 2 - x ** 3
    # Note: use x as starting point
    return newton_raphson(implicit, x)


def test_newton_derivatives_scalar():
    x = 5.0
    v = sqrt_cubed_newton(x)
    g = jax.jacfwd(sqrt_cubed_newton)(x)
    np.testing.assert_allclose(v, x ** 1.5)
    np.testing.assert_allclose(g, 1.5 * x ** 0.5)


def test_newton_derivatives_scalar_array():
    x = jnp.array([5.0])
    v = sqrt_cubed_newton(x)
    g = jax.jacfwd(sqrt_cubed_newton)(x)
    np.testing.assert_allclose(v, x ** 1.5)
    np.testing.assert_allclose(g, 1.5 * x[:, jnp.newaxis] ** 0.5)


def test_newton_derivatives_vector():
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    v = sqrt_cubed_newton(x)
    g = jax.jacfwd(sqrt_cubed_newton)(x)
    np.testing.assert_allclose(v, x ** 1.5)
    np.testing.assert_allclose(g, 1.5 * jnp.diag(x) ** 0.5)


def test_newton_derivatives_vmap():
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    v = jax.vmap(sqrt_cubed_newton)(x)
    g = jax.vmap(jax.jacfwd(sqrt_cubed_newton))(x)
    np.testing.assert_allclose(v, x ** 1.5)
    np.testing.assert_allclose(g, 1.5 * x ** 0.5)


def test_linear_solve():
    A, b = jnp.array(1.0), jnp.array(1.0)
    np.testing.assert_almost_equal(linear_solve(A, b), 1.0)

    A2, b2 = jnp.array([[2.0, 0.0], [0.0, 1.0]]), jnp.array([2.0, 1.0])
    np.testing.assert_array_almost_equal(linear_solve(A2, b2), jnp.array([1.0, 1.0]))
    np.testing.assert_array_almost_equal(
        linear_solve(A2, b2.reshape((-1, 1))), jnp.array([[1.0, 1.0]]).T
    )

    jit_solve = jax.jit(linear_solve)
    np.testing.assert_array_almost_equal(jit_solve(A2, b2), jnp.array([1.0, 1.0]))
    np.testing.assert_array_almost_equal(
        jit_solve(A2, b2.reshape((-1, 1))), jnp.array([[1.0, 1.0]]).T
    )


if __name__ == "__main__":
    import pytest

    pytest.main()
