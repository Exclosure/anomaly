import jax
from jax import numpy as jnp
import numpy as np
from scipy.linalg.special_matrices import toeplitz

from anomaly.optimizers.newton import newton_raphson


def one_dim_test(x: jnp.ndarray) -> jnp.ndarray:
  # Has roots at 1, 2, -2
  return jnp.power(2*x - 2, 3) * jnp.power(jnp.power(x, 2)-4, 2)

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


def test_newton_raphson_1d():
  # Note: The optimization to has trouble
  # with the flatness near the root at 1,
  # resulting in lower-quality solutions.

  for x0,y0 in test_points:
    result = newton_raphson(one_dim_test, x0)
    np.testing.assert_almost_equal(result, y0, decimal=5)

def test_newton_raphson_diagonal():
  # This is a diagonal test -- mixed partials are zero
  result = newton_raphson(one_dim_test, x0=test_points[:, 0])
  np.testing.assert_array_almost_equal(result, test_points[:, 1])

def test_newton_raphson_rotated():
  # Here, we rotate the inputs and outputs in our 1D method
  # to make a more complex optimization.
  transform = jnp.array(
    toeplitz([0.9 ** i for i in range(test_points.shape[0])])
  )
  transform_inv = jnp.linalg.inv(transform)
  complex_method = lambda x: transform @ one_dim_test(transform_inv @ x)
  result = newton_raphson(complex_method, test_points[:, 0])
  np.testing.assert_array_almost_equal(
    complex_method(result), jnp.zeros((test_points.shape[0],)))
