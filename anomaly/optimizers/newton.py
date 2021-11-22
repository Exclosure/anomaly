"""Newton and related methods for optimization.

TODO:
  - [ ] Create a fixed-iteration Newton-Raphson using JAX
        control-flow.
"""
import logging
from typing import Callable, Optional

import jax
import jax.numpy as jnp


MAX_NEWTON_RAPHSON_ITERATIONS = 100
NEWTON_RAPHSON_EPS = 1e-9
NEWTON_RAPHSON_EPS_64 = 1e-16


def newton_raphson(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    eps: Optional[float]=None,
    max_iter: int=MAX_NEWTON_RAPHSON_ITERATIONS,
) -> jnp.ndarray:
  """Newton-Raphson method for root-finding.

  Parameters:
    f: The function to find the zero of.
    x0: The starting point for Newton-Raphson iteration.
    eps: The stopping value. By default, we choose eps=1e-16
      if the starting point is 64-bit, and 1e-9 otherwise.
    max_iter: Maximum number of iterations in the updates.

  Returns:
    x_best: The argument closest to zero in the max-norm.
  """
  logger = logging.getLogger(__name__)

  if eps is None:
    if jnp.asarray(x0).dtype == jnp.float64:
      eps = NEWTON_RAPHSON_EPS_64
    else:
      eps = NEWTON_RAPHSON_EPS

  x0 = jnp.asarray(x0)

  f_jac = jax.jacfwd(f)
  best_gap = jnp.inf
  best_x = x0

  for i in range(max_iter):
    fx0 = jnp.array(f(x0)).reshape((-1, 1))
    gap = jnp.abs(fx0).max()

    logger.debug(f"Iteration {i}: best gap {best_gap}.")

    if gap < best_gap:
      best_x = x0
      best_gap = gap

    if gap < eps:
      # Happy case, convergence reached
      logger.info(
          f"Convergence reached after {i+1} steps. "
          f"Optimality gap is {best_gap}."
      )
      break

    # Compute Jacobian at x0, reshape into matrix
    dfx0 = f_jac(x0).reshape((-1, len(fx0)))

    # Compute gradient step
    if len(fx0) == 1:
      # Scalar case
      step = fx0 / dfx0
    else:
      # Requires linear solve
      step = jnp.linalg.solve(dfx0, fx0).reshape(x0.shape)

    # Check for numerical issues with zero gradients
    step = jnp.where(jnp.isfinite(step), step, 0.0)
    step_size = jnp.max(jnp.abs(step))
    if step_size < eps:
      # Stalled: too small of an update
      logger.warning(
          f"Newton-Raphson method stalled at "
          f"iteration {i} with step size {step_size}. "
          f"Current gap is {best_gap}"
      )
      break

    # Update points and continue
    x0 = x0 - step
  else:
    logger.warning(
        f"Convergence not reached after {max_iter} iterations; "
        f"gap is {best_gap}."
    )
  return best_x
