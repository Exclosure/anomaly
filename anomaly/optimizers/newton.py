"""Newton and related methods for optimization.

TODO:
  - [ ] Use jax.lax.custom_root to make our Newton method
        differentiable.

"""
from typing import Callable, Tuple, NamedTuple

import jax
from jax.lax import stop_gradient
from jax import lax
import jax.numpy as jnp

from anomaly.optimizers._custom_root_patch import custom_root


MAX_NEWTON_RAPHSON_ITERATIONS = 100
NEWTON_RAPHSON_EPS = 1e-9
NEWTON_RAPHSON_EPS_64 = 1e-16


class _NewtonState(NamedTuple):
    x: jnp.ndarray
    fx: jnp.ndarray
    dfx: jnp.ndarray
    best_x: jnp.ndarray
    best_dfx: jnp.ndarray
    best_gap: float
    iteration: int
    step_size: float


def newton_raphson(
    f: Callable,
    x0: jnp.ndarray,
    eps: float = NEWTON_RAPHSON_EPS_64,
    max_iter: int = MAX_NEWTON_RAPHSON_ITERATIONS,
) -> jnp.ndarray:
    """Newton-Raphson method for root-finding.

    Args:
      f: The function to find zeros. For consistent updates,
         we require ``f(x0).shape == x0.shape``.
      x0: The starting point for Newton-Raphson iteration.
      eps: The stopping value. By default, we choose eps=1e-16
        if the starting point is 64-bit, and 1e-9 otherwise.
      max_iter: Maximum number of iterations in the updates.

    Returns:
      x_best: The argument closest to zero in the max-norm.
    """
    tangent_solve = lambda g, t: linear_solve(jax.jacobian(g)(t), t)
    solve = lambda f, x0: newton_raphson_solve(f, x0, eps, max_iter)
    return custom_root(f, stop_gradient(x0), solve, tangent_solve)


def newton_raphson_solve(
    f,
    x0: jnp.ndarray,
    eps: float = None,
    max_iter: int = MAX_NEWTON_RAPHSON_ITERATIONS,
):
    if (eps is None) and (x0.dtype == jnp.float64):
        _eps = NEWTON_RAPHSON_EPS_64
    elif eps is None:
        _eps = NEWTON_RAPHSON_EPS
    else:
        _eps = eps

    f_jac = jax.jacfwd(f)
    fx0 = f(x0)
    dfx0 = f_jac(x0)
    initial_gap = _gap(fx0)
    initial_state = _NewtonState(
        x=x0,
        fx=fx0,
        dfx=dfx0,
        best_x=x0,
        best_dfx=dfx0,
        best_gap=initial_gap,
        iteration=0,
        step_size=float("inf"),
    )

    def loop_continue(state: _NewtonState) -> bool:
        """Returns False when the loop should stop."""
        return stop_gradient(
            lax.cond(
                (state.iteration < max_iter)
                & (state.best_gap > _eps)
                & (state.step_size > _eps / 2),
                lambda _: True,
                lambda _: False,
                None,
            )
        )

    def update_fun(state: _NewtonState) -> _NewtonState:
        return stop_gradient(_newton_raphson_step(f, f_jac, state))

    final_state = stop_gradient(
        lax.while_loop(
            cond_fun=loop_continue,
            body_fun=update_fun,
            init_val=initial_state,
        )
    )

    return final_state.best_x


def _newton_raphson_step(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    f_jac: Callable[[jnp.ndarray], jnp.ndarray],
    state: _NewtonState,
) -> _NewtonState:
    x0, fx0, dfx0, _, _, _, _, step_size = state
    x, step_size = _take_step(x0, fx0, dfx0)
    fx = f(x)
    dfx = f_jac(x)
    return _update_state(x, fx, dfx, step_size, state)


def _update_state(x, fx, dfx, step_size, prev_state) -> _NewtonState:
    x0, fx0, dfx0, best_x, best_dfx, best_gap, it, step_size = prev_state
    gap = _gap(fx)
    (best_x, best_gap, best_dfx) = lax.cond(
        gap < best_gap,
        lambda _: (x.reshape(best_x.shape), gap, dfx.reshape(best_dfx.shape)),
        lambda _: (best_x, best_gap, best_dfx),
        None,
    )
    return _NewtonState(
        x=x.reshape(x0.shape),
        fx=fx.reshape(fx0.shape),
        dfx=dfx.reshape(dfx0.shape),
        best_x=best_x,
        best_dfx=best_dfx,
        best_gap=best_gap,
        iteration=lax.add(it + 1, 1),
        step_size=step_size,
    )


def _take_step(x0, fx0, dfx0) -> Tuple[jnp.ndarray, float]:
    step = linear_solve(dfx0, fx0)
    # step = jnp.where(jnp.isfinite(step), step, 0.0)
    step_size = _gap(step)
    x = x0 - step
    return x, step_size


def linear_solve(A: jnp.ndarray, b: jnp.ndarray):
    # TODO: Are there better ways to do a switch on the
    # dimension?
    return lax.cond(
        b.size == 1,
        # The max() helps the JIT understand
        # that we get the same output size for both
        # branches.
        lambda _: jnp.divide(b, A.max()).reshape(b.shape),
        lambda _: jnp.linalg.solve(
            A.reshape((-1, b.size)), b.reshape((b.size, 1))
        ).reshape(b.shape),
        None,
    )


def _gap(fx: jnp.ndarray) -> float:
    return jnp.abs(fx).max()


def newton_1d(f, x0, eps=None, max_iter=100):
    """Simpler implementation of Newton-Raphson for 1d problems."""
    if (eps is None) and (x0.dtype == jnp.float64):
        _eps = NEWTON_RAPHSON_EPS_64
    elif eps is None:
        _eps = NEWTON_RAPHSON_EPS
    else:
        _eps = eps

    def cond(state):
        it, _, fx, dfx = state
        return (jnp.abs(fx) > _eps) & (it < max_iter) & (jnp.abs(dfx) > _eps)

    def body(state):
        it, x0, fx0, dfx0 = state
        x = x0 - fx0 / dfx0
        fx, dfx = jax.value_and_grad(f)(x)
        return (it + 1, x, fx, dfx)

    fx0, dfx0 = jax.value_and_grad(f)(x0)
    initial_state = (0, x0, fx0, dfx0)
    final_state = lax.while_loop(cond, body, initial_state)
    return final_state[1]
