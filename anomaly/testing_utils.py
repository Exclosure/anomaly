"""Utilities used in testing."""
from numpy.testing import assert_allclose
import jax
from jax import numpy as jnp


def assert_trees_allclose(
    x: jnp.ndarray, y: jnp.ndarray, rtol: float = 1e-7, atol: float = 0.0
):
    """Check that two trees have almost equal elements.

    Arguments:
        x:
            The first tree.
        y:
            The second tree.
        rtol:
            Relative tolerance.
        atol:
            Absolute tolerance. If zero, no absolute tolerance
            is checked.

    Raises:
        AssertionError:
            When the tree structure does not match
    """
    x_flat, x_tree = jax.tree_flatten(x)
    y_flat, y_tree = jax.tree_flatten(y)
    if x_tree != y_tree:
        raise AssertionError(f"x and y do not have the same structure:\nx={x}\ny={y}")

    good = "OK"

    def check(a, b):
        try:
            assert_allclose(a, b, rtol=rtol, atol=atol)
        except AssertionError as e:
            return str(e)
        return good

    error_flat = jax.tree_map(check, x_flat, y_flat)
    has_error = any(e != good for e in error_flat)
    if has_error:
        error_tree = jax.tree_unflatten(x_tree, error_flat)
        raise AssertionError(f"Detected unequal arrays: {error_tree}")
