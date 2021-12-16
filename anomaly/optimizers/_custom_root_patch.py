from functools import partial
import itertools
import operator

import jax
from jax import core
from jax import linear_util as lu
from jax.interpreters import ad
from jax.tree_util import tree_flatten, tree_unflatten, treedef_children, treedef_tuple
from jax._src.traceback_util import api_boundary
from jax._src.lax.control_flow import (
    _abstractify,
    _map,
    _initial_style_jaxpr,
    _check_tree,
    _RootTuple,
    _flatten,
    _split_root_args,
)


# This is the patch
def _stop_gradient_fun(f):
    """This is the only patch."""
    return f


@api_boundary
def custom_root(f, initial_guess, solve, tangent_solve):
    """Differentiably solve for a roots of a function.

    This is a low-level routine, mostly intended for internal use in JAX.
    Gradients of custom_root() are defined with respect to closed-over variables
    from the provided function ``f`` via the implicit function theorem:
    https://en.wikipedia.org/wiki/Implicit_function_theorem

    Args:
      f: function for which to find a root. Should accept a single argument,
        return a tree of arrays with the same structure as its input.
      initial_guess: initial guess for a zero of f.
      solve: function to solve for the roots of f. Should take two positional
        arguments, f and initial_guess, and return a solution with the same
        structure as initial_guess such that func(solution) = 0. In other words,
        the following is assumed to be true (but not checked)::

          solution = solve(f, initial_guess)
          error = f(solution)
          assert all(error == 0)

      tangent_solve: function to solve the tangent system. Should take two
        positional arguments, a linear function ``g`` (the function ``f``
        linearized at its root) and a tree of array(s) ``y`` with the same
        structure as initial_guess, and return a solution ``x`` such that
        ``g(x)=y``:

        - For scalar ``y``, use ``lambda g, y: y / g(1.0)``.
        - For vector ``y``, you could use a linear solve with the Jacobian, if
          dimensionality of ``y`` is not too large:
          ``lambda g, y: np.linalg.solve(jacobian(g)(y), y)``.

    Returns:
      The result of calling solve(f, initial_guess) with gradients defined via
      implicit differentiation assuming ``f(solve(f, initial_guess)) == 0``.
    """
    guess_flat, in_args_tree = tree_flatten((initial_guess,))
    guess_avals = tuple(_map(_abstractify, guess_flat))
    f_jaxpr, f_consts, out_tree = _initial_style_jaxpr(f, in_args_tree, guess_avals)

    (in_tree,) = treedef_children(in_args_tree)
    _check_tree("f", "initial_guess", out_tree, in_tree)

    solve_jaxpr, solve_consts, solution_tree = _initial_style_jaxpr(
        partial(solve, _stop_gradient_fun(f)), in_args_tree, guess_avals
    )
    _check_tree("solve", "initial_guess", solution_tree, in_tree)

    def linearize_and_solve(x, b):
        unchecked_zeros, f_jvp = jax.linearize(f, x)
        return tangent_solve(f_jvp, b)

    l_and_s_jaxpr, l_and_s_consts, out_tree = _initial_style_jaxpr(
        linearize_and_solve, treedef_tuple((in_tree,) * 2), guess_avals * 2
    )
    _check_tree("tangent_solve", "x", out_tree, in_tree)

    all_consts = [f_consts, solve_consts, l_and_s_consts]
    const_lengths = _RootTuple(*_map(len, all_consts))
    jaxprs = _RootTuple(f_jaxpr, solve_jaxpr, l_and_s_jaxpr)

    out_flat = _custom_root(const_lengths, jaxprs, *(_flatten(all_consts) + guess_flat))
    return tree_unflatten(out_tree, out_flat)


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _custom_root(const_lengths, jaxprs, *args):
    params, initial_guess = _split_root_args(args, const_lengths)
    solution = core.jaxpr_as_fun(jaxprs.solve)(*(params.solve + initial_guess))
    return solution


@_custom_root.defjvp
def _root_jvp(const_lengths, jaxprs, primals, tangents):
    params, _ = _split_root_args(primals, const_lengths)
    solution = _custom_root(const_lengths, jaxprs, *primals)

    params_dot, _ = _split_root_args(tangents, const_lengths)

    # F(m, u) = 0      # system of equations in u, parameterized by m
    #                  # solution is u*(m) defined in a neighborhood
    # F(m, u*(m)) = 0  # satisfied in a neighborhood
    #
    # ∂_0 F(m, u*(m)) + ∂_1 F(m, u*(m)) ∂ u*(m) = 0       # implied by line above
    # ∂ u*(m) = - (∂_1 F(m, u*(m)))^{-1} ∂_0 F(m, u*(m))  # rearrange
    #
    # ∂ u*(m)[v] = - (∂_1 F(m, u*(m)))^{-1} [∂_0 F(m, u*(m))[v]]  # jvp

    f = core.jaxpr_as_fun(jaxprs.f)
    linearize_and_solve = partial(core.jaxpr_as_fun(jaxprs.l_and_s), *params.l_and_s)
    f_at_solution = lambda *params: f(*itertools.chain(params, solution))
    _, rhs = ad.jvp(lu.wrap_init(f_at_solution)).call_wrapped(params.f, params_dot.f)
    solution_dot = _map(
        operator.neg, linearize_and_solve(*itertools.chain(solution, rhs))
    )

    return solution, solution_dot
