"""Utilities and constants."""
from functools import wraps
import math
from typing import Callable, Tuple

import jax
import jax.numpy as jnp


TWOPI = 6.283185307179586

# Earth's gravitational constant.
#
# Accurate to 8 significant figures.
#
# References:
#    [1] https://iau-a3.gitlab.io/NSFA/NSFA_cbe.html#GME2009
#    [2] Ries, J. C., Eanes, R. J., Shum, C. K., and Watkins, M. M.,
#        1992, "Progress in the Determination of the Gravitational
#        Coefficient of the Earth," Geophys. Res. Lett., 19(6),
#        pp. 529-531.
#    [3] Vallado, D.A. Fundamentals of Astrophysics and Applications,
#        2013.
MU_METERS_CUBED_PER_SECOND_SQUARED = 3.986004415e5  #  Units: km^3/s^2


def clip_to_rads(f: Callable[..., jnp.ndarray]) -> Callable[..., jnp.ndarray]:
    """Clips the output of a function to [0, 2*pi].

    Used when the final call is a Newton-Raphson iteration
    that may result in out-of-bounds radians.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        return jnp.remainder(f(*args, **kwargs), TWOPI)

    return wrapper


def signed_arccos(x: float, sign: float) -> float:
    """Return the quadrant-aware arccosine.

    If ``sign >= 0``, the result is between zero and π.

    If ``sign < 0``, the result is between π and 2π.

    This is used as an alternative to ``jnp.arctan2``
    when it is more convenient to check a sign of
    a value to determine the quadrant.

    Arguments:
        x:
            The argument of arccos.
        sign:
            If ``sign >= 0``, the result is between zero and π.
            If ``sign < 0``, the result is between π and 2π.

    Returns:
        The quandrant-aware arccosine of ``x``.
    """
    return jax.lax.cond(
        sign >= 0,
        lambda t: jnp.arccos(t),
        lambda t: TWOPI - jnp.arccos(t),
        x,
    )


def norm_and_norm_squared(x: jnp.ndarray) -> Tuple[float, float]:
    """Return the norm and squared norm of a vector.

    Arguments:
        x:
            The vector to compute the norm.

    Returns:
        A pair ``(x_norm, x_norm ** 2)``.
    """
    r2 = jnp.sum(x ** 2)
    r = jnp.sqrt(r2)
    return (r, r2)


def rad2deg(rad: float) -> jnp.ndarray:
    """Convert radians to degrees.

    Unlike `jnp.rad2deg`, the result is normalized between
    zero and 360.
    """
    return jnp.mod(jnp.rad2deg(rad), 360)
