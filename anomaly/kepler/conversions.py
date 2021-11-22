"""Conversions for Keplerian orbit parameters.

For conversions between anomalies, we use Newton-Raphson
iteration over the implicit relations between Keplerian
parameters.

Abbreviations:

  We use the following abbreviations in the code.

    M = mean anomaly
    e = eccentricity
    E = eccentric anomaly
    theta = true anomaly
    a = semi-major axis
    b = semi-minor axis
    tau = time of pericenter passage
    mu = gravitational constant (mass-dependent)
    n = sqrt(mu/a^3)

Relations:

  Anomalies:

    The key equations relating the anomalies are the following.
    First, we have Kepler's equation:

      M = E - e * sin(E)

    The second relation is the following:

      (1 - e) tan^2(theta/2) = (1 + e) tan^2(E / 2)

  Propagation equations:

    M(t) = M(t0) + n * (t - t0)
    M(t) = n (t - tau)

TODO:
  - [ ] Make custom vjps and jvps using the implicit
        function theorem.
"""
from functools import wraps
import math
from typing import Callable, Any

import jax
from jax._src.numpy.lax_numpy import result_type
import jax.numpy as jnp
from anomaly.optimizers.newton import MAX_NEWTON_RAPHSON_ITERATIONS, newton_raphson

# Newton steps here converge in very few iterations to converge
_MAX_NEWTON_ITERATIONS = 8


def _clip_to_rads(f: Callable[..., jnp.ndarray]):
  """Clips the output of a function to [0, 2*pi].

  Used when the final call is a Newton-Raphson iteration
  that may result in out-of-bounds radians.
  """
  @wraps(f)
  def wrapper(*args, **kwargs):
    return jnp.remainder(f(*args, **kwargs), 2*math.pi)
  return wrapper


@_clip_to_rads
def mean_to_eccentric_anomaly(
    mean_anomaly: jnp.ndarray,
    eccentricity: jnp.ndarray,
) -> jnp.ndarray:
  """Convert mean anomaly to eccentric anomaly.

  Parameters:
    mean_anomaly: The mean anomaly (radians).
    eccentricity: Eccentricity (between zero and one).
    **kwargs: Additional kewyord arguments passed to
      the Newton-Raphson optimizer.

  Returns:
    The eccentric anomaly.
  """
  M, e = mean_anomaly, eccentricity
  eccentric_anomaly_start = _mean_to_eccentric_anomaly_approx(
      mean_anomaly=M,
      eccentricity=e,
  )
  implicit_ea = lambda E: _kepler_implicit(
    eccentric_anomaly=E, mean_anomaly=M, eccentricity=e)
  return newton_raphson(implicit_ea, x0=eccentric_anomaly_start, max_iter=_MAX_NEWTON_ITERATIONS)

@_clip_to_rads
def mean_to_true_anomaly(
    mean_anomaly: jnp.ndarray,
    eccentricity: jnp.ndarray,
) -> jnp.ndarray:
  """Convert mean anomaly to true anomaly.

  Parameters:
    mean_anomaly: The mean anomaly (radians).
    eccentricity: Eccentricity (between zero and one).

  Returns:
    The true anomaly.
  """
  M, e = mean_anomaly, eccentricity
  E = mean_to_eccentric_anomaly(mean_anomaly=M, eccentricity=e)
  implicit_theta = lambda theta: _anomaly_implicit(
    eccentric_anomaly=E,
    true_anomaly=theta,
    eccentricity=e,
  )
  theta_start = _mean_to_true_anomaly_approx(mean_anomaly=M, eccentricity=e)
  return newton_raphson(implicit_theta, x0=theta_start, max_iter=_MAX_NEWTON_ITERATIONS)


@_clip_to_rads
def true_to_mean_anomaly(
    true_anomaly: jnp.ndarray,
    eccentricity: jnp.ndarray,
) -> jnp.ndarray:
  """Convert true anomaly to mean anomaly.

  Parameters:
    true_anomaly: The true anomaly (radians).
    eccentricity: Eccentricity (between zero and one).

  Returns:
    The mean anomaly.
  """
  theta, e = true_anomaly, eccentricity
  E = true_to_eccentric_anomaly(true_anomaly=theta, eccentricity=e)
  implicit_m = lambda M: _kepler_implicit(mean_anomaly=M, eccentric_anomaly=E, eccentricity=e)
  M_start = _true_to_mean_anomaly_approx(true_anomaly=theta, eccentricity=e)
  return newton_raphson(implicit_m, M_start, max_iter=_MAX_NEWTON_ITERATIONS)


def eccentric_to_true_anomaly(
    eccentric_anomaly: jnp.ndarray,
    eccentricity: jnp.ndarray,
) -> jnp.ndarray:
  """Convert eccentric anomaly to true anomaly.

  Parameters:
    eccentric_anomaly: The eccentric anomaly (radians).
    eccentricity: Eccentricity (between zero and one).

  Returns:
    The true anomaly.
  """
  E, e = eccentric_anomaly, eccentricity
  # Use M as intermediate because E -> M is an exact formula.
  M = eccentric_to_mean_anomaly(eccentric_anomaly=E, eccentricity=e)
  return mean_to_true_anomaly(mean_anomaly=M, eccentricity=e)


@_clip_to_rads
def true_to_eccentric_anomaly(
    true_anomaly: jnp.ndarray,
    eccentricity: jnp.ndarray,
) -> jnp.ndarray:
  """Convert true anomaly to eccentric anomaly.

  Parameters:
    true_anomaly: The true anomaly (radians).
    eccentricity: Eccentricity (between zero and one).

  Returns:
    The eccentric anomaly.
  """
  theta, e = true_anomaly, eccentricity
  implicit_E = lambda E: _anomaly_implicit(
    eccentric_anomaly=E, true_anomaly=theta, eccentricity=e)
  E_start = _true_to_eccentric_anomaly_approx(
    true_anomaly=theta,
    eccentricity=e,
  )
  return newton_raphson(implicit_E, x0=E_start, max_iter=_MAX_NEWTON_ITERATIONS)


def eccentric_to_mean_anomaly(
    eccentric_anomaly: jnp.ndarray,
    eccentricity: jnp.ndarray,
) -> jnp.ndarray:
  """Convert eccentric anomaly to mean anomaly.

  Parameters:
    eccentric_anomaly: The eccentric anomaly (radians).
    eccentricity: Eccentricity (between zero and one).

  Returns:
    The mean anomaly.
  """
  E, e = eccentric_anomaly, eccentricity
  return E - e * jnp.sin(E)


def _kepler_implicit(
    mean_anomaly: jnp.ndarray,
    eccentric_anomaly: jnp.ndarray,
    eccentricity: jnp.ndarray,
) -> jnp.ndarray:
  """The Kepler equation in implicit form.

  The value of the Kepler equation is zero if and only if
  the three parameters correspond to a valid triple.

  NOTE: Units of anomalies are radians.
  """
  M, E, e = mean_anomaly, eccentric_anomaly, eccentricity
  return M - E + e * jnp.sin(E)


def _anomaly_implicit(
    eccentric_anomaly: jnp.ndarray,
    true_anomaly: jnp.ndarray,
    eccentricity: jnp.ndarray,
) -> jnp.ndarray:
  """Impicit equation relating eccentricities to the true anomaly.

  The value of this function is zero if and only if
  the three parameters correspond to a valid triple.

  NOTE: Units of anomalies are radians.
  """
  E, theta, e =  eccentric_anomaly, true_anomaly, eccentricity
  # The implicit equation is usually written as
  #
  # (1 - e) tan^2(theta/2) = (1 + e) tan^2(E / 2)
  #
  # To avoid singularities, we use the equivalent formulation
  #
  # sqrt(1-e) sin(theta/2) cos(E/2) = sqrt(1+e) cos(theta/2) sin(E/2)
  return (
    jnp.sqrt(1 - e)*jnp.sin(theta/2)*jnp.cos(E/2)
    - jnp.sqrt(1 + e)*jnp.cos(theta/2)*jnp.sin(E/2)
  )


@_clip_to_rads
def _true_to_mean_anomaly_approx(
    true_anomaly: jnp.ndarray,
    eccentricity: jnp.ndarray,
) -> jnp.ndarray:
  """Approximate value for mean anomaly.

  This is a truncated Fourier expansion of the
  full relation, valid to fourth order in the
  eccentricity. Extremely accurate when the
  eccentricity is small.

  Parameters:
    true_anomaly: The true anomaly (radians).
    eccentricity: Eccentricity (between zero and one).

  Returns:
    The approximate mean anomaly.
  """
  theta, e = true_anomaly, eccentricity
  return (
      theta
      - 2*e*jnp.sin(theta)
      + (3/4 * jnp.power(e, 2) + 1/8 * jnp.power(e, 4))*jnp.sin(2*theta)
      - (1/3 * jnp.power(e, 3))*jnp.sin(3*theta)
      + (5/32 * jnp.power(e,4))*jnp.sin(4*theta)
  )


@_clip_to_rads
def _mean_to_true_anomaly_approx(
    mean_anomaly: jnp.ndarray,
    eccentricity: jnp.ndarray,
) -> jnp.ndarray:
  """Approximate value for true anomaly.

  This is a truncated Fourier expansion of the
  full relation, valid to third order in the
  eccentricity. Extremely accurate when the
  eccentricity is small.

  Parameters:
    mean_anomaly: The mean anomaly (radians).
    eccentricity: Eccentricity (between zero and one).

  Returns:
    The approximate true anomaly.
  """
  M, e = mean_anomaly, eccentricity
  return (
      M
      + (2*e - (1/4) * jnp.power(e,3)) * jnp.sin(M)
      + (5/4 * jnp.power(e, 2)) * jnp.sin(2*M)
      + (13/12 * jnp.power(e, 3)) * jnp.sin(3*M)
  )


@_clip_to_rads
def _mean_to_eccentric_anomaly_approx(
    mean_anomaly: jnp.ndarray,
    eccentricity: jnp.ndarray,
) -> jnp.ndarray:
  """Approximate value for true anomaly.

    Parameters:
      mean_anomaly: The mean anomaly (radians).
      eccentricity: Eccentricity (between zero and one).

    Returns:
      The approximate eccentric anomaly.
  """
  M, e = mean_anomaly, eccentricity
  # From 1st order Taylor series approximation around E = M
  return (e * (jnp.sin(M) + jnp.cos(M)*M) + M) / (1 + e * jnp.cos(M) )

@_clip_to_rads
def _true_to_eccentric_anomaly_approx(
  true_anomaly: jnp.ndarray,
  eccentricity: jnp.ndarray,
) -> jnp.ndarray:
  # Use 1st order Taylor approximation
  # E ~ theta - e /(1+e) sin(theta)
  theta, e = true_anomaly, eccentricity
  return theta - e/(1 + e)*jnp.sin(theta)
