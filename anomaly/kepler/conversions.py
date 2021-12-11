"""Conversions for Keplerian orbit parameters.

For conversions between anomalies, we use Newton-Raphson
iteration over the implicit relations between Keplerian
parameters.

Abbreviations:

  We use the following abbreviations in the code.

    M = mean anomaly
    e = eccentricity
    E = eccentric anomaly
    nu = true anomaly
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

      (1 - e) tan^2(nu/2) = (1 + e) tan^2(E / 2)

  Propagation equations:

    M(t) = M(t0) + n * (t - t0)
    M(t) = n (t - tau)

TODO:
  - [ ] Make custom vjps and jvps using the implicit
        function theorem.
"""
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient

from anomaly.optimizers.newton import newton_1d
from anomaly.utils import clip_to_rads


# Newton steps here converge in very few iterations to converge
_MAX_NEWTON_ITERATIONS = 12

@jax.custom_jvp
@clip_to_rads
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
  eccentric_anomaly_start = stop_gradient(_mean_to_eccentric_anomaly_approx(
      mean_anomaly=M,
      eccentricity=e,
  ))
  implicit_ea = lambda E: _kepler_implicit(
    eccentric_anomaly=E, mean_anomaly=M, eccentricity=e)
  return newton_1d(implicit_ea, eccentric_anomaly_start, max_iter=_MAX_NEWTON_ITERATIONS)

@mean_to_eccentric_anomaly.defjvp
def _mean_to_eccentric_anomaly_jvp(primals, tangents):
  """Compute the JVP using the implicit function theorem."""
  M, e = primals
  tM, te = tangents
  E = mean_to_eccentric_anomaly(M, e)
  (dM, dE, de) = jax.grad(_kepler_implicit, argnums=(0, 1, 2))(M, E, e)
  return E, -(dM*tM + de*te) / dE


@clip_to_rads
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
  return eccentric_to_true_anomaly(E, e)


@clip_to_rads
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
  nu, e = true_anomaly, eccentricity
  E = true_to_eccentric_anomaly(true_anomaly=nu, eccentricity=e)
  return eccentric_to_mean_anomaly(E, e)


@clip_to_rads
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
  x = jnp.cos(E) - e
  y = jnp.sin(E) * jnp.sqrt(1 - e **2)
  return jnp.arctan2(y, x)


@clip_to_rads
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
  nu, e = true_anomaly, eccentricity
  x = e + jnp.cos(nu)
  y = jnp.sin(nu) * jnp.sqrt(1- e **2)
  return jnp.arctan2(y, x)


@clip_to_rads
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


@clip_to_rads
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
  # cf. Vallado, p. 75
  return M + e * jnp.sign(jnp.pi - M)
