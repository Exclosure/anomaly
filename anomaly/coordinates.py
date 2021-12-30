# pylint: disable=R0914,non-ascii-name
"""Module for coordinates and transformation functions."""
from datetime import datetime

import jax
from chex import dataclass
from jax import numpy as jnp

from anomaly.utils import (
    MU_METERS_CUBED_PER_SECOND_SQUARED,
    deg2rad,
    norm_and_norm_squared,
    rad2deg,
    rot1,
    rot3,
    signed_arccos,
)

J2000_DATETIME = datetime(2000, 1, 1, 12)
PRECISION = jax.lax.Precision.HIGHEST
_TOLERANCE = 1e-6


# Earth-centered inertial coordinate basis
_I = jnp.array([1.0, 0.0, 0.0])  # To sun on reference vernal equinox
_K = jnp.array([0.0, 0.0, 1.0])  # North pole on reference vernal equinox


@dataclass(frozen=True)
class ClassicalOrbitalElement:
    """Classical orbital element relative to J2000 / ICRF frame.

    Orbital elements represent instantaneous state of a satellite
    at the epoch time.

    References:
        [1] https://www.amsat.org/keplerian-elements-tutorial/

        [2] Vallado, D.A. Fundamentals of Astrophysics and Applications, 2013.

    """

    epoch_sec: float
    """The epoch second for this measurement.

    Also known as:
        ``T0``

    Units:
        Seconds, relative to the Unix epoch of 1970-01-01.
    """

    semiparameter_km: float
    """Elliptical semiparameter (p).

    Defined as :math:`a * (1 - e)` (for circular and elliptical
    orbits) or :math:`h^2 / mu` (valid for all conical sections).

    Also known as:
        ``p``

    Units:
        Kilometers (km).
    """

    @property
    def semimajor_axis_km(self) -> float:
        """Length of the semimajor axis in kilometers (a).

        Also known as:
            ``a``

        Units:
            Kilometers (km)
        """
        e_squared = self.eccentricity ** 2  # == 1 - b^2 / a^2
        p = self.semiparameter_km  # == b^2 / a
        return p / (1 - e_squared)

    eccentricity: float
    """Eccentricity of the orbital ellipse (e).

    Defined as the norm of the perigree.

    Also known as:
        ``e``, ``ecce``, ``E0``

    Units:
        Unitless, in the range [0, 1). (Zero is a circle.)
    """

    inclination_deg: float
    """Orbital inclination in degrees (i).

    Angle between orbital plane and the north pole (``K``).

    Also known as:
       ``i``, ``I0``

    Units:
      Degrees, in the range [0, 180).
    """

    ascension_deg: float
    r"""Right ascension of ascending node in degrees (Ω).

    Defined as the angle between the node and I in the
    (I, J) plane.

    Also known as:
        ``Ω``, ``RAAN``, ``O0``, ``longitude``,

    Units:
        Degrees, in the range [0, 360).
    """

    perigree_deg: float
    """Argument of perigree in degrees (ω).

    This is the angle between the perigree and K
    in the ``(node, K)`` plane.

    Also known as:
        ``ω``, ``ARGP``, ``W0``

    Units:
        Degrees, in the range [0, 360).
    """

    true_anomaly_deg: float
    """True anomaly in degrees (ν).

    Defined as the angle between the perigree and the
    position in the (perigree, -node) plane.

    Also known as:
        ``ν``, ``nu``
    """

    true_longitude_of_periapsis_deg: float
    """True longitude of periapsis (ω̃ₜᵣᵤₑ).

    Defined as the angle betwen ``I`` and the perigree
    in the (``I``, ``J``) plane

    Units:
        Degrees, in the range [0, 360).
    """

    argument_of_latitude_deg: float
    """Argument of latitude in degrees (u).

    Defined as the angle between the node and the radius
    in the (node, velocity) plane.

    Units:
        Degrees, in the range [0, 360).
    """

    true_longitude_deg: float
    """True longitude in degrees (λₜᵣᵤₑ).

    Defined as the angle between ``I`` axis and
    the position ``r`` in the (``I``, ``J``) plane.

    Units:
        Degrees, in the range [0, 360).
    """


@dataclass(frozen=True)
class OrbitalStateVector:
    """Orbital state vector relative to J2000 / ICRF reference frame."""

    epoch_sec: float
    """The epoch relative to this measurement (aka T0).

    Units:
        Seconds, relative to the Unix epoch of 1970-01-01.
    """

    position: jnp.ndarray
    """Position vector.

    This is a 3-vector of cartesian coordinates, with origin
    at the center of mass of the earth. The position is in the
    EC

    Units:
        Kilometers (km)
    """

    velocity: jnp.ndarray
    """Velocity vector.

    This is a 3-vector of cartesian velocity coordinates.

    Units:
        Kilometers / second (km/s)
    """


# Vallejo: rv2coe, Algorithm 9
def orbital_state_vector_to_orbital_element(
    state_vector: OrbitalStateVector,
) -> ClassicalOrbitalElement:
    """Convert orbital state vector to classical orbital element."""
    # pylint: disable-C0103
    mu = MU_METERS_CUBED_PER_SECOND_SQUARED
    r, _ = norm_and_norm_squared(state_vector.position)

    # Inverse radius scaled by mu.
    mu_rinv = mu / r

    # Velocity
    _, v_squared = norm_and_norm_squared(state_vector.velocity)

    # Specific momentum vector
    h_vec = jnp.cross(state_vector.position, state_vector.velocity)
    h, h_squared = norm_and_norm_squared(h_vec)
    h_norm = h_vec / h

    # Node vector
    node_vec = _get_node_vec(h_vec, state_vector.position)
    n, _ = norm_and_norm_squared(node_vec)
    n_vec_norm = node_vec / n

    # Eccentricity vector (always points to perigree)
    eccentricity_vec = (
        (v_squared - mu_rinv) * state_vector.position
        - jnp.dot(state_vector.position, state_vector.velocity, precision=PRECISION)
        * state_vector.velocity
    ) / mu

    # Eccentricity
    eccentricity, _ = norm_and_norm_squared(eccentricity_vec)
    eccentricity_vec_norm = eccentricity_vec / eccentricity
    semiparameter_km = h_squared / mu  # semiparameter (p)

    # Angle between orbital plane and K
    inclination_rad = jnp.arccos(h_norm[2])

    # Angle between the node and I
    ascention_rad = signed_arccos(
        n_vec_norm[0],
        n_vec_norm[1],
    )

    # Angle of the perigree vector in the orbital plane
    perigree_rad = signed_arccos(
        jnp.dot(n_vec_norm, eccentricity_vec_norm, precision=PRECISION),
        eccentricity_vec[2],
    )

    # Angle between the perigree and the position in the orbital plane
    true_anomaly_rad = signed_arccos(
        jnp.dot(eccentricity_vec_norm, state_vector.position, precision=PRECISION) / r,
        jnp.dot(state_vector.position, state_vector.velocity, precision=PRECISION),
    )

    # Angle betwen the perigree and I in the (I, J) plane
    true_longitude_of_periapsis_rad = signed_arccos(
        eccentricity_vec_norm[0],
        eccentricity_vec[1],
    )

    # Angle between the node and the position in the orbital plane
    argument_of_latitude_rad = signed_arccos(
        jnp.dot(n_vec_norm, state_vector.position, precision=PRECISION) / r,
        state_vector.position[2],
    )

    # The ecliptic longitude at which an orbiting body could
    # actually be found if its inclination were zero
    true_longitude_rad = signed_arccos(
        state_vector.position[0] / r,
        state_vector.position[1],
    )

    return ClassicalOrbitalElement(
        epoch_sec=state_vector.epoch_sec,
        semiparameter_km=semiparameter_km,
        eccentricity=eccentricity,
        inclination_deg=rad2deg(inclination_rad),
        ascension_deg=rad2deg(ascention_rad),
        perigree_deg=rad2deg(perigree_rad),
        true_anomaly_deg=rad2deg(true_anomaly_rad),
        true_longitude_of_periapsis_deg=rad2deg(true_longitude_of_periapsis_rad),
        argument_of_latitude_deg=rad2deg(argument_of_latitude_rad),
        true_longitude_deg=rad2deg(true_longitude_rad),
    )  # type: ignore


def _get_node_vec(h_vec, r_vec):
    node_vec = jnp.cross(_K, h_vec)
    return jax.lax.cond(
        jnp.linalg.norm(node_vec) < _TOLERANCE,
        lambda _: _I * jnp.linalg.norm(r_vec),
        lambda _: node_vec,
        None,
    )


def orbital_element_to_orbital_state_vector(
    coe: ClassicalOrbitalElement,
) -> OrbitalStateVector:
    """Convert a classical orbital element to an orbital state vector."""
    p, nu, e = coe.semiparameter_km, deg2rad(coe.true_anomaly_deg), coe.eccentricity
    r_pqw = _get_rpqw(p, nu, e)
    v_pqw = _get_vpqw(p, nu, e)

    r = pqw_to_ijk(
        r_pqw,
        inclination_rad=deg2rad(coe.inclination_deg),
        perigree_rad=deg2rad(coe.perigree_deg),
        ascension_rad=deg2rad(coe.ascension_deg),
    )
    v = pqw_to_ijk(
        v_pqw,
        inclination_rad=deg2rad(coe.inclination_deg),
        perigree_rad=deg2rad(coe.perigree_deg),
        ascension_rad=deg2rad(coe.ascension_deg),
    )
    return OrbitalStateVector(
        epoch_sec=coe.epoch_sec,
        position=r,
        velocity=v,
    )  # type: ignore


def _get_rpqw(p: float, nu: float, e: float) -> jnp.ndarray:
    return (
        jnp.array(
            [
                p * jnp.cos(nu),
                p * jnp.sin(nu),
                0.0,
            ]
        )
        / (1 + e * jnp.cos(nu))
    )


def _get_vpqw(p: float, nu: float, e: float) -> jnp.ndarray:
    mu = MU_METERS_CUBED_PER_SECOND_SQUARED
    return jnp.array(
        [
            -jnp.sqrt(mu / p) * jnp.sin(nu),
            jnp.sqrt(mu / p) * (e + jnp.cos(nu)),
            0.0,
        ]
    )


def pqw_to_ijk(
    x_pqw: jnp.ndarray,
    inclination_rad: float,
    perigree_rad: float,
    ascension_rad: float,
) -> jnp.ndarray:
    """Rotate from the PQW frame to the IJC (ECI) frame."""
    return rot3(-ascension_rad)(rot1(-inclination_rad)(rot3(-perigree_rad)(x_pqw)))


def ijk_to_pqw(
    x_ijk: jnp.ndarray,
    inclindation_rad: float,
    perigree_rad: float,
    ascention_rad: float,
) -> jnp.ndarray:
    """Rotate from IJK (ECI) frame to PQW frame."""
    return rot3(perigree_rad)(rot1(inclindation_rad)(rot3(ascention_rad)(x_ijk)))
