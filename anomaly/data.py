from datetime import datetime
from typing import Iterable
import sys

from chex import dataclass
from jax import numpy as jnp


J2000 = datetime(2000, 1, 1, 12)
_SECONDS_PER_DAY = 60 * 60 * 24

# Earth's gravitational constant.
MU = 3.98574405096e14  #  Units: m^3/s^2


@dataclass(frozen=True)
class KeplerianOrbitalElement:
    """Keplerian orbital element relative to J2000 / ICRF frame.

    By convention, orbital elements represent instantaneous
    state. In particular, these parameters should be interpreted
    as taken at the epoch; propagation forward or backwards in time
    may result in errors.

    References:

        [1] https://www.amsat.org/keplerian-elements-tutorial/

    """

    epoch: datetime
    """The epoch relative to this measurement (aka T0).

    Units:
        The epoch is represented as python datetime object.
    """

    inclination_deg: float
    """Orbital inclination (I0).

    Units:
      Degrees, in the range [0, 180).
    """

    ascension_deg: float
    """Right ascension of ascending node (aka RAAN, O0, longitude).

    Units:
        Degrees, in the range [0, 360).
    """

    perigree_deg: float
    """Argument of perigree (aka ARGP, W0).

    Units:
        Degrees, in the range [0, 360).
    """

    eccentricity: float
    """Eccentricity (aka ecce, E0, e).

    The eccentricity of the orbital ellipse.

    Units:
        Unitless, in the range [0, 1). (Zero is a circle.)
    """

    mean_motion_rev_per_day: float
    """Mean motion (aka N0).

    This is the reciprocal of "orbital period".

    Units:
        Revolutions per day. Positive number.
    """

    # TODO: derive true anomaly, eccentric anomaly(?)
    mean_anomaly_deg: float
    """Mean anomaly (aka M0, MA, phase).

    An angle that represents the location on the orbit.
    Perigree occurs at MA = 0, while apogee occurs at MA = 180.

    Units:
        Degrees between [0, 360).
    """

    def to_numpy(self):
        """Return this as a numpy vector.

        The order is
            [epoch, inclincation, ascension, perigree,
             eccentricity, mean_motion, mean_anomaly]
        """
        return jnp.array(
            [
                self.epoch,  # TODO: Should be a datetime object
                self.inclination_deg,
                self.ascension_deg,
                self.perigree_deg,
                self.eccentricity,
                self.mean_motion_rev_per_day,
                self.mean_anomaly_deg,
            ]
        )

    def __post_init__(self):
        _validate_range(self.inclination_deg, 0, 180, "inclination")
        _validate_range(self.ascension_deg, 0, 360, "ascension")
        _validate_range(self.perigree_deg, 0, 360, "perigree")
        _validate_range(self.eccentricity, 0, 1, "eccentricity")
        _validate_range(
            self.mean_motion_rev_per_day, 0, sys.float_info.max, "mean_motion"
        )
        _validate_range(self.mean_anomaly_deg, 0, 360, "mean_anomaly")


@dataclass
class OrbitalStateVector:
    """Orbital state vector relative to J2000 / ICRF reference frame."""

    epoch: datetime
    """The epoch relative to this measurement (aka T0).

    Units:
        The epoch is represented as python datetime object.
    """

    position: jnp.ndarray
    """Position vector.

    This is a 3-vector of cartesian coordinates, with origin
    at the center of mass of the earth.

    Units:
        Meters.
    """

    velocity: jnp.ndarray
    """Velocity vector.

    This is a 3-vector of cartesian velocity coordinates.

    Units:
        Meters / second.
    """

    def to_numpy(self) -> jnp.ndarray:
        """Return the [epoch, position, velocity] vector."""
        return jnp.concatenate([[self.epoch.timestamp], self.position, self.velocity])

    def __post_init__(self):
        self.position = _ensure_shape(self.position, (3,))
        self.velocity = _ensure_shape(self.velocity, (3,))


def _validate_range(value: float, min_val: float, max_val: float, name: str):
    if value < min_val or value >= max_val:
        raise ValueError(
            f"Value {value} not in range [{min_val}, {max_val}) allowed for {name}"
        )


def _ensure_shape(x: jnp.ndarray, shape: Iterable[int]) -> jnp.ndarray:
    try:
        return jnp.reshape(x, shape)
    except Exception as e:
        raise ValueError(
            f"Variable {x} not standardizable to a vector with shape {shape}."
        ) from e
