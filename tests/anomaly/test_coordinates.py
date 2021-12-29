"""Tests for coordinate transformations."""
from itertools import product

import jax
from jax import numpy as jnp
import pytest
from pytest import approx

from anomaly.testing_utils import assert_trees_almost_equal
from anomaly.coordinates import (
    J2000_DATETIME,
    ClassicalOrbitalElement,
    OrbitalStateVector,
    orbital_state_vector_to_orbital_element,
)


@pytest.mark.parametrize("do_jit", [False, True])
def test_orbital_state_vector_to_orbital_element(do_jit):
    """Test example case of orbital state vectors to orbital elements.

    The test here is taken from Vallado, p. 115-116. Note that Vallado
    rounds to 7 figures, so it's likely that our 64-bit computations
    are more accurate.
    """
    position = jnp.array([6524.834, 6862.875, 6448.296])
    velocity = jnp.array([4.901327, 5.533756, -1.976341])
    epoch_sec = J2000_DATETIME.timestamp()
    state = OrbitalStateVector(
        epoch_sec=epoch_sec, position=position, velocity=velocity
    )  # type: ignore

    if do_jit:
        coe = jax.jit(orbital_state_vector_to_orbital_element)(state)
    else:
        coe = orbital_state_vector_to_orbital_element(state)

    assert coe.semiparameter_km == approx(11067.790)
    assert coe.semimajor_axis_km == approx(36127.343)
    assert coe.inclination_deg == approx(87.870, rel=1e-5)
    assert coe.ascension_deg == approx(227.898, rel=1e-5)
    assert coe.perigree_deg == approx(53.38, rel=1e-4)
    assert coe.true_anomaly_deg == approx(92.335, rel=1e-5)
    assert coe.true_longitude_of_periapsis_deg == approx(247.806, rel=1e-5)

    # Note: Error in Vallado, p. 116, u should be 145.71937, not 145.60549
    # using the 7-sig rounded values.
    assert coe.argument_of_latitude_deg == approx(145.71937, rel=1e-5)
    assert coe.true_longitude_deg == approx(55.282587, rel=1e-5)


@pytest.mark.parametrize(
    "do_jit,jac_mode", list(product([False, True], ["fwd", "rev"]))
)
def test_orbital_state_vector_to_orbital_element_jacobian(do_jit, jac_mode):
    """Test example case of orbital state vectors to orbital elements.

    The test here is taken from Vallado, p. 115-116. Note that Vallado
    rounds to 7 figures, so it's likely that our 64-bit computations
    are more accurate.
    """
    position = jnp.array([6524.834, 6862.875, 6448.296])
    velocity = jnp.array([4.901327, 5.533756, -1.976341])
    epoch_sec = J2000_DATETIME.timestamp()
    state = OrbitalStateVector(
        epoch_sec=epoch_sec, position=position, velocity=velocity
    )  # type: ignore

    def maybe_jit(fun):
        if do_jit:
            return jax.jit(fun)
        return fun

    if jac_mode == "fwd":
        jac_fun = maybe_jit(jax.jacfwd(orbital_state_vector_to_orbital_element))
    else:
        jac_fun = maybe_jit(jax.jacrev(orbital_state_vector_to_orbital_element))

    coe_jac = jac_fun(state)

    # pytype: disable=wrong-keyword-args
    expected = ClassicalOrbitalElement(
        # epoch_sec depends only on original epoch_sec
        epoch_sec=OrbitalStateVector(
            epoch_sec=jnp.array(1.0),
            position=jnp.array([0.0, 0.0, 0.0]),
            velocity=jnp.array([0.0, 0.0, 0.0]),
        ),  # type: ignore
        semiparameter_km=OrbitalStateVector(
            epoch_sec=jnp.array(0.0),
            position=jnp.array([0.50985685, 0.42761464, 2.46176658]),
            velocity=jnp.array([1354.75795856, 1674.21379212, -3152.69193108]),
        ),  # type: ignore
        semimajor_axis_km=OrbitalStateVector(
            epoch_sec=jnp.array(0.0),
            position=jnp.array([11.32680329, 11.91362648, 11.19393694]),
            velocity=jnp.array([32097.98851187, 36239.66254761, -12942.7338175]),
        ),  # type: ignore
        eccentricity=OrbitalStateVector(
            epoch_sec=jnp.array(0.0),
            position=jnp.array([4.91905754e-05, 5.35446667e-05, 1.60783275e-05]),
            velocity=jnp.array([0.14089354, 0.15666965, -0.01349975]),
        ),  # type: ignore
        inclination_deg=OrbitalStateVector(
            epoch_sec=jnp.array(0.0),
            position=jnp.array([-0.00472777, 0.00427213, 0.00023709]),
            velocity=jnp.array([6.05467427, -5.47115195, -0.30363255]),
        ),  # type: ignore
        ascension_deg=OrbitalStateVector(
            epoch_sec=jnp.array(0.0),
            position=jnp.array([-1.26579470e-03, 1.14380309e-03, 6.34776473e-05]),
            velocity=jnp.array([-4.12996487, 3.7319374, 0.20711135]),
        ),  # type: ignore
        perigree_deg=OrbitalStateVector(
            epoch_sec=jnp.array(0.0),
            position=jnp.array([0.00171713, 0.00139262, 0.00744011]),
            velocity=jnp.array([8.69064532, 10.3624466, -18.99262793]),
        ),  # type: ignore
        true_anomaly_deg=OrbitalStateVector(
            epoch_sec=jnp.array(0.0),
            position=jnp.array([0.00010446, 0.00075782, -0.01157202]),
            velocity=jnp.array([-8.53708417, -10.50120821, 18.98492708]),
        ),  # type: ignore
        true_longitude_of_periapsis_deg=OrbitalStateVector(
            epoch_sec=jnp.array(0.0),
            position=jnp.array([0.00343494, -0.00134036, 0.00433575]),
            velocity=jnp.array([-0.7492671, 11.58767227, -11.07745861]),
        ),  # type: ignore
        argument_of_latitude_deg=OrbitalStateVector(
            epoch_sec=jnp.array(-0.0),
            position=jnp.array([0.00182159, 0.00215044, -0.00413191]),
            velocity=jnp.array([0.15356114, -0.13876161, -0.00770085]),
        ),  # type: ignore
        true_longitude_deg=OrbitalStateVector(
            epoch_sec=jnp.array(-0.0),
            position=jnp.array([-0.00411079, 0.00207576, 0.00195037]),
            velocity=jnp.array([-0.0, -0.0, -0.0]),
        ),  # type: ignore
    )  # type: ignore

    assert_trees_almost_equal(
        coe_jac,
        expected,
    )


@pytest.mark.parametrize("do_jit", [False, True])
def test_equatorial_orbital_state_vector_to_orbital_element(do_jit):
    """Test equatorial orbital state vectors to orbital elements."""
    position = jnp.array([0.999999999, 0, 0])
    velocity = jnp.array([0, 1.000000001, 0])
    epoch_sec = J2000_DATETIME.timestamp()
    state = OrbitalStateVector(
        epoch_sec=epoch_sec, position=position, velocity=velocity
    )  # type: ignore

    if do_jit:
        coe = jax.jit(orbital_state_vector_to_orbital_element)(state)
    else:
        coe = orbital_state_vector_to_orbital_element(state)
    assert coe.semimajor_axis_km == approx(0.5, rel=1e-5)
    assert coe.eccentricity == approx(1.0, rel=1e-5)
    assert coe.inclination_deg == approx(0.0, rel=1e-5)
    assert coe.ascension_deg == approx(0.0, rel=1e-5)
    assert coe.perigree_deg == approx(180.0, rel=1e-5)
    assert coe.true_anomaly_deg == approx(180.0, rel=1e-5)
    assert coe.true_longitude_of_periapsis_deg == approx(180.0, rel=1e-5)
    assert coe.argument_of_latitude_deg == approx(0.0, rel=1e-5)
    assert coe.true_longitude_deg == approx(0.0, rel=1e-5)
