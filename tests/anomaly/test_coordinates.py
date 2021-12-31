"""Tests for coordinate transformations."""
from itertools import product

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from pytest import approx

from anomaly.coordinates import (
    J2000_DATETIME,
    ClassicalOrbitalElement,
    OrbitalStateVector,
    ijk_to_pqw,
    orbital_element_to_orbital_state_vector,
    orbital_state_vector_to_orbital_element,
    pqw_to_ijk,
)
from anomaly.testing_utils import assert_trees_allclose
from anomaly.utils import deg2rad


def _jit(func, do_jit):
    """JIT the function if do_jit is True."""
    if do_jit:
        return jax.jit(func)
    return func


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

    func = _jit(orbital_state_vector_to_orbital_element, do_jit)
    coe = func(state)

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

    if jac_mode == "fwd":
        jac_fun = _jit(jax.jacfwd(orbital_state_vector_to_orbital_element), do_jit)
    else:
        jac_fun = _jit(jax.jacrev(orbital_state_vector_to_orbital_element), do_jit)

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
            position=jnp.array(
                [0.00010446313823647312, 0.0007578172888617357, -0.011572024203925656]
            ),
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

    assert_trees_allclose(coe_jac, expected, rtol=1e-5)


@pytest.mark.parametrize("do_jit", [False, True])
def test_equatorial_orbital_state_vector_to_orbital_element(do_jit):
    """Test equatorial orbital state vectors to orbital elements."""
    position = jnp.array([0.999999999, 0, 0])
    velocity = jnp.array([0, 1.000000001, 0])
    epoch_sec = J2000_DATETIME.timestamp()
    state = OrbitalStateVector(
        epoch_sec=epoch_sec, position=position, velocity=velocity
    )  # type: ignore

    coe = _jit(orbital_state_vector_to_orbital_element, do_jit)(state)

    assert coe.semimajor_axis_km == approx(0.5, rel=1e-5)
    assert coe.eccentricity == approx(1.0, rel=1e-5)
    assert coe.inclination_deg == approx(0.0, rel=1e-5)
    assert coe.ascension_deg == approx(0.0, rel=1e-5)
    assert coe.perigree_deg == approx(180.0, rel=1e-5)
    assert coe.true_anomaly_deg == approx(180.0, rel=1e-5)
    assert coe.true_longitude_of_periapsis_deg == approx(180.0, rel=1e-5)
    assert coe.argument_of_latitude_deg == approx(0.0, rel=1e-5)
    assert coe.true_longitude_deg == approx(0.0, rel=1e-5)


@pytest.mark.parametrize("do_jit", [False, True])
def test_orbital_element_to_orbital_state_vector(do_jit):
    """Test orbital element to state vector.

    The test here is taken from Vallado, p. 115-116. Note that Vallado
    rounds to 7 figures, so it's likely that our 64-bit computations
    are more accurate.
    """
    epoch_sec = J2000_DATETIME.timestamp()
    coe = ClassicalOrbitalElement(
        epoch_sec=epoch_sec,
        semiparameter_km=11067.790,
        eccentricity=0.83285,
        inclination_deg=87.87,
        ascension_deg=227.89,
        perigree_deg=53.38,
        true_anomaly_deg=92.335,
        true_longitude_of_periapsis_deg=000,
        argument_of_latitude_deg=000,
        true_longitude_deg=000,
    )  # type: ignore

    state = _jit(orbital_element_to_orbital_state_vector, do_jit)(coe)

    assert_trees_allclose(
        state,
        OrbitalStateVector(
            epoch_sec=epoch_sec,
            position=jnp.array([6525.368, 6861.532, 6449.119]),
            velocity=jnp.array([4.902279, 5.533140, -1.975710]),
        ),  # type: ignore
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "do_jit,jac_mode", list(product([False, True], ["fwd", "rev"]))
)
def test_pqw_to_ijk(do_jit, jac_mode):
    """Test conversion from PQW to IJK."""
    inclination_rad = deg2rad(87.87)
    ascension_rad = deg2rad(227.89)
    perigree_rad = deg2rad(53.38)
    x_pqw = jnp.array(
        [
            -466.7679,
            11447.0219,
            0.0,
        ]
    )

    args = (
        x_pqw,
        inclination_rad,
        perigree_rad,
        ascension_rad,
    )

    x_ijk = _jit(pqw_to_ijk, do_jit)(*args)
    if jac_mode == "fwd":
        rotation_matrix = _jit(jax.jacfwd(pqw_to_ijk, argnums=(0,)), do_jit)(*args)
    else:
        rotation_matrix = _jit(jax.jacrev(pqw_to_ijk, argnums=(0,)), do_jit)(*args)
    np.testing.assert_allclose(
        x_ijk,
        jnp.array([6525.368, 6861.532, 6449.119]),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        rotation_matrix,
        jnp.array(
            [
                [
                    [-0.37786007, 0.55464179, -0.74134625],
                    [-0.46252560, 0.58055638, 0.67009280],
                    [0.80205476, 0.59609293, 0.03716695],
                ]
            ]
        ),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "do_jit,jac_mode", list(product([False, True], ["fwd", "rev"]))
)
def test_ijk_to_pqw(do_jit, jac_mode):
    """Test conversion from IJK to PQW."""
    x_ijk = jnp.array([6525.36961515, 6861.53366465, 6449.11542819])
    inclination_rad = deg2rad(87.87)
    ascension_rad = deg2rad(227.89)
    perigree_rad = deg2rad(53.38)

    args = (
        x_ijk,
        inclination_rad,
        perigree_rad,
        ascension_rad,
    )

    x_pqw = _jit(ijk_to_pqw, do_jit)(*args)
    if jac_mode == "fwd":
        rotation_matrix = _jit(jax.jacfwd(ijk_to_pqw, argnums=(0,)), do_jit)(*args)
    else:
        rotation_matrix = _jit(jax.jacrev(ijk_to_pqw, argnums=(0,)), do_jit)(*args)

    np.testing.assert_allclose(
        x_pqw,
        jnp.array([-466.7679, 11447.0219, 0.0]),
        rtol=1e-9,
        atol=1e-9,
    )

    np.testing.assert_allclose(
        rotation_matrix,
        jnp.array(
            [
                [
                    [-0.37786007, -0.46252560, 0.80205476],
                    [0.55464179, 0.58055638, 0.59609293],
                    [-0.74134625, 0.67009280, 0.03716695],
                ]
            ]
        ),
        rtol=1e-7,
    )


@pytest.mark.parametrize(
    "do_jit,jac_mode", list(product([False, True], ["fwd", "rev"]))
)
def test_orbital_state_vector_to_orbital_element_round_trip_jacobian(do_jit, jac_mode):
    """Test round-trip orbital state vectors to orbital state vectors."""
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

    def round_trip(state_):
        coe = orbital_state_vector_to_orbital_element(state_)
        return orbital_element_to_orbital_state_vector(coe)

    if jac_mode == "fwd":
        jac_fun = maybe_jit(jax.jacfwd(round_trip))
    else:
        jac_fun = maybe_jit(jax.jacrev(round_trip))

    assert_trees_allclose(round_trip(state), state)

    round_trip_jac = jac_fun(state)
    assert_trees_allclose(
        round_trip_jac,
        OrbitalStateVector(
            epoch_sec=OrbitalStateVector(
                epoch_sec=jnp.array(
                    1.0,
                ),
                position=jnp.array(
                    [0.0, 0.0, 0.0],
                ),
                velocity=jnp.array(
                    [0.0, 0.0, 0.0],
                ),
            ),  # type: ignore
            position=OrbitalStateVector(
                epoch_sec=jnp.array(
                    [0.0, 0.0, 0.0],
                ),
                position=jnp.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                ),
                velocity=jnp.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                ),
            ),  # type: ignore
            velocity=OrbitalStateVector(
                epoch_sec=jnp.array(
                    [0.0, 0.0, 0.0],
                ),
                position=jnp.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                ),
                velocity=jnp.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                ),
            ),  # type: ignore
        ),
        rtol=0,
        atol=1e-11,
    )
