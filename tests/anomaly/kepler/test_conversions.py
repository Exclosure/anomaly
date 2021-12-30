"""Test conversion functions."""
import math
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from anomaly.kepler.conversions import (
    eccentric_to_mean_anomaly,
    eccentric_to_true_anomaly,
    mean_to_eccentric_anomaly,
    mean_to_true_anomaly,
    true_to_eccentric_anomaly,
    true_to_mean_anomaly,
)

REQUIRED_DECIMALS = 12

round_trip_test_data = list(
    product(
        [
            (eccentric_to_mean_anomaly, mean_to_eccentric_anomaly),
            (eccentric_to_true_anomaly, true_to_eccentric_anomaly),
            (mean_to_eccentric_anomaly, eccentric_to_mean_anomaly),
            (mean_to_true_anomaly, true_to_mean_anomaly),
            (true_to_eccentric_anomaly, eccentric_to_true_anomaly),
            (true_to_mean_anomaly, mean_to_true_anomaly),
        ],
        [
            jnp.array(
                list(
                    product(
                        jnp.linspace(0, 2 * math.pi, 41)[:-1],  # anomalies in [0, 2*pi)
                        jnp.linspace(0, 1, 41)[:-1],  # eccentricity in [0, 1)
                    )
                )
            ),
        ],
    )
)


@pytest.mark.parametrize("func_pair,anomaly_and_eccentricity", round_trip_test_data)
def test_round_trips(func_pair, anomaly_and_eccentricity):
    """Test round-trip anomaly conversions."""
    anomaly = anomaly_and_eccentricity[:, 0]
    eccentricity = anomaly_and_eccentricity[:, 1]
    forward, backward = func_pair
    intermediate = jax.vmap(forward, in_axes=(0, 0))(anomaly, eccentricity)
    result = jax.vmap(backward, in_axes=(0, 0))(intermediate, eccentricity)
    np.testing.assert_array_almost_equal(result, anomaly, decimal=REQUIRED_DECIMALS)


@pytest.mark.parametrize("func_pair,anomaly_and_eccentricity", round_trip_test_data)
def test_round_trips_jit(func_pair, anomaly_and_eccentricity):
    """Test round-trip anomaly conversions."""
    anomaly = anomaly_and_eccentricity[:, 0]
    eccentricity = anomaly_and_eccentricity[:, 1]
    forward, backward = func_pair
    intermediate = jax.vmap(jax.jit(forward), in_axes=(0, 0))(anomaly, eccentricity)
    result = jax.vmap(jax.jit(backward), in_axes=(0, 0))(intermediate, eccentricity)
    np.testing.assert_array_almost_equal(result, anomaly, decimal=REQUIRED_DECIMALS)


@pytest.mark.parametrize("func_pair,anomaly_and_eccentricity", round_trip_test_data)
def test_round_trips_jacfwd(func_pair, anomaly_and_eccentricity):
    """Test round trip for conversions and their inverses with forward-mode."""
    anomaly = anomaly_and_eccentricity[:, 0]
    eccentricity = anomaly_and_eccentricity[:, 1]
    forward, backward = func_pair
    intermediate = jax.vmap(forward, in_axes=(0, 0))(anomaly, eccentricity)
    g_fwd = jax.vmap(jax.jacfwd(forward, argnums=(0,)), in_axes=(0, 0))(
        anomaly, eccentricity
    )
    g_bwd = jax.vmap(jax.jacfwd(backward, argnums=(0,)), in_axes=(0, 0))(
        intermediate, eccentricity
    )
    np.testing.assert_array_almost_equal(
        g_fwd[0], 1.0 / g_bwd[0], decimal=REQUIRED_DECIMALS
    )


@pytest.mark.parametrize("func_pair,anomaly_and_eccentricity", round_trip_test_data)
def test_round_trips_jacrev(func_pair, anomaly_and_eccentricity):
    """Test round trip for conversions and their inverses with reverse-mode."""
    anomaly = anomaly_and_eccentricity[:, 0]
    eccentricity = anomaly_and_eccentricity[:, 1]
    forward, backward = func_pair
    intermediate = jax.vmap(forward, in_axes=(0, 0))(anomaly, eccentricity)
    g_fwd = jax.vmap(jax.jacrev(forward, argnums=(0,)), in_axes=(0, 0))(
        anomaly, eccentricity
    )
    g_bwd = jax.vmap(jax.jacrev(backward, argnums=(0,)), in_axes=(0, 0))(
        intermediate, eccentricity
    )
    np.testing.assert_array_almost_equal(
        g_fwd[0], 1.0 / g_bwd[0], decimal=REQUIRED_DECIMALS
    )
