from itertools import product
import math

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


REQUIRED_DECIMALS = 14

round_trip_test_data = list(product(
  [
    (eccentric_to_mean_anomaly, mean_to_eccentric_anomaly),
    (eccentric_to_true_anomaly, true_to_eccentric_anomaly),
    (mean_to_eccentric_anomaly, eccentric_to_mean_anomaly),
    (mean_to_true_anomaly, true_to_mean_anomaly),
    (true_to_eccentric_anomaly, eccentric_to_true_anomaly),
    (true_to_mean_anomaly, mean_to_true_anomaly),
  ],
  [jnp.linspace(0, 2*math.pi, 11)[:-1]],  # anomalies in [0, 2*pi)
  [jnp.linspace(0, 1, 11)[:-1]],  # eccentricity in [0, 1)
))


@pytest.mark.parametrize("func_pair,anomaly,eccentricity", round_trip_test_data)
def test_round_trips(func_pair, anomaly, eccentricity):
  """Test round-trip anomaly conversions."""
  forward, backward = func_pair
  intermediate = forward(anomaly, eccentricity=eccentricity)
  result = backward(intermediate, eccentricity=eccentricity)
  np.testing.assert_array_almost_equal(result, anomaly, decimal=REQUIRED_DECIMALS)
