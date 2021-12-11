
import math
from functools import wraps
import jax.numpy as jnp
from typing import Callable

def clip_to_rads(f: Callable[..., jnp.ndarray]):
  """Clips the output of a function to [0, 2*pi].

  Used when the final call is a Newton-Raphson iteration
  that may result in out-of-bounds radians.
  """
  @wraps(f)
  def wrapper(*args, **kwargs):
    return jnp.remainder(f(*args, **kwargs), 2 * math.pi)
  return wrapper

