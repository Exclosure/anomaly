import logging
logging.basicConfig(level=logging.DEBUG)
import jax

# All tests use 64-bit floating point
jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_traceback_filtering", "off")
#jax.config.update('jax_disable_jit', True)
#jax.config.update("jax_check_tracer_leaks", True)
