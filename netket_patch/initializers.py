from jax import numpy as jnp
from jax import random
from jax.nn.initializers import Initializer
from netket.jax import dtype_real
from netket.utils.types import Array, DType, PRNGKeyT, Shape


def _complex_truncated_normal(
    key: PRNGKeyT, upper: Array, shape: Shape, dtype: DType
) -> Array:
    key_r, key_theta = random.split(key)
    upper = jnp.asarray(upper, dtype=dtype)
    dtype = dtype_real(dtype)
    t = (1 - jnp.exp(-(upper**2))) * random.uniform(key_r, shape, dtype)
    r = jnp.sqrt(-jnp.log(1 - t))
    theta = 2 * jnp.pi * random.uniform(key_theta, shape, dtype)
    out = r * jnp.exp(1j * theta)
    return out


def truncated_normal(stddev: float) -> Initializer:
    def init(key: PRNGKeyT, shape: Shape, dtype: DType) -> Array:
        if jnp.issubdtype(dtype, jnp.floating):
            # constant is stddev of standard normal truncated to (-2, 2)
            _stddev = jnp.asarray(stddev / 0.87962566103423978, dtype)
            return random.truncated_normal(key, -2, 2, shape, dtype) * _stddev
        else:
            # constant is stddev of complex standard normal truncated to 2
            _stddev = jnp.asarray(stddev / 0.95311164380491208, dtype)
            return _complex_truncated_normal(key, 2, shape, dtype) * _stddev

    return init
