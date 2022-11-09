from jax import lax
from jax import numpy as jnp
from jax import random
from jax.nn.initializers import Initializer
from netket.jax import dtype_real
from netket.utils.types import Array, DType, PRNGKeyT, Shape


# stddev < 1 because of the truncation
def _complex_truncated_normal(
    key: PRNGKeyT, upper: float, shape: Shape, dtype: DType
) -> Array:
    key_r, key_theta = random.split(key)
    dtype = dtype_real(dtype)

    r = -jnp.expm1(-(upper**2))
    r = r * random.uniform(key_r, shape, dtype)
    r = jnp.sqrt(-jnp.log1p(-r))

    theta = 2 * jnp.pi * random.uniform(key_theta, shape, dtype)
    out = r * jnp.exp(1j * theta)
    return out


def truncated_normal(stddev: float, upper: float = 2) -> Initializer:
    def init(key: PRNGKeyT, shape: Shape, dtype: DType) -> Array:
        if jnp.issubdtype(dtype, jnp.floating):
            # `c` is the stddev of standard normal truncated to `(-upper, upper)`
            # When `upper = 2`, `c` becomes the magic number in jax.nn.initializers
            c = jnp.sqrt(2 / jnp.pi) * upper * jnp.exp(-(upper**2) / 2)
            c /= lax.erf(upper / jnp.sqrt(2))
            # Avoid numerical issue when `upper` is small
            c = jnp.where(upper < 1e-4, upper / jnp.sqrt(3), jnp.sqrt(1 - c))

            _stddev = stddev / c
            return random.truncated_normal(key, -upper, upper, shape, dtype) * _stddev
        else:
            # `c` is the stddev of complex standard normal truncated to `upper`
            c = upper**2 / jnp.expm1(upper**2)
            # Avoid numerical issue when `upper` is small
            c = jnp.where(upper < 1e-4, upper / jnp.sqrt(2), jnp.sqrt(1 - c))

            _stddev = stddev / c
            return _complex_truncated_normal(key, upper, shape, dtype) * _stddev

    return init
