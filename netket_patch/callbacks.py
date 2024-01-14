import jax
import numpy as np
from jax import numpy as jnp
from jax.tree_util import tree_leaves


def _reduce_and(xs):
    out = True
    for x in xs:
        out = out & x
    return out


@jax.jit
def _all_finite(tree):
    return _reduce_and(jnp.isfinite(x).all() for x in tree_leaves(tree))


def check_finite_callback(step, log_data, driver):
    # Uninitialized caches may be nan intentionally
    return np.isfinite(driver._loss_stats.mean) and _all_finite(driver.state.parameters)
