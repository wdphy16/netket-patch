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


def build_check_finite_snapshot_callback(num_snapshots=10, snapshot_step=100):
    snapshots = []

    def callback(step, log_data, driver):
        if check_finite_callback(step, log_data, driver):
            if step % snapshot_step == 0:
                snapshot = {
                    "log_data": log_data,
                    "variables": driver.state.variables,
                    "optimizer_state": driver._optimizer_state,
                }
                snapshots.append(snapshot)
                if len(snapshots) > num_snapshots:
                    del snapshots[0]
        else:
            if len(snapshots) <= 0:
                print("Failed to recover from snapshots")
                return False

            snapshot = snapshots.pop()
            for k, v in snapshot["log_data"].items():
                log_data[k] = v
            driver.state.variables = snapshot["variables"]
            driver._optimizer_state = snapshot["optimizer_state"]

            log_data["non_finite"] = True

        return True

    return callback
