import jax
import numpy as np
from jax import numpy as jnp
from jax.tree_util import tree_leaves


def _reduce_and(a):
    out = 1
    for x in a:
        out = out * x
    return out


@jax.jit
def _all_finite(tree):
    return _reduce_and(jnp.isfinite(x).all() for x in tree_leaves(tree))


def build_check_finite_callback(num_snapshots=3):
    snapshots = []

    def callback(step, log_data, driver):
        # Uninitialized caches may be nan intentionally
        if np.isfinite(driver._loss_stats.mean) and _all_finite(
            driver.state.parameters
        ):
            snapshot = {
                "log_data": log_data,
                "variables": driver.state.variables,
                "loss_stats": driver._loss_stats,
                "loss_grad": driver._loss_grad,
                "updates": driver._updates,
                "optimizer_state": driver._optimizer_state,
            }

            snapshots.append(snapshot)
            if len(snapshots) > num_snapshots:
                del snapshots[0]
        else:
            if len(snapshots) <= 0:
                raise RuntimeError("Failed to recover from snapshots")
            snapshot = snapshots.pop()

            for k, v in snapshot["log_data"].items():
                log_data[k] = v
            driver.state.variables = snapshot["variables"]
            driver._loss_stats = snapshot["loss_stats"]
            driver._loss_grad = snapshot["loss_grad"]
            driver._updates = snapshot["updates"]
            driver._optimizer_state = snapshot["optimizer_state"]

            log_data["non_finite"] = True

        return True

    return callback
