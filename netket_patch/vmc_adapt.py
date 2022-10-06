from functools import partial

import jax
import netket as nk
import optax
from optax_patch import AdjustableLRState, ReduceOnPlateauState

from .opt_state import find_state


def log_lr_callback(step, log_data, driver):
    state = find_state(
        driver._optimizer_state, (AdjustableLRState, ReduceOnPlateauState)
    )
    assert state is not None
    log_data["lr"] = state.lr
    if hasattr(state, "t_stat"):
        log_data["t_stat"] = state.t_stat
    return True


class VMCAdapt(nk.VMC):
    def update_parameters(self):
        self._optimizer_state, self.state.parameters = apply_gradient(
            self._optimizer.update,
            self._optimizer_state,
            self._updates,
            self.state.parameters,
            self._loss_stats.mean,
        )


@partial(jax.jit, static_argnums=0)
def apply_gradient(optimizer_fun, optimizer_state, dp, params, loss):
    updates, new_optimizer_state = optimizer_fun(dp, optimizer_state, params, loss)

    new_params = optax.apply_updates(params, updates)
    return new_optimizer_state, new_params
