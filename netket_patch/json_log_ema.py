import time

import netket as nk
from flax import serialization
from flax.core import unfreeze
from optax_patch import ExponentialMovingAverageState

from .opt_state import find_state


def log_ema_callback(step, log_data, driver):
    state = find_state(driver._optimizer_state, ExponentialMovingAverageState)
    assert state is not None
    log_data["ema_state"] = state
    return True


class JsonLogEMA(nk.logging.JsonLog):
    def __call__(self, step, item, variational_state):
        ema_state = item.pop("ema_state")

        old_step = self._old_step
        nk.logging.RuntimeLog.__call__(self, step, item, variational_state)

        # Check if the time from the last flush is higher than the maximum
        # allowed runtime cost of flushing
        elapsed_time = time.time() - self._last_flush_time
        flush_anyway = (self._last_flush_runtime / elapsed_time) < self._autoflush_cost

        if (
            self._steps_notflushed_write % self._write_every == 0
            or step == old_step - 1
            or flush_anyway
        ):
            self._flush_log()

        if (
            self._steps_notflushed_pars % self._save_params_every == 0
            or step == old_step - 1
        ):
            self._flush_params(variational_state, ema_state)

        self._old_step = step
        self._steps_notflushed_write += 1
        self._steps_notflushed_pars += 1

    def _flush_params(self, variational_state, ema_state=None):
        if not self._save_params:
            return

        _time = time.time()

        binary_data = serialization.to_bytes(variational_state.variables)
        with open(self._prefix + ".mpack", "wb") as outfile:
            outfile.write(binary_data)

        if ema_state is not None:
            binary_data = serialization.msgpack_serialize(
                {"params": unfreeze(ema_state.params)}
            )
            with open(self._prefix + "_ema.mpack", "wb") as outfile:
                outfile.write(binary_data)

        self._steps_notflushed_pars = 0
        self._flush_pars_time += time.time() - _time
