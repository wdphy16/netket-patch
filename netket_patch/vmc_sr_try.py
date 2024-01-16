import netket as nk
import numpy as np
from optax_patch import AdjustableLRState, get_slope_t_stat

from .opt_state import find_state
from .vmc_try import VMCTry


def log_diag_shift_callback(step, log_data, driver):
    log_data["diag_shift"] = driver.diag_shift
    return True


class VMCSRTry(nk.VMC):
    def __init__(self, *args, **kwargs):
        self._pre_init(args, kwargs)
        self.diag_shift = kwargs.pop("diag_shift")

        super().__init__(*args, **kwargs)

        self._post_init()
        self.state.diag_shift = self.diag_shift

    def _pre_init(self, args, kwargs):
        VMCTry._pre_init(self, args, kwargs)

    def _post_init(self):
        VMCTry._post_init(self)
        assert self.diag_shift > 0
        self._last_diag_shift = self._diag_shift_init = self.diag_shift

    def _try_lr(self, step_size):
        yield from VMCTry._try_lr(self, step_size)

    def _try_diag_shift(self, step_size):
        if self._stage == self.n_trials:
            slopes = [get_slope_t_stat(x) for x in self._losses]
            idx = np.argmin(slopes)

            lr = self._lr_multipliers[idx] * self._last_lr
            self._last_lr = min(
                max(lr, self.lr_min / self.lr_decay),
                self.lr_max / self.lr_growth,
            )

            self._losses[0] = self._losses[idx]
            self._trial_variables[0] = self._trial_variables[idx]
            self._trial_optimizer_states[0] = self._trial_optimizer_states[idx]

        idx = self._stage - self.n_trials + 1

        self.state.variables = self._last_variables
        self._optimizer_state = self._trial_optimizer_states[0]
        self.diag_shift = self._lr_multipliers[idx] * self._last_diag_shift

        for buffer_idx in range(self.try_steps):
            dp = self._forward_and_backward()
            if self._step_count % step_size == 0:
                yield self._step_count

            self._losses[idx][buffer_idx] = self._loss_stats.mean.real

            self._step_count += 1
            self.update_parameters(dp)

        self._trial_variables[idx] = self.state.variables
        self._trial_optimizer_states[idx] = self._optimizer_state
        self._stage += 1

    def _run_main(self, step_size):
        slopes = [get_slope_t_stat(x) for x in self._losses]
        idx = np.argmin(slopes)

        self.state.variables = self._trial_variables[idx]
        self._optimizer_state = self._trial_optimizer_states[idx]

        self.diag_shift = self._lr_multipliers[idx] * self._last_diag_shift
        self._last_diag_shift = min(
            max(self.diag_shift, self.lr_min / self.lr_decay),
            self.lr_max / self.lr_growth,
        )

        print(
            "lr",
            find_state(self._optimizer_state, AdjustableLRState).lr,
            "diag_shift",
            self.diag_shift,
            "slope",
            slopes[idx],
        )

        for _ in range(self.run_steps):
            dp = self._forward_and_backward()
            if self._step_count % step_size == 0:
                yield self._step_count

            self._step_count += 1
            self.update_parameters(dp)

        self._last_variables = self.state.variables
        self._last_optimizer_state = self._optimizer_state
        self._stage = 0

    def iter(self, n_steps, step_size=1):  # noqa: A003
        epoch_steps = (self.n_trials * 2 - 1) * self.try_steps + self.run_steps
        assert n_steps % epoch_steps == 0

        init_step_count = self._step_count
        while self._step_count < init_step_count + n_steps:
            restart_steps = epoch_steps * self.restart_epochs
            if restart_steps and self._step_count % restart_steps == 0:
                self._last_lr = self._lr_init
                self._last_diag_shift = self._diag_shift_init

            if 0 <= self._stage < self.n_trials:
                yield from self._try_lr(step_size)
            elif self._stage < self.n_trials * 2 - 1:
                yield from self._try_diag_shift(step_size)
            elif self._stage == self.n_trials * 2 - 1:
                yield from self._run_main(step_size)
            else:
                raise ValueError(f"Unknown stage: {self._stage}")

    def _forward_and_backward(self):
        self.state.diag_shift = self.diag_shift

        return super()._forward_and_backward()
