import netket as nk
import numpy as np
from netket.jax import dtype_real
from optax_patch import AdjustableLRState, get_slope_t_stat

from .opt_state import find_state, replace_attr


class VMCTry(nk.VMC):
    def __init__(self, *args, **kwargs):
        self.lr_min = kwargs.pop("lr_min")
        self.lr_max = kwargs.pop("lr_max")
        self.lr_decay = kwargs.pop("lr_decay")
        self.lr_grow = kwargs.pop("lr_grow")
        self.n_trials = 3
        self.try_steps = kwargs.pop("try_steps")
        self.run_steps = kwargs.pop("run_steps")
        self.restart_epochs = kwargs.pop("restart_epochs")

        assert 0 < self.lr_min < self.lr_max
        assert 0 < self.lr_decay <= 1
        assert self.lr_grow >= 1
        assert self.n_trials > 0
        assert self.try_steps > 0
        assert self.run_steps > 0
        assert self.restart_epochs > 0

        super().__init__(*args, **kwargs)

        state = find_state(self._optimizer_state, AdjustableLRState)
        assert state is not None
        assert self.lr_min <= state.lr <= self.lr_max
        self._lr_init = state.lr

        self._last_variables = self.state.variables
        self._last_optimizer_state = self._optimizer_state

        self._stage = 0
        self._lr_multipliers = [1, self.lr_decay, self.lr_grow]
        assert len(self._lr_multipliers) == self.n_trials

        # TODO: Use output dtype
        self._losses = [
            np.empty(self.try_steps, dtype=dtype_real(self.state.model.param_dtype))
            for _ in range(self.n_trials)
        ]
        self._trial_variables = [None for _ in range(self.n_trials)]
        self._trial_optimizer_states = [None for _ in range(self.n_trials)]

    def iter(self, n_steps, step_size=1):  # noqa: A003
        epoch_steps = self.n_trials * self.try_steps + self.run_steps
        assert n_steps % epoch_steps == 0

        init_step_count = self._step_count
        while self._step_count < init_step_count + n_steps:
            if self._step_count % (epoch_steps * self.restart_epochs) == 0:
                self._last_lr = self._lr_init

            if 0 <= self._stage < self.n_trials:
                self.state.variables = self._last_variables
                self._optimizer_state = replace_attr(
                    self._last_optimizer_state,
                    AdjustableLRState,
                    {"lr": self._lr_multipliers[self._stage] * self._last_lr},
                )

                for buffer_idx in range(self.try_steps):
                    self._forward_and_backward()
                    if self._step_count % step_size == 0:
                        yield self._step_count

                    self._losses[self._stage][buffer_idx] = self._loss_stats.mean.real

                    self._step_count += 1
                    self.update_parameters()

                self._trial_variables[self._stage] = self.state.variables
                self._trial_optimizer_states[self._stage] = self._optimizer_state
                self._stage += 1

            elif self._stage == self.n_trials:
                idx = np.argmin([get_slope_t_stat(x) for x in self._losses])

                self.state.variables = self._trial_variables[idx]

                lr = self._lr_multipliers[idx] * self._last_lr
                self._last_lr = min(
                    max(lr, self.lr_min / self.lr_decay), self.lr_max / self.lr_grow
                )
                self._optimizer_state = self._trial_optimizer_states[idx]

                for _ in range(self.run_steps):
                    self._forward_and_backward()
                    if self._step_count % step_size == 0:
                        yield self._step_count

                    self._step_count += 1
                    self.update_parameters()

                self._last_variables = self.state.variables
                self._last_optimizer_state = self._optimizer_state
                self._stage = 0

            else:
                raise ValueError(f"Unknown stage: {self._stage}")
