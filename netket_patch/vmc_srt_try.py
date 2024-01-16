import netket.experimental as nkx

from .vmc_sr_try import VMCSRTry
from .vmc_try import VMCTry


class VMCSRtTry(nkx.driver.VMC_SRt):
    def __init__(self, *args, **kwargs):
        self._pre_init(args, kwargs)

        super().__init__(*args, **kwargs)

        self._post_init()

    def _pre_init(self, args, kwargs):
        VMCTry._pre_init(self, args, kwargs)

    def _post_init(self):
        VMCSRTry._post_init(self)

    def _try_lr(self, step_size):
        yield from VMCTry._try_lr(self, step_size)

    def _try_diag_shift(self, step_size):
        yield from VMCSRTry._try_diag_shift(self, step_size)

    def _run_main(self, step_size):
        yield from VMCSRTry._run_main(self, step_size)

    def iter(self, n_steps, step_size=1):  # noqa: A003
        yield from VMCSRTry.iter(self, n_steps, step_size)
