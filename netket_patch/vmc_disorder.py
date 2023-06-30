import netket as nk
from jax import random
from netket import jax as nkjax


class VMCDisorder(nk.VMC):
    def __init__(self, *args, **kwargs):
        if "ham_seed" in kwargs:
            ham_seed = kwargs.pop("ham_seed")
        else:
            ham_seed = None
        self.key_ham = nkjax.PRNGKey(ham_seed)

        super().__init__(*args, **kwargs)

    def _forward_and_backward(self):
        self.key_ham, key = random.split(self.key_ham)
        self._ham.sample_disorder(key)

        super()._forward_and_backward()
