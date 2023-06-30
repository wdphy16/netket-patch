from functools import partial, wraps

import jax
from jax import numpy as jnp
from jax import random
from netket.operator._ising.jax import _ising_conn_states_jax
from netket.operator._ising.numba import Ising


@partial(jax.vmap, in_axes=(0, None, None, None))
def _ising_mels_jax(x, edges, h, J):
    same_spins = x[edges[:, 0]] == x[edges[:, 1]]
    mels = jnp.concatenate([J * (2 * same_spins - 1).sum(keepdims=True), -h])
    return mels


@partial(jax.jit, static_argnames=("local_states"))
def _ising_kernel_jax(x, edges, flip, h, J, local_states):
    batch_shape = x.shape[:-1]
    x = x.reshape((-1, x.shape[-1]))

    mels = _ising_mels_jax(x, edges, h, J)
    mels = mels.reshape(batch_shape + mels.shape[1:])

    x_prime = _ising_conn_states_jax(x, flip, local_states)
    x_prime = x_prime.reshape(batch_shape + x_prime.shape[1:])

    return x_prime, mels


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None, None))
def _ising_n_conn_jax(x, edges, h, J):
    same_spins = x[edges[:, 0]] == x[edges[:, 1]]
    # TODO duplicated with _ising_mels_jax
    mels_ZZ = J * (2 * same_spins - 1).sum()
    n_conn_ZZ = mels_ZZ != 0
    return x.size + n_conn_ZZ


class IsingDisorderJax(Ising):
    @wraps(Ising.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._edges_jax = jnp.asarray(self.edges, dtype=jnp.int32)
        self._flip = jnp.eye(self.max_conn_size, self.hilbert.size, k=-1, dtype=bool)

        if len(self.hilbert.local_states) != 2:
            raise ValueError(
                "IsingDisorderJax only supports Hamiltonians with two local states"
            )
        self._hi_local_states = tuple(self.hilbert.local_states)

    def n_conn(self, x):
        return _ising_n_conn_jax(x, self._edges_jax, self.h, self.J)

    def get_conn_padded(self, x):
        return _ising_kernel_jax(
            x, self._edges_jax, self._flip, self.h, self.J, self._hi_local_states
        )

    def sample_disorder(self, key):
        self._h = random.bernoulli(key, shape=(self.hilbert.size,)) * 2 - 1

    def get_disorder(self):
        return self._h
