# TODO: machine_pow is not used?

from functools import partial
from typing import Callable, Optional, Tuple

import jax
from flax.core.scope import CollectionFilter
from jax import numpy as jnp
from netket import jax as nkjax
from netket.operator import AbstractOperator
from netket.stats import Stats, statistics
from netket.utils import mpi
from netket.utils.types import Array, PyTree
from netket.vqs import MCState


@partial(jax.vmap, in_axes=(None, None, None, 0), out_axes=0)
def local_value_kernel(
    Ô: AbstractOperator, logpsi: Callable, pars: PyTree, σ: Array
) -> Array:
    σp, mels = Ô.get_conn_padded(σ)
    return jnp.sum(mels * jnp.exp(logpsi(pars, σp) - logpsi(pars, σ)))


class MCStateSimple(MCState):
    def expect(self, Ô: AbstractOperator) -> Stats:
        return _expect(
            Ô,
            local_value_kernel,
            self._apply_fun,
            self.chunk_size,
            self.sampler.machine_pow,
            self.parameters,
            self.model_state,
            self.samples,
        )

    # Assuming Ô is Hermitian and use_covariance == True
    def expect_and_grad(
        self,
        Ô: AbstractOperator,
        *,
        mutable: CollectionFilter = False,
        use_covariance: Optional[bool] = None
    ) -> Tuple[Stats, PyTree]:
        Ō, Ō_grad, new_model_state = _expect_and_grad(
            Ô,
            local_value_kernel,
            self._apply_fun,
            mutable,
            self.chunk_size,
            self.sampler.machine_pow,
            self.parameters,
            self.model_state,
            self.samples,
        )

        if mutable is not False:
            self.model_state = new_model_state

        return Ō, Ō_grad


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _expect(
    Ô: AbstractOperator,
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    chunk_size: int,
    machine_pow: int,
    parameters: PyTree,
    model_state: PyTree,
    σ: Array,
) -> Stats:
    n_chains_per_rank, n_batches, hilbert_size = σ.shape
    σ = σ.reshape((n_chains_per_rank * n_batches, hilbert_size))

    _local_value_kernel = partial(
        local_value_kernel, Ô, model_apply_fun, {"params": parameters, **model_state}
    )
    if chunk_size:
        _local_value_kernel = nkjax.apply_chunked(
            _local_value_kernel, chunk_size=chunk_size
        )
    O_loc = _local_value_kernel(σ)

    Ō = statistics(O_loc.reshape((n_chains_per_rank, n_batches)).T)
    return Ō


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def _expect_and_grad(
    Ô: AbstractOperator,
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    mutable: CollectionFilter,
    chunk_size: int,
    machine_pow: int,
    parameters: PyTree,
    model_state: PyTree,
    σ: Array,
) -> Tuple[Stats, PyTree]:
    n_chains_per_rank, n_batches, hilbert_size = σ.shape
    σ = σ.reshape((n_chains_per_rank * n_batches, hilbert_size))
    n_samples = σ.shape[0] * mpi.n_nodes

    _local_value_kernel = partial(
        local_value_kernel, Ô, model_apply_fun, {"params": parameters, **model_state}
    )
    if chunk_size:
        _local_value_kernel = nkjax.apply_chunked(
            _local_value_kernel, chunk_size=chunk_size
        )
    O_loc = _local_value_kernel(σ)

    Ō = statistics(O_loc.reshape((n_chains_per_rank, n_batches)).T)

    O_loc -= Ō.mean

    is_mutable = mutable is not False
    _, vjp_fun, *new_model_state = nkjax.vjp(
        lambda w: model_apply_fun({"params": w, **model_state}, σ, mutable=mutable),
        parameters,
        conjugate=True,
        has_aux=is_mutable,
    )
    Ō_grad = vjp_fun(jnp.conjugate(O_loc) / n_samples)[0]
    Ō_grad = jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], Ō_grad)
    Ō_grad = jax.tree_map(
        lambda x, target: (x if jnp.iscomplexobj(target) else 2 * x.real).astype(
            target.dtype
        ),
        Ō_grad,
        parameters,
    )

    new_model_state = new_model_state[0] if is_mutable else None

    return Ō, Ō_grad, new_model_state
