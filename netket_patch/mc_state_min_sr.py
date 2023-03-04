# TODO: machine_pow is not used?
# TODO: MPI
# TODO: mutable
# TODO: non-holomorphic complex

from functools import partial
from math import sqrt
from typing import Callable, Optional, Tuple

import jax
from flax.core.scope import CollectionFilter
from jax import numpy as jnp
from netket import jax as nkjax
from netket.operator import AbstractOperator
from netket.stats import Stats, statistics
from netket.utils import mpi
from netket.utils.types import Array, PyTree

from .mc_state_simple import MCStateSimple, local_value_kernel


class MCStateMinSR(MCStateSimple):
    # Assuming H is Hermitian and use_covariance == True
    def expect_and_grad(
        self,
        H: AbstractOperator,
        *,
        mutable: CollectionFilter = False,
        use_covariance: Optional[bool] = None
    ) -> Tuple[Stats, PyTree]:
        assert mutable is False
        return _expect_and_grad(
            H,
            local_value_kernel,
            self._apply_fun,
            self.chunk_size,
            self.solver,
            self.diag_shift,
            self.parameters,
            self.model_state,
            self.samples,
        )


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def _expect_and_grad(
    H: AbstractOperator,
    local_value_kernel: Callable,
    apply_fun: Callable,
    chunk_size: int,
    solver: Callable,
    diag_shift: float,
    parameters: PyTree,
    model_state: PyTree,
    σ: Array,
) -> Tuple[Stats, PyTree]:
    n_chains_per_rank, n_batches, hilbert_size = σ.shape
    σ = σ.reshape((n_chains_per_rank * n_batches, hilbert_size))

    _local_value_kernel = partial(
        local_value_kernel, H, apply_fun, {"params": parameters, **model_state}
    )
    if chunk_size:
        _local_value_kernel = nkjax.apply_chunked(
            _local_value_kernel, chunk_size=chunk_size
        )
    E_loc = _local_value_kernel(σ)
    E_stat = statistics(E_loc.reshape((n_chains_per_rank, n_batches)).T)
    E_loc -= E_stat.mean
    n_samples = σ.shape[0] * mpi.n_nodes
    E_loc /= sqrt(n_samples)

    if jnp.iscomplexobj(E_loc):
        mode = "holomorphic"
    else:
        mode = "real"

    O_loc = nkjax.jacobian(
        apply_fun,
        parameters,
        σ,
        model_state,
        mode=mode,
        chunk_size=chunk_size,
        center=True,
        dense=True,
    )

    # E_grad = E_loc @ O_loc

    # S = O_loc.T @ O_loc.conj()
    # S += diag_shift * jnp.eye(S.shape[0])
    # E_O = E_loc @ O_loc
    # E_grad = solver(S, E_O)[0]

    T = O_loc.conj() @ O_loc.T
    T += diag_shift * jnp.eye(T.shape[0])
    T_inv_E = solver(T, E_loc)[0]
    E_grad = T_inv_E @ O_loc

    _, unravel = nkjax.tree_ravel(parameters)
    E_grad = unravel(E_grad)
    E_grad = jax.tree_map(
        lambda x, target: (x if jnp.iscomplexobj(target) else 2 * x.real).astype(
            target.dtype
        ),
        E_grad,
        parameters,
    )

    return E_stat, E_grad
