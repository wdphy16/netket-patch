# TODO: MPI

from functools import cache, partial
from typing import Callable, Optional, Tuple

import jax
from flax.core.scope import CollectionFilter
from jax import numpy as jnp
from jax.experimental.sparse import BCOO
from netket import jax as nkjax
from netket.operator import AbstractOperator
from netket.stats import Stats
from netket.utils.types import Array, PyTree
from netket.vqs import FullSumState


class FullSumStateSimple(FullSumState):
    def expect(self, H: AbstractOperator) -> Stats:
        return _expect(
            H,
            self._apply_fun,
            self.chunk_size,
            self.parameters,
            self.model_state,
            self._all_states,
        )

    def expect_and_grad(
        self,
        H: AbstractOperator,
        *,
        mutable: CollectionFilter = False,
        use_covariance: Optional[bool] = None
    ) -> Tuple[Stats, PyTree]:
        O_stat, O_grad, new_model_state = _expect_and_grad(
            H,
            self._apply_fun,
            mutable,
            self.chunk_size,
            self.parameters,
            self.model_state,
            self._all_states,
        )

        if mutable is not False:
            self.model_state = new_model_state

        return O_stat, O_grad


@cache
def sparsify(H):
    return BCOO.from_scipy_sparse(H.to_sparse())


@partial(jax.jit, static_argnums=(0, 1, 2))
def _expect(
    H: AbstractOperator,
    apply_fun: Callable,
    chunk_size: int,
    parameters: PyTree,
    model_state: PyTree,
    σ: Array,
) -> Stats:
    _apply_fun = partial(apply_fun, {"params": parameters, **model_state})
    if chunk_size:
        _apply_fun = nkjax.apply_chunked(_apply_fun, chunk_size=chunk_size)
    Ψ = _apply_fun(σ)
    Ψ /= jnp.sqrt(Ψ.conj() @ Ψ)

    H = sparsify(H)
    OΨ = H @ Ψ
    mean = Ψ.conj() @ OΨ

    OΨ_centered = OΨ - mean * Ψ
    variance = (OΨ_centered.conj() @ OΨ_centered).real

    return Stats(mean=mean, error_of_mean=0, variance=variance)


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _expect_and_grad(
    H: AbstractOperator,
    apply_fun: Callable,
    mutable: CollectionFilter,
    chunk_size: int,
    parameters: PyTree,
    model_state: PyTree,
    σ: Array,
) -> Tuple[Stats, PyTree, PyTree]:
    _apply_fun = partial(apply_fun, {"params": parameters, **model_state})
    if chunk_size:
        _apply_fun = nkjax.apply_chunked(_apply_fun, chunk_size=chunk_size)
    Ψ = _apply_fun(σ)
    Ψ /= jnp.sqrt(Ψ.conj() @ Ψ)

    H = sparsify(H)
    OΨ = H @ Ψ
    mean = Ψ.conj() @ OΨ

    OΨ_centered = OΨ - mean * Ψ
    variance = (OΨ_centered.conj() @ OΨ_centered).real

    ΔOΨ = OΨ_centered.conj() * Ψ
    is_mutable = mutable is not False
    _, vjp_fun, *new_model_state = nkjax.vjp(
        lambda w: apply_fun({"params": w, **model_state}, σ, mutable=mutable),
        parameters,
        conjugate=True,
        has_aux=is_mutable,
    )
    O_grad = vjp_fun(ΔOΨ)[0]

    new_model_state = new_model_state[0] if is_mutable else None

    return (
        Stats(mean=mean, error_of_mean=0, variance=variance),
        O_grad,
        new_model_state,
    )
