from functools import partial

import jax
from jax.tree_util import tree_map


@partial(jax.jit, static_argnums=1)
def find_state(state, Type):
    if isinstance(state, Type):
        return state

    if isinstance(state, tuple):
        for _state in state:
            out = find_state(_state, Type)
            if out is not None:
                return out

    if isinstance(state, dict):
        for _state in state.values():
            out = find_state(_state, Type)
            if out is not None:
                return out

    return None


@partial(jax.jit, static_argnums=1)
def replace_attr(state, Type, attr_dict):
    return tree_map(
        lambda x: x._replace(**attr_dict) if isinstance(x, Type) else x,
        state,
        is_leaf=lambda x: isinstance(x, Type),
    )
