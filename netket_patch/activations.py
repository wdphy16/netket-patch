from math import ceil, log, pi, sqrt

from jax import numpy as jnp
from jax import random


def log_sinh(x):
    is_neg = x.real < 0
    # sign = 1 if x.real == 0, so that x.imag is unchanged
    sign = -2 * is_neg + 1
    # neg_sign_imag = 1 if x.imag == 0, because log(-1+0j).imag > 0
    neg_sign_imag = -2 * (x.imag > 0) + 1
    x *= sign
    out = x + jnp.log1p(-jnp.exp(-2 * x)) - log(2) + is_neg * neg_sign_imag * pi * 1j
    return out


def init_log_sinh(stddev):
    def init(key, shape, dtype):
        assert len(shape) == 2
        in_size, out_size = shape
        assert out_size % in_size == 0
        alpha = out_size // in_size

        out = log(1 + sqrt(2)) * jnp.eye(in_size, dtype=dtype)
        out = jnp.tile(out, (1, alpha))
        out += stddev * random.normal(key, shape, dtype)
        return out

    return init


def init_sym_log_sinh(stddev):
    def init(key, shape, dtype):
        assert len(shape) == 3
        out_feat, in_feat, in_size = shape
        in_feat_size = in_feat * in_size

        out = log(1 + sqrt(2)) * jnp.eye(in_feat_size, dtype=dtype)
        out = jnp.tile(out, (ceil(out_feat / in_feat_size), 1))
        out = out[:out_feat, :]
        out = out.reshape(shape)
        out += stddev * random.normal(key, shape, dtype)
        return out

    return init
