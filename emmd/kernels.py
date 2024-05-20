# ---------------------------------------------------------------------------------------- #
#                                          KERNELS                                         #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import jit, vmap
from jaxtyping import Array, Float
from typing import Callable
import tinygp
from functools import partial
from tinygp.helpers import JAXArray


# ----------------------------------------- UTILS ---------------------------------------- #
def min_max_dist(X):
    assert jnp.ndim(X) == 2

    # calculate distance max/min
    train_x_sort = jnp.sort(X, axis=-2)
    max_dist = train_x_sort[..., -1, :] - train_x_sort[..., 0, :]
    dists = train_x_sort[..., 1:, :] - train_x_sort[..., :-1, :]
    dists = jnp.where(dists == 0., 1e-10, dists)
    sorted_dists = jnp.sort(dists, axis=-2)
    min_dist = sorted_dists[..., 0, :]

    return min_dist, max_dist


# -------------------------------------- RFF KERNEL -------------------------------------- #
class RFF(tinygp.kernels.base.Kernel):
    w: Float[Array, "R d"]
    R: int = eqx.field(static=True)

    def __init__(self, 
            key: jax.random.PRNGKey, R, d, prior=None
        ):

        self.R = R        
        if prior is None:
            self.w = jax.random.normal(key, (R, d))
        else:
            self.w = prior

    @classmethod
    def initialize_from_data(cls, key, R, X):
        d = X.shape[-1]
        min_dist, _ = min_max_dist(X)
        w = jax.random.uniform(key, (R, d), minval=0, maxval=0.5) / min_dist

        return cls(key, R, d, prior=w)
    
    @jax.jit
    def phi(self, _X):
        cos_feats = jnp.cos(_X @ self.w.T)
        sin_feats = jnp.sin(_X @ self.w.T)
        projected = jnp.concatenate([cos_feats, sin_feats], axis=-1)

        return projected / jnp.sqrt(self.R)
    
    @jax.jit
    def evaluate(self, X1, X2):
        phiX1 = self.phi(X1)
        phiX2 = self.phi(X2)
        return phiX1 @ phiX2.T


# ----------------------------------- NONSTATIONARY RFF ---------------------------------- #
class NonstationaryRFF(tinygp.kernels.base.Kernel):
    w: Float[Array, "R d2"]
    b: Float[Array, "R"]
    R: int = eqx.field(static=True)
    d: int = eqx.field(static=True)

    def __init__(self, key, R, d, prior=None):
        self.R = R
        self.d = d
        
        if prior is None:
            if prior is None:
                self.w = jax.random.normal(key, (2 * R, d))
            else:
                self.w = jnp.concatenate([prior, prior], axis=0)

    @classmethod
    def initialize_from_data(cls, key, R, X):
        d = X.shape[-1]
        min_dist, _ = min_max_dist(X)
        min_dist = jnp.concatenate([min_dist, min_dist], axis=-1)
        w = jax.random.uniform(key, (R, d * 2), minval=0, maxval=0.5) / min_dist

        return cls(key, R, d, prior=w)

    @jit
    def phi(self, _X):
        d, R = self.d, self.R
        w1 = self.w[:, :d]
        w2 = self.w[:, d:]

        # multiply by sqrt 2?
        cos_feats = jnp.cos(_X @ w1.T) + jnp.cos(_X @ w2.T)
        sin_feats = jnp.sin(_X @ w1.T) + jnp.sin(_X @ w2.T)
        projected = jnp.concatenate([cos_feats, sin_feats], axis=-1)

        return projected / jnp.sqrt(4 * R)

    @jit
    def evaluate(self, X1, X2):
        return self.phi(X1) @ self.phi(X2).T


# -------------------------------------- MATERN 3/2 -------------------------------------- #
class M32(tinygp.kernels.base.Kernel):
    @jit
    def evaluate(self, x1, x2):
        l1 = jnp.abs(x1 - x2).sum(axis=-1)
        arg = jnp.sqrt(3) * l1

        val = jnp.where(
            l1 == 0., 1., 
            (1 + arg) * jnp.exp(-arg)
        )

        return val


# --------------------------------------- PERIODIC --------------------------------------- #
class Periodic(tinygp.kernels.base.Kernel):
    """Formulated as a product kernel for N_d"""
    gamma: Float

    def __init__(self, gamma):
        self.gamma = jnp.log(gamma)

    @property
    def _gamma(self):
        return jnp.exp(self.gamma)

    @jit
    def evaluate(self, x1, x2):
        l1 = jnp.abs(x1 - x2)
        val = jnp.where(
            l1 == 0., 1., 
            jnp.exp(-self._gamma * jnp.square(jnp.sin(jnp.pi * l1)))
        )

        return jnp.prod(val)
