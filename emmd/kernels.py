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
