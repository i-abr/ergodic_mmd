# ---------------------------------------------------------------------------------------- #
#                                    ERGODIC MMD CLASSES                                   #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import jit, vmap
from jaxtyping import Array, Float
from typing import Callable
from tensorflow_probability import distributions as tfd
import jraph


# ------------------------------------ ERGODIC METRIC ------------------------------------ #




# ------------------------------------ POINT CLOUD MMD ----------------------------------- #
class ImpCloudMMD(eqx.Module):
    """Implicit p(x) MMD for point clouds."""
    k: eqx.Module
    w: Float[Array, "N d"]

    def __call__(self, x):
        t1 = jnp.mean(self.k(self.w, self.w))
        t2 = 2 * jnp.mean(self.k(x, self.w))
        t3 = jnp.mean(self.k(x, x))
        return t1 - t2 + t3

    def two_sample_mmd(self, x1, x2):
        t1 = jnp.mean(self.k(x1, x1))
        t2 = 2 * jnp.mean(self.k(x1, x2))
        t3 = jnp.mean(self.k(x2, x2))
        return t1 - t2 + t3

    @jit
    def power(self, x, alpha=0):
        n = x.shape[0]
        K_xx = self.k(x, x)
        K_pp = self.k(self.w, self.w)
        K_xp = self.k(x, self.w)
        K_px = self.k(self.w, x)

        H = K_xx + K_pp - K_xp - K_px
        mmd_mu = jnp.mean(H, where=jnp.where(jnp.eye(n), 0, 1))
        mmd_var = jnp.sum(H.mean(axis=0)**2) * (4 / n**3) \
            + H.sum()**2 * 4 / n**4 + alpha
        
        return mmd_mu / jnp.sqrt(mmd_var)
    
    @jit
    def two_sample_power(self, x, y, alpha=0):
        n = x.shape[0]
        K_xx = self.k(x, x)
        K_yy = self.k(y, y)
        K_xy = self.k(x, y)
        K_yx = self.k(y, x)

        H = K_xx + K_yy - K_xy - K_yx
        mmd_mu = jnp.mean(H, where=jnp.where(jnp.eye(n), 0, 1))
        mmd_var = jnp.sum(H.mean(axis=0)**2) * (4 / n**3) \
            + H.sum()**2 * 4 / n**4 + alpha
        
        return mmd_mu / jnp.sqrt(mmd_var)


class ExpCloudMMD(eqx.Module):
    """Explict p(x) MMD for point clouds."""
    k: eqx.Module
    px: Callable
    w: Float[Array, "N d"]

    def __init__(self, k: eqx.Module, particles, p=None, norm=True) -> None:
        self.k = k
        self.particles = particles

        # form 
        if p is None:  # initialize as uniform
            _p = lambda _x: jnp.ones(_x.shape[0])
        else:
            if isinstance(p, tfd.Distribution):
                _p = lambda _x: p.prob(_x)
            else:
                _p = lambda _x: vmap(p)(_x)

        self.px = lambda _x: _p(_x) / _p(_x).sum() if norm else _p(_x)

    def __call__(self, x, particles):
        """
        Small difference from general MMD in that we assume access to p(x) implicitly.
        """

        n_p = particles.shape[0]
        px = self.px(x)
        t1 = jnp.mean(self.k(particles, particles))
        t2 = 2 * jnp.sum(px @ self.k(x, particles)) / n_p
        return t1 - t2


# --------------------------------------- MESH MMD --------------------------------------- #
class MeshMMD(eqx.Module):
    """MMD for meshes"""
    k: eqx.Module
    w: jraph.GraphsTuple

    def __init__(self, k, w, edge_update_fn, graph_fn=None):
        if graph_fn is not None:
            w = graph_fn(w)
        self.w = w
        self.k = k

    def edge_update_fn(self, x: jraph.GraphsTuple) -> jraph.GraphsTuple:
        pass

    def __call__(self, x):
        t1 = jnp.mean(self.k(self.w, self.w))
        t2 = 2 * jnp.mean(self.k(x, self.w))
        t3 = jnp.mean(self.k(x, x))
        return t1 - t2 + t3

    def two_sample_mmd(self, x1, x2):
        t1 = jnp.mean(self.k(x1, x1))
        t2 = 2 * jnp.mean(self.k(x1, x2))
        t3 = jnp.mean(self.k(x2, x2))
        return t1 - t2 + t3

    @jit
    def power(self, x, alpha=0):
        n = x.shape[0]
        K_xx = self.k(x, x)
        K_pp = self.k(self.w, self.w)
        K_xp = self.k(x, self.w)
        K_px = self.k(self.w, x)

        H = K_xx + K_pp - K_xp - K_px
        mmd_mu = jnp.mean(H, where=jnp.where(jnp.eye(n), 0, 1))
        mmd_var = jnp.sum(H.mean(axis=0)**2) * (4 / n**3) \
            + H.sum()**2 * 4 / n**4 + alpha
        
        return mmd_mu / jnp.sqrt(mmd_var)
    
    @jit
    def two_sample_power(self, x, y, alpha=0):
        n = x.shape[0]
        K_xx = self.k(x, x)
        K_yy = self.k(y, y)
        K_xy = self.k(x, y)
        K_yx = self.k(y, x)

        H = K_xx + K_yy - K_xy - K_yx
        mmd_mu = jnp.mean(H, where=jnp.where(jnp.eye(n), 0, 1))
        mmd_var = jnp.sum(H.mean(axis=0)**2) * (4 / n**3) \
            + H.sum()**2 * 4 / n**4 + alpha
        
        return mmd_mu / jnp.sqrt(mmd_var)



