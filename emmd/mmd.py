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
from jax.scipy.stats import multivariate_normal, uniform
import equinox as eqx
from tensorflow_probability.substrates.jax import distributions as tfd
from jaxtyping import Array, Float

from emmd.utils import grid

DEFAULT_KEY = jax.random.PRNGKey(2024)



# ------------------------------------ ERGODIC METRIC ------------------------------------ #
def ergodic_basis_fn(k_vec, traj, lengths, grids, dxdy):
    # coefficients
    fk_vals = jnp.prod(jnp.cos(jnp.pi * k_vec / lengths * traj), axis=1)
    hk = jnp.sqrt(jnp.sum(jnp.square(jnp.prod(jnp.cos(
        jnp.pi * k_vec / lengths * grids), axis=1
    ))) * jnp.prod(dxdy))
    fk_vals /= hk
    
    # integral
    ck = jnp.sum(fk_vals) * 1 / len(traj)
    return ck


def reconstruction_basis_fn(k_vec, lengths, grids, dxdy):
    fk_vals = jnp.prod(jnp.cos(jnp.pi * k_vec / lengths * grids), axis=1)
    hk = jnp.sqrt(jnp.sum(jnp.square(fk_vals)) * jnp.prod(dxdy))
    fk_vals /= hk
    return fk_vals


def ergodic_metric(trajectory, bounds, n_modes, n_per_dim=100):
    n, d = trajectory.shape
    lengths = jnp.abs(bounds[1] - bounds[0])

    # data grids
    grids = grid(bounds, n_per_dim)
    dxdy = lengths / (jnp.array([n_per_dim] * d) -1)

    # create indices
    axes = [jnp.arange(n_modes) for _ in range(d)]
    inds = jnp.meshgrid(*axes)
    inds = jnp.stack(inds, axis=-1).reshape(-1, d)

    # create basis functions
    coefficients = jax.vmap(
        lambda k_vec: ergodic_basis_fn(k_vec, trajectory, lengths, grids, dxdy)
    )(inds)

    # reconstruct pdf
    # return reconstruction_basis_fn(inds[0], lengths, grids, dxdy)
    pdf_recon = jax.vmap(
        lambda k_vec: reconstruction_basis_fn(k_vec, lengths, grids, dxdy)
    )(inds)
    
    pdf_recon = pdf_recon * coefficients[:, None]
    pdf_recon = jnp.sum(pdf_recon, axis=0)

    return pdf_recon



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


# --------------------------------------- SCORE MMD -------------------------------------- #
class ScoreMMD(eqx.Module):
    k: eqx.Module
    alpha: Float[Array, "Z"]
    z: Float[Array, "Z d"]
    q_params: Array
    l: Float[Array, "2"]  # lambda regularization term
    _q: str = eqx.field(static=True)
    w: Float[Array, "N d"]  # particles

    def __init__(self, key, k, w, q="normal", q_params=None, l=None, **kwargs):
        self.k = k
        R, d = w.shape

        # inducing points and particles
        z = kwargs.get("z", None)
        if z is None:
            R = kwargs.get("R", 100)
            z = jax.random.choice(key, w, (R,), replace=False)
        self.z = z
        self.w = w

        #### create q_0
        if q == "normal":
            self._q = "normal"
            if q_params is None:
                q_params = jnp.array([
                    jnp.mean(z, axis=0),
                    jnp.log(jnp.std(z, axis=0))
                ])
        elif q == "uniform":
            self._q = "uniform"
            # turn bounds into midpoint and scale
            if q_params is None:  # without bounds
                loc = jnp.min(z, axis=0)
                ub = jnp.max(z, axis=0)
                scale = jnp.abs(ub - loc)
            else:  # given bounds as q_params
                loc = q_params[0]
                scale = jnp.abs(q_params[1] - q_params[0])
            q_params = jnp.array([loc, jnp.log(scale)])
        else:
            raise ValueError("Invalid q distribution.")
        
        self.q_params = q_params
        if l is None:
            l = jnp.ones(2) * 1e-3
        self.l = jnp.log(l)

        self.alpha = self.compute_alpha(z, key=key)

    #### helpers
    @property
    def _l(self):
        return jnp.exp(self.l)

    #### functions for base distribution
    @property
    def logpdf_q(self):
        if self._q == "normal":
            return self.norm_logpdf
        
        elif self._q == "uniform":
            return self.uni_logpdf

    def uni_logpdf(self, x):
        loc, scale = self.q_params[0], jnp.exp(self.q_params[1])
        logpdf_vals = jnp.log(jnp.prod(
            uniform.pdf(x, loc=loc, scale=scale), axis=-1
        ))
        return logpdf_vals

    def norm_logpdf(self, x):
        loc, scale = self.q_params[0], jnp.exp(self.q_params[1])
        return multivariate_normal.logpdf(x, loc, jnp.diag(scale))

    @eqx.filter_jit
    def _dkdx(self, X, key=DEFAULT_KEY):
        grad_k = jax.jacfwd(lambda x: self.k(x[None, :], self.z))
        dkdx = jax.vmap(grad_k)(X).squeeze(axis=1)
        return dkdx
    
    @eqx.filter_jit
    def _dk2dx2(self, X, key=DEFAULT_KEY, chunk_size=100):
        grad2_k = jax.hessian(lambda x: self.k(x[None, :], self.z))
        grad2_k_vmap = jax.jit(jax.vmap(
            lambda x: jnp.diagonal(grad2_k(x), axis1=-2, axis2=-1)
        ))

        n_chunks = X.shape[0] // chunk_size
        if X.shape[0] % chunk_size != 0:
            n_chunks += 1
        
        dk2dx2 = []
        for i in range(n_chunks):
            start, end = i * chunk_size, (i + 1) * chunk_size
            dk2dx2_chunk = grad2_k_vmap(X[start:end])
            dk2dx2.append(dk2dx2_chunk)

        dk2dx2 = jnp.concatenate(dk2dx2, axis=0).squeeze(axis=1)
        return dk2dx2

    @eqx.filter_jit
    def compute_alpha(self, X, key=DEFAULT_KEY, chunk_size=100):
        # gradients of kernels
        dkdx = self._dkdx(X, key=key)
        l1, l2 = self._l

        # chunk hessian computation
        dk2dx2 = self._dk2dx2(X, key=key, chunk_size=chunk_size)

        # gradients of q_0
        grad_q = jax.grad(self.logpdf_q)
        dqdx = jax.vmap(grad_q)(X)
        grad2_q = jax.jit(jax.hessian(self.logpdf_q))
        dq2dx2 = jax.vmap(
            lambda x: jnp.diagonal(grad2_q(x), axis1=-2, axis2=-1)
        )(X)

        # compute alpha
        G = jnp.mean(jnp.einsum("nij,nlp->nil", dkdx, dkdx), axis=0)
        U = jnp.mean(jnp.einsum("nij,nlp->nil", dk2dx2, dk2dx2), axis=0)
        b1 = dk2dx2 + dqdx[:, None, :] * dkdx
        b2 = l2 * dk2dx2 * dq2dx2[:, None, :]
        b = jnp.mean(jnp.sum(b1 + b2, axis=-1), axis=0)

        # solve
        A = G + l1 * jnp.eye(G.shape[0]) + l2 * U
        alpha = jnp.linalg.solve(A, b)

        return alpha

    #### functions to compute score
    @eqx.filter_jit
    def log_density(self, X, key=DEFAULT_KEY):  # matrix
        K = self.k(X, self.z)
        return K @ self.alpha + self.logpdf_q(X)

    @eqx.filter_jit
    def _log_density(self, x, key=DEFAULT_KEY):  # vec
        K_val = (self.k(x[None, :], self.z) @ self.alpha).squeeze()
        q_val = self.logpdf_q(x)
        return K_val + q_val
    
    @eqx.filter_jit
    def score(self, X, key=DEFAULT_KEY):
        l1, l2 = self._l
        grad_log_density = jax.grad(lambda x: self._log_density(x))
        second_grad_log_density = jax.hessian(lambda x: self._log_density(x))
        
        term1 = jax.vmap(second_grad_log_density)(X)
        term1 = jax.vmap(jnp.diag)(term1)
        term2 = 0.5 * jax.vmap(grad_log_density)(X)**2
        term3 = l1 / 2 * jnp.sum(self.alpha**2)
        term3 = l2 / (2 * X.shape[0]) * jnp.sum(term1)**2

        score = term1 + term2
        score = jnp.sum(score, axis=-1)
        return jnp.mean(score) + term3

    @eqx.filter_jit
    def two_sample_mmd(self, x1, x2):
        t1 = jnp.mean(self.k(x1, x1))
        t2 = 2 * jnp.mean(self.k(x1, x2))
        t3 = jnp.mean(self.k(x2, x2))
        return t1 - t2 + t3
    
    @eqx.filter_jit
    def __call__(self, x):
        t1 = jnp.mean(self.k(self.w, self.w))
        t2 = 2 * jnp.mean(self.k(x, self.w))
        t3 = jnp.mean(self.k(x, x))
        return t1 - t2 + t3
