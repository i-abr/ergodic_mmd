# ---------------------------------------------------------------------------------------- #
#                                      SCORE MATCHING                                      #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal, uniform
import equinox as eqx
from tensorflow_probability.substrates.jax import distributions as tfd
from jaxtyping import Array, Float
from functools import partial


from emmd.kernels import RFF, RBF
DEFAULT_KEY = jax.random.PRNGKey(2024)


# -------------------------------------- BASIC MODEL ------------------------------------- #
class ScoreKernel(eqx.Module):
    k: eqx.Module
    alpha: Float[Array, "Z"]
    z: Float[Array, "Z d"]
    q_params: Array
    _q: str = eqx.field(static=True)
    l: Float[Array, "2"]  # lambda regularization term

    def __init__(self, key, k, z, q="normal", q_params=None, l=None):
        self.k = k
        R, d = z.shape

        self.alpha = jnp.log(jnp.ones(R) / R)
        self.z = z

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

    #### helpers
    @property
    def _l(self):
        return jnp.exp(self.l)

    @property
    def _alpha(self):
        return jnp.exp(self.alpha)

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
        grad_k = jax.jacfwd(lambda x: self.k(x[None, :], self.z, key=key))
        dkdx = jax.vmap(grad_k)(X).squeeze(axis=1)
        return dkdx
    
    # @eqx.filter_jit
    def _dk2dx2(self, X, key=DEFAULT_KEY, chunk_size=100):
        grad2_k = jax.hessian(lambda x: self.k(x[None, :], self.z, key=key))
        grad2_k_vmap = jax.vmap(
            lambda x: jnp.diagonal(grad2_k(x), axis1=-2, axis2=-1)
        )
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

    def compute_alpha_params(self, X, key=DEFAULT_KEY, chunk_size=100):
        # gradients of kernels
        dkdx = self._dkdx(X, key=key)
        return dkdx

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
        b2 = self._l[1] * dk2dx2 * dq2dx2[:, None, :]
        b = jnp.mean(jnp.sum(b1 + b2, axis=-1), axis=0)

        return G, U, b

    #### functions to compute score
    def log_density(self, X, key=DEFAULT_KEY):  # matrix
        K = self.k(X, self.z, key=key)
        return K @ self._alpha + self.logpdf_q(X)

    def _log_density(self, x, key=DEFAULT_KEY):  # vec
        K_val = (self.k(x[None, :], self.z, key=key) @ self._alpha).squeeze()
        q_val = self.logpdf_q(x)
        return K_val + q_val
    
    def score(self, X, key=DEFAULT_KEY):
        grad_log_density = jax.grad(lambda x: self._log_density(x, key=key))
        second_grad_log_density = jax.hessian(lambda x: self._log_density(x, key=key))
        
        term1 = jax.vmap(second_grad_log_density)(X)
        term1 = jax.vmap(jnp.diag)(term1)
        term2 = 0.5 * jax.vmap(grad_log_density)(X)**2
        term3 = self._L / (2 * X.shape[0]) * jnp.sum(term1)**2

        score = term1 + term2 
        score = jnp.sum(score, axis=-1)
        return jnp.mean(score) + term3

    def __call__(self, X1: Array, X2: Array, key=DEFAULT_KEY) -> Array:
        # return self.k(X1, X2)
        # rewrite with respect to Z
        pass


# # -------------------------------------- BASIC MODEL ------------------------------------- #
# class ScoreDensity(eqx.Module):
#     k: eqx.Module
#     alpha: Float[Array, "m"]
#     z: Float[Array, "m d"]
#     q_params: Array
#     _q: str = eqx.field(static=True)
#     L: Float[Array, "l"]

#     def __init__(self, key, k, z, q="normal", q_params=None, L=None):
#         self.k = k
#         R, d = z.shape

#         self.alpha = jnp.log(jnp.ones(R) / R)
#         self.z = z

#         #### create q_0
#         if q == "normal":
#             self._q = "normal"
#             if q_params is None:
#                 q_params = jnp.array([
#                     jnp.mean(z, axis=0),
#                     jnp.log(jnp.std(z, axis=0))
#                 ])
#         elif q == "uniform":
#             self._q = "uniform"
#             # turn bounds into midpoint and scale
#             if q_params is None:  # without bounds
#                 loc = jnp.min(z, axis=0)
#                 ub = jnp.max(z, axis=0)
#                 scale = jnp.abs(ub - loc)
#             else:  # given bounds as q_params
#                 loc = q_params[0]
#                 scale = jnp.abs(q_params[1] - q_params[0])
#             q_params = jnp.array([loc, jnp.log(scale)])
#         else:
#             raise ValueError("Invalid q distribution.")
        
#         self.q_params = q_params
#         if L is None:
#             L = jnp.array(1e-2)
#         self.L = jnp.log(L)

#     @property
#     def _L(self):
#         return jnp.exp(self.L)

#     @property
#     def _alpha(self):
#         return jnp.exp(self.alpha)

#     @property
#     def logpdf_q(self):
#         if self._q == "normal":
#             return self.norm_logpdf
        
#         elif self._q == "uniform":
#             return self.uni_logpdf

#     def uni_logpdf(self, x):
#         loc, scale = self.q_params[0], jnp.exp(self.q_params[1])
#         logpdf_vals = jnp.log(jnp.prod(
#             uniform.pdf(x, loc=loc, scale=scale), axis=-1
#         ))
#         return logpdf_vals

#     def norm_logpdf(self, x):
#         loc, scale = self.q_params[0], jnp.exp(self.q_params[1])
#         return multivariate_normal.logpdf(x, loc, jnp.diag(scale))

#     def log_density(self, X):  # matrix
#         K = self.k(X, self.z)
#         return K @ self._alpha + self.logpdf_q(X)

#     def _log_density(self, x):  # vec
#         K_val = (self.k(x[None, :], self.z) @ self._alpha).squeeze()
#         q_val = self.logpdf_q(x)
#         return K_val + q_val

#     def __call__(self, X):
#         grad_log_density = jax.grad(self._log_density)
#         second_grad_log_density = jax.hessian(self._log_density)
        
#         term1 = jax.vmap(second_grad_log_density)(X)
#         term1 = jax.vmap(jnp.diag)(term1)
#         term2 = 0.5 * jax.vmap(grad_log_density)(X)**2
#         term3 = self._L / (2 * X.shape[0]) * jnp.sum(term1)**2

#         score = term1 + term2 
#         score = jnp.sum(score, axis=-1)
#         return jnp.mean(score) + term3


