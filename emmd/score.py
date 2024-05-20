# ---------------------------------------------------------------------------------------- #
#                                      SCORE MATCHING                                      #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
import equinox as eqx
from tensorflow_probability.substrates.jax import distributions as tfd
from jaxtyping import Array, Float
from functools import partial

# -------------------------------------- BASIC MODEL ------------------------------------- #
class ScoreDensity(eqx.Module):
    k: eqx.Module
    mu_q: Float[Array, "d"]
    sigma_q: Float[Array, "d"]
    alpha: Float[Array, "m"]
    z: Float[Array, "m d"]

    def __init__(self, key, k, z, mu=None, sigma=None):
        self.k = k
        R, d = z.shape

        if mu is None:
            mu = jnp.zeros(d)
        self.mu_q = mu
        if sigma is None:
            sigma = jnp.log(jnp.ones(d))
        self.sigma_q = sigma

        self.alpha = jnp.log(jnp.ones(R) / R)
        self.z = z

    @property
    def _alpha(self):
        return jnp.exp(self.alpha)
    
    def _q(self, X):
        sigma = jnp.exp(self.sigma_q)
        return tfd.MultivariateNormalDiag(self.mu_q, sigma).log_prob(X)
    
    def log_density(self, X):  # matrix
        K = self.k(X, self.z)
        return K @ self._alpha + self._q(X)

    def _log_density(self, x):  # vec
        K_val = (self.k(x[None, :], self.z) @ self._alpha).squeeze()
        q_val = self._q(x)
        return K_val + q_val

    def __call__(self, X):
        grad_log_density = jax.grad(self._log_density)
        second_grad_log_density = jax.hessian(self._log_density)
        
        term1 = jax.vmap(second_grad_log_density)(X)
        term1 = jax.vmap(jnp.diag)(term1)
        term2 = 0.5 * jax.vmap(grad_log_density)(X)**2

        score = term1 + term2
        score = jnp.sum(score, axis=-1)
        return jnp.mean(score)
