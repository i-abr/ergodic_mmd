# ---------------------------------------------------------------------------------------- #
#                                EQUINOX GP IMPLEMENTATIONS                                #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import equinox as eqx
from jaxtyping import Array, Float, Bool
from typing import Callable
from .kernels import RFF
from jax import jit, vmap
from functools import partial
from jax.lax import cond
import jax.tree_util as jtu
from tensorflow_probability.substrates.jax import distributions as tfd
from tinygp.helpers import JAXArray

# from steinRF.utils import stabilize


# -------------------------------------- GP HELPERS -------------------------------------- #
@jax.jit
def gp_nll(k, X, y, diag=1e-4):
    K = k(X, X) + jnp.eye(X.shape[0]) * diag
    n = y.shape[0]
    L = jnp.linalg.cholesky(K)

    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y))
    term1 = 0.5 * jnp.dot(y.T, alpha)
    term2 = jnp.sum(jnp.log(jnp.diag(L)))
    term3 = 0.5 * n * jnp.log(2 * jnp.pi)
    return term1 + term2 + term3


def gp_pred(k, X, y, X_test, diag=1e-4):
    K = k(X, X) + jnp.eye(X.shape[0]) * diag
    K_star = k(X, X_test)
    K_star_star = k(X_test, X_test) + jnp.eye(X_test.shape[0]) * diag

    L = jnp.linalg.cholesky(K)
    
    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y))
    mu_pred = jnp.dot(K_star.T, alpha)
    v = jnp.linalg.solve(L, K_star)
    sigma_pred = jnp.sqrt(jnp.diag(K_star_star - jnp.dot(v.T, v)))

    return mu_pred, sigma_pred

@jax.jit
def lrgp_nll(k, X, y, diag=1e-4):
    phiX = k.phi(X)
    n, m = phiX.shape
    A = phiX.T @ phiX + jnp.eye(m) * diag
    R = jnp.linalg.cholesky(A)
    y_loss = jnp.linalg.solve(R, phiX.T @ y)

    lml_1 = -((y.T @ y) - (y_loss.T @ y_loss)) / (2 * diag)
    lml_2 = -0.5 * jnp.sum(jnp.log(jnp.diag(R)**2))
    lml_3 = m * jnp.log(m * diag)
    lml_4 = -0.5 * n * jnp.log(2 * jnp.pi * diag)
    lml = lml_1 + lml_2 + lml_3 + lml_4
    return -lml


# ----------------------------------- LOG GAUSSIAN COX ----------------------------------- #
def lgcp_nll(key, k, X, y, volume, diag=1e-4, n_samples=1):
    """Log gaussian cox process"""
    n = y.shape[0]
    mu = jnp.zeros(n)

    #### sample latent f
    K = k(X, X) + jnp.eye(X.shape[0]) * diag
    L = jnp.linalg.cholesky(K)
    z = jax.random.normal(key, (n_samples, n))
    f = mu[:, None] + L.T @ z.T

    #### log gp prior of latent f
    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, f))
    term1 = -0.5 * jax.vmap(
        lambda _f, _alpha: jnp.dot(_f.T, _alpha), (1, 1)
    )(f, alpha)
    term2 = -jnp.sum(jnp.log(jnp.diag(L)))
    term3 = -0.5 * n * jnp.log(2 * jnp.pi)
    log_gp_prior = term1 + term2 + term3
    
    #### log likelihood of poisson
    intensity = jnp.exp(f)
    rates = intensity * volume[:, None]
    log_likelihood = jax.vmap(
        jax.scipy.stats.poisson.logpmf, (None, 1)
    )(y, rates).sum(axis=1)

    return jnp.mean(-log_gp_prior - log_likelihood)


from blackjax.mcmc.marginal_latent_gaussian import (
    init,
    build_kernel,
    svd_from_covariance,
)


## adapted from https://blackjax-devs.github.io/sampling-book/models/LogisticRegressionWithLatentGaussianSampler.html
def lgcp_posterior(key, k, X, y, volumes, diag=1e-4, n_samples=100, n_burnin=1_000):
    K = k(X, X) + jnp.eye(X.shape[0]) * diag
    cov_svd = svd_from_covariance(K)
    U, Gamma, U_t = cov_svd

    #### define likelihood
    def log_likelihood(f):
        #### log likelihood of poisson
        intensity = jnp.exp(f)
        rates = intensity * volumes
        _log_likelihood = jax.scipy.stats.poisson.logpmf(y, rates).sum()
        return _log_likelihood

    #### initialize
    f0 = jnp.zeros(X.shape[0])

    init_fn = lambda x: init(x, log_likelihood, U_t)
    initial_state = init_fn(f0)

    kernel = build_kernel(cov_svd)
    step = lambda k, x, delta: kernel(k, x, log_likelihood, delta)

    def calibration_loop(
        rng_key,
        initial_state,
        initial_delta,
        num_steps,
        update_every=100,
        target=0.5,
        rate=0.5,
    ):
        
        def body(carry):
            i, state, delta, pct_accepted, rng_key = carry
            rng_key, rng_key2 = jax.random.split(rng_key, 2)
            state, info = step(rng_key, state, delta)

            # restart calibration of delta
            j = i % update_every
            pct_accepted = (j * pct_accepted + info.is_accepted) / (j + 1)
            diff = target - pct_accepted
            delta = jax.lax.cond(
                j == 0, lambda _: delta * (1 - diff * rate), lambda _: delta, None
            )

            return i + 1, state, delta, pct_accepted, rng_key2

        _, final_state, final_delta, final_pct_accepted, _ = jax.lax.while_loop(
            lambda carry: carry[0] < num_steps,
            body,
            (0, initial_state, initial_delta, 0.0, rng_key),
        )

        return final_state, final_delta

    def inference_loop(rng_key, initial_delta, initial_state, num_samples, num_burnin):
        rng_key, rng_key2 = jax.random.split(rng_key, 2)

        initial_state, delta = calibration_loop(
            rng_key, initial_state, initial_delta, num_burnin
        )

        @jax.jit
        def one_step(carry, rng_key):
            i, pct_accepted, state = carry
            state, info = step(rng_key, state, delta)
            pct_accepted = (i * pct_accepted + info.is_accepted) / (i + 1)
            return (i + 1, pct_accepted, state), state

        keys = jax.random.split(rng_key, num_samples)
        (_, tota_pct_accepted, _), states = jax.lax.scan(
            one_step, (0, 0.0, initial_state), keys
        )
        return states, tota_pct_accepted
    
    #### run inference
    rng_key, sample_key = jax.random.split(key)
    states, tota_pct_accepted = inference_loop(sample_key, 0.5, initial_state, n_samples, n_burnin)
    
    return states.position
