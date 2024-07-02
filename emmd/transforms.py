# ---------------------------------------------------------------------------------------- #
#                                      DATA TRANSFORMS                                     #
# ---------------------------------------------------------------------------------------- #
from typing import Any
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
import jraph

import tinygp
from tinygp.helpers import JAXArray
from emmd.kernels import RFF, RBF

DEFAULT_KEY = jax.random.PRNGKey(2024)


# -------------------------------------- TRANSFORMS -------------------------------------- #
class Transform(tinygp.kernels.Kernel):
    transform: eqx.Module
    kernel: tinygp.kernels.Kernel

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.evaluate(self.transform(X1), self.transform(X2))

    def phi(self, X: JAXArray) -> JAXArray:
        return self.kernel.phi(self.transform(X))

class ARD(eqx.Module):
    scale: Float[Array, "d"]

    def __init__(self, scale):
        self.scale = jnp.log(scale)

    @property
    def _scale(self):
        return jnp.exp(self.scale)
    
    def __call__(self, X: JAXArray) -> JAXArray:
        return X / self._scale


# ------------------------------------- DEEP KERNELS ------------------------------------- #
class MLP(eqx.Module):
    layers: list
    scale: Float[Array, "d"]
    dropout: eqx.nn.Dropout = eqx.field(static=True)

    def __init__(self, key, in_dim, out_dim, d_hidden=15, n_hidden=3, dropout=0.):
        # nn layers
        nn_keys = jax.random.split(key, n_hidden + 1)
        layers = [eqx.nn.Linear(in_dim, d_hidden, key=nn_keys[0])]
        for i in range(n_hidden - 1):
            layers.append(eqx.nn.Linear(d_hidden, d_hidden, key=nn_keys[i + 1]))
        layers.append(eqx.nn.Linear(d_hidden, out_dim, key=nn_keys[-1], use_bias=False))
        self.layers = layers

        self.dropout = eqx.nn.Dropout(dropout)

        # scale final output
        self.scale = jnp.log(jnp.ones(out_dim))

    @property
    def _scale(self):
        return jnp.exp(self.scale)

    @eqx.filter_jit
    def evaluate(self, x: JAXArray, key=DEFAULT_KEY) -> JAXArray:
        for layer in self.layers[:-1]:
            key, subkey = jax.random.split(key)
            x = jax.nn.relu(layer(x))
            x = self.dropout(x, key=subkey)
        x = self.layers[-1](x)
        x = x / self._scale
        return x

    @eqx.filter_jit
    def __call__(self, x: JAXArray, key=DEFAULT_KEY) -> JAXArray:
        for layer in self.layers[:-1]:
            key, subkey = jax.random.split(key)
            x = jax.nn.relu(jax.vmap(layer)(x))
            x = self.dropout(x, key=subkey)
        x = jax.vmap(self.layers[-1])(x)
        x = x / self._scale
        return x


class DeepKernel(eqx.Module):
    mlp: eqx.Module
    base: eqx.Module

    def __init__(self, key, in_dim, out_dim, d_hidden=15, n_layers=3, lowrank=True, **kwargs):
        if lowrank:
            R = kwargs.pop("R", 100)
            deep_output_k = RFF(key, d=out_dim, R=R)
        else:
            deep_output_k = RBF(jnp.ones(out_dim))
        dropout = kwargs.pop("dropout", 0.)
        self.mlp = MLP(
            key, in_dim, out_dim, d_hidden=d_hidden, n_hidden=n_layers, dropout=dropout
        )
        self.base = deep_output_k

    @eqx.filter_jit
    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        X1 = self.mlp.evaluate(X1)
        X2 = self.mlp.evaluate(X2)
        K_phi = self.base.evaluate(X1, X2)
        return K_phi

    @eqx.filter_jit
    def phi(self, X: JAXArray) -> JAXArray:
        return self.mlp(X)

    @eqx.filter_jit
    def __call__(self, X1: JAXArray, X2: JAXArray, key=DEFAULT_KEY) -> JAXArray:
        X1 = self.mlp(X1, key=key)
        X2 = self.mlp(X2, key=key)
        K_phi = self.base(X1, X2)
        return K_phi


class MultiDeepKernel(eqx.Module):
    kernels: list
    weights: Float[Array, "n_kernels"]

    def __init__(self, key, in_dim, out_dim, d_hidden=15, n_layers=3, n_kernels=3, **kwargs):
        dropout = kwargs.pop("dropout", 0.)

        ks = []
        for _ in range(n_kernels):
            key, subkey = jax.random.split(key)
            k = DeepKernel(
                subkey, in_dim, out_dim, d_hidden=d_hidden, n_layers=n_layers, 
                dropout=dropout, **kwargs
            )
            ks.append(k)

        self.kernels = ks
        self.weights = jnp.log(jnp.ones(n_kernels) / n_kernels)

    @property
    def _weights(self):
        return jax.nn.softmax(jnp.exp(self.weights))

    @eqx.filter_jit
    def evaluate(self, X1: JAXArray, X2: JAXArray, key=DEFAULT_KEY) -> JAXArray:
        keys = jax.random.split(key, len(self.mlps))
        X1 = jnp.array([mlp.evaluate(X1, key=keys[i]) for i, mlp in enumerate(self.mlps)])
        X2 = jnp.array([mlp.evaluate(X2, key=keys[i]) for i, mlp in enumerate(self.mlps)])
        K_phi = jax.vmap(self.base.evaluate)(X1, X2)
        K_phi = jnp.sum(K_phi * self._weights)

        return K_phi

    @eqx.filter_jit
    def __call__(self, X1: JAXArray, X2: JAXArray, key=DEFAULT_KEY) -> JAXArray:
        keys = jax.random.split(key, len(self.kernels))
        K_phi = jnp.array([
            k(X1, X2, key=keys[i]) for i, k in enumerate(self.kernels)
        ])
        # K_phi = jax.vmap(self.base)(X1, X2)
        K_phi = jnp.sum(K_phi * self._weights[:, None, None], axis=0)
        return K_phi


# ------------------------------ DEEP CHARACTERISTIC KERNELS ----------------------------- #
class DeepCK(eqx.Module):
    """
    Deep characteristic kernel.
    
    Liu et al., 2020, "Learning Deep Kernels for Non-Parametric Two-Sample Tests"
    """
    dk: DeepKernel
    ck: eqx.Module
    epsilon: Float

    def __init__(self, key, in_dim, out_dim, epsilon=0.2, **kwargs):
        #### characteristic kernel
        ls = kwargs.pop("ls", 1.0)
        self.ck = RBF(jnp.ones(in_dim) * ls)
        self.dk = DeepKernel(key, in_dim, out_dim, **kwargs)

        #### tradeoff value
        self.epsilon = jnp.log(epsilon)

    @property
    def _epsilon(self):
        return jax.nn.sigmoid(jnp.exp(self.epsilon))

    @eqx.filter_jit
    def evaluate(self, X1: Array, X2: Array, key=DEFAULT_KEY) -> Array:
        K_phi = self.dk.evaluate(X1, X2, key=key)
        K_char = self.ck.evaluate(X1, X2)
        eps = self._epsilon
        return ((1 - eps) * K_phi + eps) * K_char
    
    @eqx.filter_jit
    def __call__(self, X1: Array, X2: Array, key=DEFAULT_KEY) -> Array:
        K_phi = self.dk(X1, X2, key=key)
        K_char = self.ck(X1, X2)
        eps = self._epsilon
        return ((1 - eps) * K_phi + eps) * K_char


class MultiDeepCK(eqx.Module):
    """
    Deep characteristic kernel.
    
    Liu et al., 2020, "Learning Deep Kernels for Non-Parametric Two-Sample Tests"
    """
    mdk: MultiDeepKernel
    ck: eqx.Module
    epsilon: Float

    def __init__(self, key, in_dim, out_dim, epsilon=0.2, **kwargs):
        #### characteristic kernel
        ls = kwargs.pop("ls", 1.0)

        self.ck = RBF(jnp.ones(in_dim) * ls)
        self.mdk = MultiDeepKernel(key, in_dim, out_dim, **kwargs)

        #### tradeoff value
        self.epsilon = jnp.log(epsilon)

    @property
    def _epsilon(self):
        return jax.nn.sigmoid(jnp.exp(self.epsilon))

    @eqx.filter_jit
    def evaluate(self, X1: Array, X2: Array, key=DEFAULT_KEY) -> Array:
        K_phi = self.mdk.evaluate(X1, X2, key=key)
        K_char = self.ck.evaluate(X1, X2)
        eps = self._epsilon
        return ((1 - eps) * K_phi + eps) * K_char
    
    @eqx.filter_jit
    def __call__(self, X1: Array, X2: Array, key=DEFAULT_KEY) -> Array:
        K_phi = self.mdk(X1, X2, key=key)
        K_char = self.ck(X1, X2)
        eps = self._epsilon
        return ((1 - eps) * K_phi + eps) * K_char

