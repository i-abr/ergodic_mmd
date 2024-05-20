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
from emmd.kernels import RFF

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

    def __init__(self, key, dims, d_hidden=16, n_hidden=4):
        d_in, d_out = dims

        # nn layers
        nn_keys = jax.random.split(key, n_hidden + 1)
        layers = [eqx.nn.Linear(d_in, d_hidden, key=nn_keys[0])]
        for i in range(n_hidden - 1):
            layers.append(eqx.nn.Linear(d_hidden, d_hidden, key=nn_keys[i + 1]))
        layers.append(eqx.nn.Linear(d_hidden, d_out, key=nn_keys[-1], use_bias=False))
        self.layers = layers

        # scale final output
        self.scale = jnp.log(jnp.ones(d_out))

    @property
    def _scale(self):
        return jnp.exp(self.scale)

    @eqx.filter_jit
    def __call__(self, x: JAXArray) -> JAXArray:
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)
        x = x / self._scale
        return x


class DeepCK(tinygp.kernels.Kernel):
    """
    Deep characteristic kernel.
    
    Liu et al., 2020, "Learning Deep Kernels for Non-Parametric Two-Sample Tests"
    """
    nnk: eqx.Module
    ck: eqx.Module
    epsilon: Float

    def __init__(self, key, dims, epsilon=0.2, lowrank=False, **kwargs):
        #### characteristic kernel
        ls = kwargs.pop("ls", 1.0)
        self.ck = Transform(ARD(jnp.ones(dims[0]) * ls), tinygp.kernels.ExpSquared())

        #### deep kernel
        if lowrank:
            R = kwargs.pop("R", 100)
            deep_output_k = RFF(key, d=dims[1], R=R)
        else:
            deep_output_k = tinygp.kernels.ExpSquared()

        mlp = MLP(key, dims, **kwargs)
        self.nnk = Transform(mlp, deep_output_k)

        #### tradeoff value
        self.epsilon = jnp.log(epsilon)

    @property
    def _epsilon(self):
        return jax.nn.sigmoid(jnp.exp(self.epsilon))

    def evaluate(self, X1: Array, X2: Array) -> Array:
        K_phi = self.nnk.evaluate(X1, X2)
        K_char = self.ck.evaluate(X1, X2)
        eps = self._epsilon
        return ((1 - eps) * K_phi + eps) * K_char


# -------------------------------- GRAPH ATTENTION KERNEL -------------------------------- #
class GATLayer(eqx.Module):
    linear: eqx.Module
    attn: eqx.Module
    dropout_rate: Float
    apply: callable

    def __init__(self, key, dims, n_heads=4, dropout_rate=0.5):

        # TODO - add edge function and dropout
        in_dim, out_dim = dims
        out_per_head = out_dim // n_heads

        keys = jax.random.split(key, 3)
        self.linear = eqx.nn.Linear(in_dim, out_per_head, key=keys[0])
        self.attn = eqx.nn.Linear(2 * out_per_head, 1, key=keys[1])
        self.dropout_rate = dropout_rate

        # define application using jraph
        attention_query_fn = lambda x: jax.vmap(self.linear)(x)
        def attention_logit_fn(sent, received, edges):
            # x = jnp.concatenate([sent, received, edges], axis=-1)
            x = jnp.concatenate([sent, received], axis=-1)
            x = jax.vmap(self.attn)(x)
            return jax.nn.leaky_relu(x)

        self.apply = jraph.GAT(attention_query_fn, attention_logit_fn)

    def __call__(self, key, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        return self.apply(graph)

        # h = jax.vmap(self.linear)(graph.nodes)
        # h = jax.nn.elu(h)  
        # return h

        # Attention mechanism
        h_repeat = jnp.repeat(h, graph.n_node, axis=0)
        h_tiled = jnp.tile(h, (1, graph.n_node)).reshape(-1, h.shape[-1])
        edge_features = jnp.concatenate([h_repeat, h_tiled], axis=-1)

        attn_logits = self.attn_linear(edge_features)
        attn_logits = jax.nn.leaky_relu(attn_logits)
        attn_logits = attn_logits.reshape(graph.n_node, graph.n_node)

        # Masked attention
        mask = jraph.get_adjacency_matrix(graph)
        attn_logits = jnp.where(mask, attn_logits, -jnp.inf)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_weights = eqx.filter_masked_dropout(attn_weights, self.dropout_rate, key)

        # Weighted node features
        h_prime = jnp.matmul(attn_weights, h)
        return h_prime



class GAT(eqx.Module):
    layers: list
    scale: Float[Array, "d"]

    def __init__(self, key, dims, d_hidden=16, n_hidden=4):
        d_in, d_out = dims

        # nn layers
        nn_keys = jax.random.split(key, n_hidden + 1)
        layers = [eqx.nn.Linear(d_in, d_hidden, key=nn_keys[0])]
        for i in range(n_hidden - 1):
            layers.append(eqx.nn.Linear(d_hidden, d_hidden, key=nn_keys[i + 1]))
        layers.append(eqx.nn.Linear(d_hidden, d_out, key=nn_keys[-1], use_bias=False))
        self.layers = layers

        # scale final output
        self.scale = jnp.log(jnp.ones(d_out))

    @property
    def _scale(self):
        return jnp.exp(self.scale)

    @eqx.filter_jit
    def __call__(self, x: JAXArray) -> JAXArray:
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)
        x = x / self._scale
        return x


