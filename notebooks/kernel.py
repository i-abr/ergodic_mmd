import jax 
import jax.numpy as np 
from jax import vmap, pmap

def RBF(x, xp, h=0.01):
    return np.exp(
        -np.sum((x-xp)**2)/h
    )

def lie_RBF(x, xp, h=0.01):
    return np.exp(
        -np.sum((x-xp)**2)/h
    )

def create_kernel_matrix(kernel):
    return vmap(vmap(kernel, in_axes=(0, None, None)), in_axes=(None, 0, None))