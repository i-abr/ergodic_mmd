# ---------------------------------------------------------------------------------------- #
#                                     UTILITY FUNCTIONS                                    #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp


# ----------------------------------------- DATA ----------------------------------------- #
def grid_inds(bounds, N, center=False):
    """
    Create a grid of points for signal processeing and FFTs.
    """
    d = bounds.shape[-1]
    if N is not None:
        if isinstance(N, int):
            N = [N] * d
        axes = [
            jnp.linspace(bounds[0, i], bounds[1, i], N[i])
            for i in range(d)
        ]

    return axes


def grid(bounds, N=None, flatten=True):
    """
    Create a grid of points for signal processeing and FFTs.
    """
    _axes = grid_inds(bounds, N=N)

    _grid = jnp.meshgrid(*_axes)
    _grid = jnp.stack(_grid, axis=-1)
    if flatten:
        _grid = _grid.reshape(-1, _grid.shape[-1])

    return _grid


def bin_data(data: jax.Array, bounds: jax.Array, bins: tuple):
    """
    Bin data into counts, return centroids and counts.
    """    
    # Use histogramdd to bin data
    d = bounds.shape[-1]
    bounds = [bounds[:, i].tolist() for i in range(bounds.shape[1])]
    counts, edges = jnp.histogramdd(data, bins=bins, range=bounds)
    counts = counts.T.flatten()

    edges = [jnp.concat([e[:1], e[1:-1].repeat(2), e[-1:]]) for e in edges]
    edges = [e.reshape(-1, d) for e in edges]
    centroids = [jnp.mean(e, axis=-1) for e in edges]
    centroids = jnp.stack(jnp.meshgrid(*centroids), axis=-1).reshape(-1, d)

    volumes = [jnp.abs(e[...,1] - e[..., 0]) for e in edges]
    volumes = jnp.stack(jnp.meshgrid(*volumes), axis=-1).reshape(-1, d)
    volumes = jnp.prod(volumes, axis=-1)

    return centroids, counts, volumes
