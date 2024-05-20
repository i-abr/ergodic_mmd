# ---------------------------------------------------------------------------------------- #
#                                      MESH UTILITIES                                      #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
import jraph
import open3d as o3d
from plyfile import PlyData
import pyvista as pv
import numpy as np

# ------------------------------------ MESH UTILITIES ------------------------------------ #
def o3d_mesh_to_pv(mesh):
    mesh = PlyData.read(mesh.path)

    # vertices
    points = np.vstack((
        mesh['vertex']['x'],
        mesh['vertex']['y'],
        mesh['vertex']['z']
    )).T

    # faces
    faces = np.array([np.asarray(face) for face in mesh["face"]["vertex_indices"]])
    n_faces = len(faces)
    faces = np.concatenate(
        [np.ones((len(faces), 1)).astype(int) * 3, faces], axis=1
    )

    return pv.PolyData(points, faces, n_faces=n_faces)


def face_to_edge(nodes, face_inds):
    for face in face_inds:
        for ind1 in face:
            for ind2 in face:
                if ind1 != ind2:
                    delta = nodes[ind1, :] - nodes[ind2, :]
                    yield np.concatenate([delta, [ind1, ind2]])


def o3d_mesh_to_graph(mesh):
    mesh = PlyData.read(mesh.path)
    points = np.vstack((
        mesh['vertex']['x'],
        mesh['vertex']['y'],
        mesh['vertex']['z']
    )).T

    # faces to edges
    faces = np.array([
        np.asarray(face) for face in mesh["face"]["vertex_indices"]
    ])
    edges = jnp.array(list(face_to_edge(points, faces)))
    edge_features = edges[:, :-2]
    receivers = edges[:, -2].astype(int)
    senders = edges[:, -1].astype(int)

    points = jnp.asarray(points)
    n_edge = len(senders)
    n_node = len(points)
    graph = jraph.GraphsTuple(
        nodes=points, senders=senders, receivers=receivers,
        edges=edge_features, n_node=n_node, n_edge=n_edge, globals=None
    )

    return graph


def centroids(mesh):
    pass

# ---------------------------------- GRAPH CONSTRUCTORS ---------------------------------- #
def undirect_edges(senders, receivers, edges, edge_matrix=False):
    new_senders = jnp.concatenate([senders, receivers])
    new_receivers = jnp.concatenate([receivers, senders])
    if edge_matrix:
        edges = edges[new_senders, new_receivers]
    else:
        edges = jnp.concatenate([edges, edges])
    return new_senders, new_receivers, edges


def trajectory_graph(traj, directed=True):
    """line graph of a trajectory"""
    # nodes
    node_features = traj
    n_node = len(traj)

    # edges
    senders = jnp.arange(len(traj)-1)
    receivers = jnp.arange(1, len(traj))
    edges = traj[:-1] - traj[1:]
    if not directed:
        senders, receivers, edges = undirect_edges(senders, receivers, edges)
    n_edge = len(edges)

    graph = jraph.GraphsTuple(
        nodes=node_features, senders=senders, receivers=receivers,
        edges=edges, n_node=n_node, n_edge=n_edge, globals=None
    )

    return graph


def knn_graph(points, k, directed=True):
    n_node, d = points.shape

    # calculate distances
    pairwise_dists = points[:, None, :] - points[None, :, :]
    pairwise_norms = jnp.linalg.norm(pairwise_dists, axis=-1)

    # get k nearest neighbors
    senders = jnp.repeat(jnp.arange(len(points)), k)
    receivers = jnp.argsort(pairwise_norms, axis=-1)[:, 1:k+1].flatten()

    if directed:
        edges = pairwise_dists[senders, receivers]
    else:
        senders, receivers, edges = undirect_edges(
            senders, receivers, pairwise_dists
        )
    n_edge = len(edges)

    graph = jraph.GraphsTuple(
        nodes=points, senders=senders, receivers=receivers,
        edges=edges, n_node=n_node, n_edge=n_edge, globals=None
    )

    return graph


# ----------------------------------- PLOTTING HELPERS ----------------------------------- #
def plot_2d_mesh(graph, ax=None, **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    ax.triplot(graph.nodes[:, 0], graph.nodes[:, 1], graph.edges, **kwargs)
    return ax
