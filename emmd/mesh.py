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
from emmd.utils import grid

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
    points = np.asarray(mesh.vertices)

    # faces to edges
    faces = np.asarray(mesh.triangles)
    edges = jnp.array(list(face_to_edge(points, faces)))
    edge_features = edges[:, :-2]
    edge_features = jnp.sum(edge_features**2, keepdims=True, axis=-1)
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


def downsample_o3d_mesh(mesh, reduce_factor=(64,)):
    mesh = o3d.io.read_triangle_mesh(mesh.path)
    mesh.compute_vertex_normals()

    for factor in reduce_factor:
        voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / factor
        print(f'voxel_size = {voxel_size:e}')
        mesh = mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average)
        print(f"""
            Simplified mesh has {len(mesh.vertices)} vertices and \
                {len(mesh.triangles)} triangles
        """)

    return mesh


def traj_to_line(traj, color=True):
    T = traj.shape[0]

    poly = pv.PolyData()
    poly.points = np.array(traj)
    cells = np.full((T - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(traj) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(traj), dtype=np.int_)
    poly.lines = cells

    if color:
        poly["scalars"] = list(range(poly.n_points))

    return poly


def gridpt_to_row(grid_dims, grid_inds):
    _, ny, nz = grid_dims
    x, y, z = grid_inds

    row_ind1 = x * ny * nz
    row_ind2 = y * nz
    row_ind3 = z

    return row_ind1 + row_ind2 + row_ind3


def point_in_mesh_fn(mesh, resolution, bounds=None):
    """RBF kernel penalizing distance to mesh surface."""
    # make signed distance function for rectilinear grid of search space
    if bounds is None:
        bounds = mesh.bounds
        bounds = jnp.array(bounds).reshape(3, 2).T
    voxel_size = (bounds[1] - bounds[0]) / (resolution - 1)
    voxel_ball = jnp.linalg.norm(voxel_size)
    x = np.linspace(bounds[0, 0], bounds[1, 0], resolution)
    y = np.linspace(bounds[0, 1], bounds[1, 1], resolution)
    z = np.linspace(bounds[0, 2], bounds[1, 2], resolution)
    flat_grid = grid(bounds, N=resolution)

    rec_grid = pv.RectilinearGrid(x, y, z)
    rec_grid.compute_implicit_distance(mesh, inplace=True)
    distances = jnp.array(rec_grid['implicit_distance'])

    sdf = jnp.where(distances < 0, distances, 0.)
    sdf = jnp.abs(sdf)

    reduce_inds = jnp.where(sdf > 0)
    flat_grid = flat_grid[reduce_inds]
    sdf = sdf[reduce_inds]

    # create function to check for membership of point to voxel
    @jax.jit
    def sdf_point(points):        
        dists = jnp.sum((points[:, None, :] - flat_grid[None, :, :])**2, axis=-1)
        dists /= (-0.5 * (voxel_ball)**2)
        dists = jnp.exp(dists)
        return dists * sdf
    
    return sdf_point


def project_pdf_to_mesh(mesh, pdf):
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
def plot_2d_trajectory_pv(
        pv_mesh, traj, camera_pos=None, notebook=True, 
        color=True, radius=0.001, opacity=0.5
    ):

    traj = jnp.concatenate([traj, jnp.zeros((len(traj), 1))], axis=-1)
    line = traj_to_line(traj, color)
    pl = pv.Plotter(notebook=notebook)

    tube = line.tube(radius=radius)
    pl.add_mesh(pv_mesh, color="lightblue", opacity=opacity)
    pl.add_mesh(tube, smooth_shading=True, opacity=min(1, opacity * 1.5))
    
    if notebook:
        pl.show(jupyter_backend='pythreejs', cpos='xy')
    else:
        pl.show(jupyter_backend='static')



def plot_3d_trajectory_pv(
        pv_mesh, traj, camera_pos=None, notebook=True, 
        color=True, radius=0.001, opacity=0.5
    ):

    line = traj_to_line(traj, color)
    pl = pv.Plotter(notebook=notebook)

    tube = line.tube(radius=radius)
    pl.add_mesh(pv_mesh, color="lightblue", opacity=opacity)
    pl.add_mesh(tube, smooth_shading=True, opacity=min(1, opacity * 1.5))

    if camera_pos is not None:
        pl.camera.roll = camera_pos["roll"]
        pl.camera.elevation = camera_pos["elevation"]
        pl.camera.azimuth = camera_pos["azimuth"]
    
    if notebook:
        pl.show(jupyter_backend='pythreejs')
    else:
        pl.show(jupyter_backend='static')


def plot_multi_3d_traj_pv(
        pv_mesh, trajectories, shape=None, camera_pos=None, notebook=True, 
        color=True, radius=0.001, opacity=0.25
    ):

    if shape is None:
        shape = (1, len(trajectories))

    lines = [traj_to_line(traj, color) for traj in trajectories]
    pl = pv.Plotter(notebook=notebook, shape=shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            pl.subplot(i, j)
            line = lines[i * shape[1] + j]
            tube = line.tube(radius=radius)
            pl.add_mesh(pv_mesh, color="lightblue", opacity=opacity)
            pl.add_mesh(tube, smooth_shading=True, opacity=min(1, opacity * 1.5))

            if camera_pos is not None:
                pl.camera.roll = camera_pos["roll"]
                pl.camera.elevation = camera_pos["elevation"]
                pl.camera.azimuth = camera_pos["azimuth"]

    if notebook:
        pl.show(jupyter_backend='pythreejs')
    else:
        pl.show(jupyter_backend='static')