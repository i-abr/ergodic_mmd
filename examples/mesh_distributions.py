from plyfile import PlyData
import jax 
import jax.numpy as np 
import numpy as onp
import open3d as o3d 
import trimesh as tm

class MeshTemplate(object):
    def __init__(self) -> None:
        pass

class BunnyMesh(object):
    def __init__(self) -> None:
        self.dataset = o3d.data.BunnyMesh()
        plydata = PlyData.read(self.dataset.path)
        verts = np.vstack((
            plydata['vertex']['x'],
            plydata['vertex']['y'],
            plydata['vertex']['z']
        )).T

        _min_pnt = verts.min(axis=0)
        _max_pnt = verts.max(axis=0)
        _mid_pnt = verts.mean(axis=0)
        verts = (verts - _mid_pnt)/(_max_pnt-_min_pnt)

        faces = np.array(onp.vstack(plydata['face']['vertex_indices']))

        self.mesh = tm.Trimesh(vertices=verts, faces=faces)
        self.verts = self.mesh.vertices 
        self.faces = self.mesh.faces

class DownSampledBunny(BunnyMesh):
    def __init__(self) -> None:
        super().__init__()

        self.mesh = self.mesh.simplify_quadratic_decimation(int(self.verts.shape[0]/10))
        self.verts = self.mesh.vertices 
        self.faces = self.mesh.faces

        self.points = np.array(self.verts)

        self._min_pnt = self.points.min(axis=0)
        self._max_pnt = self.points.max(axis=0)
        self.distr = np.ones(self.points.shape[0])
        self.args = {
            'x_i' : self.points,
            'p(x_i)' : self.distr/np.sum(self.distr)
        }

class MeshDistr(object):
    def __init__(self, mesh, func) -> None:
        self.func = func 
        self.mesh = mesh
        self.verts = np.array(mesh.verts)
        self._min_pnt = self.verts.min(axis=0)
        self._max_pnt = self.verts.max(axis=0)
        self.func_vals = jax.vmap(func, in_axes=0)(self.verts)

class BunnyPCL(object):
    def __init__(self) -> None:
        self.dataset = o3d.data.BunnyMesh()
        plydata = PlyData.read(self.dataset.path)
        self.points = np.vstack((
            plydata['vertex']['x'],
            plydata['vertex']['y'],
            plydata['vertex']['z']
        )).T

        self.points = self.points[::10]

        _min_pnt = self.points.min(axis=0)
        _max_pnt = self.points.max(axis=0)
        _mid_pnt = self.points.mean(axis=0)

        self.points = (self.points - _mid_pnt)/(_max_pnt-_min_pnt)
        self._min_pnt = self.points.min(axis=0)
        self._max_pnt = self.points.max(axis=0)
        self.distr = np.ones(self.points.shape[0])
        self.args = {
            'x_i' : self.points,
            'p(x_i)' : self.distr/np.sum(self.distr)
        }

class SphereMesh(object):
    def __init__(self) -> None:
        plydata = PlyData.read('../assets/sphere.ply')
        verts = np.vstack((
            plydata['vertex']['x'],
            plydata['vertex']['y'],
            plydata['vertex']['z']
        )).T

        _min_pnt = verts.min(axis=0)
        _max_pnt = verts.max(axis=0)
        _mid_pnt = verts.mean(axis=0)
        verts = (verts - _mid_pnt)/(_max_pnt-_min_pnt)

        faces = np.array(onp.vstack(plydata['face']['vertex_indices']))

        self.mesh = tm.Trimesh(vertices=verts, faces=faces)
        self.verts = self.mesh.vertices 
        self.faces = self.mesh.faces