import jax.numpy as np
from jax import vmap

from plyfile import PlyData
import open3d as o3d


class BunnyMesh(object):
    def __init__(self) -> None:
        self.dataset = o3d.data.BunnyMesh()
        plydata = PlyData.read(self.dataset.path)
        self.points = np.vstack((
            plydata['vertex']['x'],
            plydata['vertex']['y'],
            plydata['vertex']['z']
        )).T

        _min_pnt = self.points.min(axis=0)
        _max_pnt = self.points.max(axis=0)

        self.points = (self.points - _min_pnt)/(_max_pnt-_min_pnt)
        self.distr = np.ones(self.points.shape[0])
        self.args = {
            'x_i' : self.points,
            'p(x_i)' : self.distr/np.sum(self.distr)
        }

class SphereMesh(object):
    def __init__(self) -> None:
        plydata = PlyData.read('./sphere.ply')
        self.points = np.vstack((
            plydata['vertex']['x'],
            plydata['vertex']['y'],
            plydata['vertex']['z']
        )).T

        _min_pnt = self.points.min(axis=0)
        _max_pnt = self.points.max(axis=0)

        self.points = (self.points - _min_pnt)/(_max_pnt-_min_pnt)
        self.distr = np.ones(self.points.shape[0])
        self.args = {
            'x_i' : self.points,
            'p(x_i)' : self.distr/np.sum(self.distr)
        }
