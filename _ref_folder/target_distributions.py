import jax.numpy as np 
from jax import vmap
import matplotlib.pyplot as plt

class Uniform2D(object):
    def __init__(self) -> None:
        self.n = 2
        self.domain = np.meshgrid(
            *[np.linspace(0,1, num=20)]*self.n
        )
        self._s = np.stack([X.ravel() for X in self.domain]).T
        # self._s = self._s.at[:,2].set(0.2)
        self.evals = (
            vmap(self.p)(self._s) , self._s
        )
        self.args = {
            'x_i' : self._s, 
            'p(x_i)' : self.evals[0]/np.sum(self.evals[0])
        }
    
    def plot(self):
        plt.contour(self.domain[0], self.domain[1], self.evals[0].reshape(self.domain[0].shape))

    def p(self, x):
        return 1.0

    def update(self):
        pass

class TriModal2D(object):
    def __init__(self, num=50) -> None:
        self.n = 2
        self.domain = np.meshgrid(
            *[np.linspace(0,1, num=num)]*self.n
        )
        self._s = np.stack([X.ravel() for X in self.domain]).T
        # self._s = self._s.at[:,2].set(0.2)
        self.evals = (
            vmap(self.p)(self._s) , self._s
        )
        self.args = {
            'x_i' : self._s, 
            'p(x_i)' : self.evals[0]/np.sum(self.evals[0])
        }
    
    def plot(self):
        plt.contour(self.domain[0], self.domain[1], self.evals[0].reshape(self.domain[0].shape))

    def p(self, x):
        return np.exp(-80 * np.sum((x[:2] - 0.2)**2)) \
                + np.exp(-80 * np.sum((x[:2] - 0.75)**2)) \
                    + np.exp(-50.5 * np.sum((x[:2] - np.array([0.23, 0.75]))**2)) \
                    # + np.exp(-60.5 * np.sum((x[:2] - np.array([0.75, 0.2]))**2))
        # return 1
        # return np.exp(-60.5 * np.sum((x[:2] - 0.2)**2)) \
        #             + np.exp(-60.5 * np.sum((x[:2] - 0.75)**2)) \
        #             + np.exp(-60.5 * np.sum((x[:2] - np.array([0.2, 0.75]))**2)) \
        #             + np.exp(-60.5 * np.sum((x[:2] - np.array([0.75, 0.2]))**2))

    def update(self):
        pass