import jax 
import jax.numpy as np 
import numpy as onp
from jax.flatten_util import ravel_pytree
import jaxopt

from .kernel import create_kernel_matrix

class ErgodicMMDPlanner(object):
    def __init__(self, mesh, kernel) -> None:
        KernelMatrix = create_kernel_matrix(kernel)

        x0 =  mesh._min_pnt
        xf =  mesh._max_pnt
        T = 80
        self.X_init = np.linspace(x0, xf, num=T, endpoint=True)
        self.sol, self.unflatten_X = ravel_pytree(self.X_init)
        self.bounds = (x0.min() * np.ones_like(self.sol), xf.max() * np.ones_like(self.sol))

        def ergodic_mmd(flat_X, args):
            X_init = self.unflatten_X(flat_X)
            T = X_init.shape[0]
            h = args['h']
            # x0 = args['x0']
            X_samples = args['X_samples']
            P_XI      = args['P_XI']
            return np.sum(KernelMatrix(X_init, X_init, h))/(T**2) \
                    - 2 * np.sum(P_XI @ KernelMatrix(X_init, X_samples, h))/T \
                    + np.mean((X_init[1:]-X_init[:-1])**2)
                    # + np.sum((X_init[0]-x0)**2)
        
        self.solver = jaxopt.ProjectedGradient(fun=ergodic_mmd, projection=jaxopt.projection.projection_box, tol=1e-12)
        self.solver_state = self.solver.init_state(self.sol, hyperparams_proj=self.bounds)
    def plan(self, args, max_iter=100):
        # self.sol = self.solver.run(init_params=self.sol, 
        #                         hyperparams_proj=self.bounds,  
        #                         args=args).params
        for _ in range(max_iter):
            (self.sol, self.solver_state) = self.solver.update(
                                params=self.sol, 
                                state=self.solver_state, 
                                hyperparams_proj=self.bounds, 
                                args=args)
        return self.unflatten_X(self.sol)