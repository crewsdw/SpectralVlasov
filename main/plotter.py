import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as animation


class Plotter:
    def __init__(self, grid, colormap='RdPu'):
        self.colormap = colormap
        # order = grid.x.order
        # Build structured grid
        self.X, self.V = np.meshgrid(grid.x.arr.flatten(), grid.v.arr.flatten(), indexing='ij')
        self.x = grid.x.arr

    def phasespace_scalar_contourf(self, ps_scalar):
        cb = cp.linspace(cp.amin(ps_scalar.arr_nodal), cp.amax(ps_scalar.arr_nodal), num=100).get()

        plt.figure()
        plt.contourf(self.X, self.V, ps_scalar.grid_flatten().get(),
                     cb, cmap=self.colormap)
        plt.xlabel('x'), plt.ylabel('v')
        plt.colorbar(), plt.tight_layout()

    def spatial_scalar_plot(self, scalar):
        # cb = cp.linspace(cp.amin(scalar.arr_nodal), cp.amax(scalar.arr_nodal), num=100).get()

        plt.figure()
        plt.plot(self.x, scalar.arr_nodal.get(), 'o')
        plt.xlabel('x'), plt.ylabel('scalar')
        plt.tight_layout()

    def show_all(self):
        plt.show()
