import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as animation


class Plotter:
    def __init__(self, grid, colormap='RdPu'):
        self.colormap = colormap
        self.grid = grid
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

    def spatial_scalar_plot(self, scalar, y_axis):
        # cb = cp.linspace(cp.amin(scalar.arr_nodal), cp.amax(scalar.arr_nodal), num=100).get()
        if scalar.arr_nodal is None:
            scalar.inverse_fourier_transform(grid=self.grid)

        plt.figure()
        plt.plot(self.x.flatten(), scalar.arr_nodal.get().flatten(), 'o')
        plt.xlabel('x'), plt.ylabel(y_axis)
        plt.tight_layout()

    def plot_saved_scalars(self, saved_array):
        plt.figure()
        for i in range(len(saved_array)):
            plt.plot(self.x.flatten(), saved_array[i].get().flatten(), 'o')
        plt.xlabel('x'), plt.ylabel('n')
        plt.title('Plot of saved scalar arrays')
        plt.tight_layout()

    def show_all(self):
        plt.show()
