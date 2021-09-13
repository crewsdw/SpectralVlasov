import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Plotter:
    def __init__(self, grid, colormap='RdPu'):
        self.colormap = colormap
        self.grid = grid
        # Build structured grid, nodal
        self.X, self.V = np.meshgrid(grid.x.arr.flatten(), grid.v.arr.flatten(), indexing='ij')
        self.x = grid.x.arr
        # Build structured grid, global spectral
        self.F, self.H = np.meshgrid(grid.x.wavenumbers / grid.x.fundamental, grid.v.modes,
                                     indexing='ij')

    def phasespace_scalar_contourf(self, ps_scalar):
        cb = cp.linspace(cp.amin(ps_scalar.arr_nodal), cp.amax(ps_scalar.arr_nodal), num=100).get()

        plt.figure()
        plt.contourf(self.X, self.V, ps_scalar.grid_flatten().get(),
                     cb, cmap=self.colormap)
        plt.xlabel('x'), plt.ylabel('v')
        plt.colorbar(), plt.tight_layout()

    def spatial_scalar_plot(self, scalar, y_axis):
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

    def plot_fourier_hermite_spectrum(self, ps_scalar):
        real_spectrum, imag_spectrum = np.real(ps_scalar.arr_spectral.get()), \
                                       np.imag(ps_scalar.arr_spectral.get())

        cb_r = cp.linspace(cp.amin(real_spectrum), cp.amax(real_spectrum), num=100).get()
        cb_i = cp.linspace(cp.amin(imag_spectrum), cp.amax(imag_spectrum), num=100).get()

        fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        c1 = ax[0].contourf(self.F, self.H, real_spectrum, cb_r)
        c2 = ax[1].contourf(self.F, self.H, imag_spectrum, cb_i)
        ax[0].set_title('Real spectrum')
        ax[1].set_title('Imaginary spectrum')
        plt.colorbar(c1, ax=ax[0]), plt.colorbar(c2, ax=ax[1])
        plt.setp(ax[0], xlabel='Fourier modes'), plt.setp(ax[1], xlabel='Fourier modes')
        plt.setp(ax[0], ylabel='Hermite modes')
        plt.tight_layout()

    def animate_phase_space_scalar(self, stepper):
        fig, ax = plt.subplots()  # subplots(1, 2, figsize=(16, 8), constrained_layout=True)
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')

        def animate_frame(idx):
            ax.collections, ax.patches = [], []
            # plot phase space scalar
            cb = cp.linspace(cp.amin(stepper.saved_array[idx]), cp.amax(stepper.saved_array[idx]), num=100).get()
            cf = ax.contourf(self.X, self.V, stepper.saved_array[idx].get().reshape(self.X.shape),
                             cb, cmap=self.colormap)
            cax.cla()
            fig.colorbar(cf, cax=cax)
            fig.suptitle('Time t={:0.3f}'.format(stepper.saved_times[idx]))

        anim = animation.FuncAnimation(fig, animate_frame, frames=len(stepper.saved_array))
        anim.save(filename='..\\movies\\animation.mp4')

    def show_all(self):
        plt.show()
