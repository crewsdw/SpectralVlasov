import numpy as np
import cupy as cp


class Scalar1D:
    def __init__(self, resolution, order):
        self.res = resolution
        self.order = order
        self.arr_nodal, self.arr_spectral = None, None

    def grid_flatten(self):
        return self.arr_nodal.reshape((self.res * self.order))

    def fourier_transform(self, grid):
        self.arr_spectral = grid.x.fourier_transform(function=self.arr_nodal, idx=[0, 1])

    def inverse_fourier_transform(self, grid):
        self.arr_nodal = cp.real(
            grid.x.inverse_fourier_transform(spectrum=self.arr_spectral, idx=[0]))


class PhaseSpaceScalar:
    def __init__(self, resolutions, orders):
        self.x_res, self.v_res = resolutions
        self.x_ord, self.v_ord = orders

        # array
        self.arr_nodal, self.arr_spectral = None, None
        self.padded_spectrum = None
        self.zero_moment = Scalar1D(resolution=resolutions[0], order=orders[0])

    def initialize(self, grid):
        ix, iv = cp.ones_like(grid.x.device_arr), cp.ones_like(grid.v.device_arr)
        pert = 1.0 + 0.1 * cp.sin(grid.x.fundamental * grid.x.device_arr)
        factor = 1.0 / (np.sqrt(np.pi)) * cp.tensordot(pert, iv, axes=0)
        vsq1 = cp.tensordot(ix, cp.power((grid.v.device_arr-2.0), 2.0), axes=0)
        vsq2 = cp.tensordot(ix, cp.power((grid.v.device_arr+2.0), 2.0), axes=0)
        gauss = 0.5 * cp.exp(-vsq1) + 0.5 * cp.exp(-vsq2)
        self.arr_nodal = cp.multiply(factor, gauss)  # + perturbation

    def zero_moment_spectral(self, grid):
        """ Fourier modes of zero-moment are the zero Hermite modes """
        if self.arr_spectral is None:
            self.fourier_hermite_transform(grid=grid)
        self.zero_moment.arr_spectral = self.arr_spectral[:, 0]

    def fourier_hermite_transform(self, grid):
        """
        Compute Fourier modes in space and Hermite modes in velocity
        """
        self.arr_spectral = grid.fourier_hermite_transform(function=self.arr_nodal)

    def invert_fourier_hermite_transform(self, grid):
        """ Inver the Fourier-Hermite transform """
        self.arr_nodal = cp.real(grid.invert_fourier_hermite_transform(spectrum=self.arr_spectral))

    def hermite_translate(self, grid):
        """ Computes the term v*H via Hermite recurrence """
        return (
                cp.multiply(cp.sqrt(0.5 * (grid.v.device_modes + 1))[None, :],
                            cp.roll(self.padded_spectrum, shift=-1, axis=1)[:, 1:-1]) +
                cp.multiply(cp.sqrt(0.5 * grid.v.device_modes)[None, :],
                            cp.roll(self.padded_spectrum, shift=+1, axis=1)[:, 1:-1])
        )

    def hermite_derivative(self, grid):
        """ Computes the term df/dv using Hermite recurrence """
        return cp.multiply(cp.sqrt(2.0 * grid.v.device_modes)[None, :],
                           cp.roll(self.padded_spectrum, shift=+1, axis=1)[:, 1:-1])
        # return cp.multiply(cp.sqrt(2.0 * (grid.v.device_modes+1))[None, :],
        #                    cp.roll(self.padded_spectrum, shift=-1, axis=1)[:, 1:-1])

    def grid_flatten(self):
        return self.arr_nodal.reshape((self.x_res * self.x_ord,
                                       self.v_res * self.v_ord))

    def pad_spectrum(self):
        self.padded_spectrum = cp.zeros((self.arr_spectral.shape[0], self.arr_spectral.shape[1] + 2)) + 0j
        self.padded_spectrum[:, 1:-1] = self.arr_spectral
        # reflect
        # self.padded_spectrum[:, 0] = self.arr_spectral[:, 0]
