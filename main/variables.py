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
        self.first_moment = Scalar1D(resolution=resolutions[0], order=orders[0])
        self.centered_second_moment = Scalar1D(resolution=resolutions[0], order=orders[0])

    def initialize(self, grid):
        a = 1.0  # / np.sqrt(2.0)
        ix, iv = cp.ones_like(grid.x.device_arr), cp.ones_like(grid.v.device_arr)
        pert = 1.0 + 0.1 * cp.sin(grid.x.fundamental * grid.x.device_arr)
        factor = 1.0 / (np.sqrt(a ** 2.0 * np.pi)) * cp.tensordot(pert, iv, axes=0)
        vsq1 = cp.tensordot(ix, cp.power((grid.v.device_arr - 2.0), 2.0), axes=0)
        vsq2 = cp.tensordot(ix, cp.power((grid.v.device_arr + 2.0), 2.0), axes=0)
        gauss = 0.5 * cp.exp(-vsq1 / a ** 2.0) + 0.5 * cp.exp(- vsq2 / a ** 2.0)
        self.arr_nodal = cp.multiply(factor, gauss)  # + perturbation

    def zero_moment_spectral(self, grid):
        """ Fourier modes of zero-moment are the zero Hermite modes """
        if self.arr_spectral is None:
            self.fourier_hermite_transform(grid=grid)
        self.zero_moment.arr_spectral = self.arr_spectral[:, 0] * cp.array(grid.v.alpha)

    def fourier_hermite_transform(self, grid):
        """
        Compute Fourier modes in space and Hermite modes in velocity
        """
        self.arr_spectral = grid.fourier_hermite_transform(function=self.arr_nodal)

    def invert_fourier_hermite_transform(self, grid):
        """ Invert the Fourier-Hermite transform """
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

    def spectral_lenard_bernstein(self, grid):
        nu = 7.5e-1
        return -1.0 * nu * (
                cp.multiply((grid.v.device_modes * (grid.v.device_modes - 1) * (grid.v.device_modes - 2))[None, :],
                            self.arr_spectral) /
                ((grid.v.cutoff - 1) * (grid.v.cutoff - 2) * (grid.v.cutoff - 3))
        )

    def grid_flatten(self):
        return self.arr_nodal.reshape((self.x_res * self.x_ord,
                                       self.v_res * self.v_ord))

    def pad_spectrum(self):
        """ Pad spectrum with zeros on either end, to prevent spectral periodicity from cp.roll() """
        self.padded_spectrum = cp.zeros((self.arr_spectral.shape[0], self.arr_spectral.shape[1] + 2)) + 0j
        self.padded_spectrum[:, 1:-1] = self.arr_spectral

    def compute_zero_moment(self, grid):
        self.zero_moment_spectral(grid=grid)
        self.zero_moment.inverse_fourier_transform(grid=grid)  # grid.v.zero_moment(function=self.arr_nodal, idx=[2, 3])

    def compute_first_moment(self, grid):
        self.first_moment.arr_nodal = cp.divide(grid.v.first_moment(function=self.arr_nodal, idx=[2, 3]),
                                                self.zero_moment.arr_nodal)

    def compute_centered_second_moment(self, grid):
        # self.centered_second_moment.arr_nodal = cp.sqrt(
        #     cp.divide(grid.v.shifted_second_moment(function=self.arr_nodal,
        #                                            shift=0.0 * cp.mean(self.first_moment.arr_nodal),
        #                                            idx=[2, 3]),
        #               self.zero_moment.arr_nodal)
        # )
        self.centered_second_moment.arr_nodal = cp.sqrt(
            grid.v.shifted_second_moment(function=cp.mean(self.arr_nodal.reshape(self.x_res * self.x_ord,
                                                                                 self.v_res, self.v_ord), axis=0),
                                         shift=0.0,  # * cp.mean(self.first_moment.arr_nodal),
                                         idx=[0, 1])
        )

    def recompute_hermite_basis(self, grid):
        """ Compute the hermite basis again based on first and second moments """
        if self.arr_spectral is not None:
            self.invert_fourier_hermite_transform(grid=grid)
        self.compute_zero_moment(grid=grid)
        # self.compute_first_moment(grid=grid)
        self.compute_centered_second_moment(grid=grid)
        # print(cp.mean(self.zero_moment.arr_nodal))
        # print(cp.mean(self.first_moment.arr_nodal))
        # print(cp.mean(self.centered_second_moment.arr_nodal))
        # print(cp.mean(cp.sqrt(2) * self.centered_second_moment.arr_nodal).get())
        grid.v.compute_hermite_basis(first_moment=0,  # * cp.mean(self.first_moment.arr_nodal).get(),
                                     centered_second_moment=cp.mean(self.centered_second_moment.arr_nodal).get())
