import numpy as np
import cupy as cp
import basis as b
import scipy.special as sp
import matplotlib.pyplot as plt

class SpaceGrid:
    def __init__(self, low, high, elements, order):
        # grid limits and elements
        self.low, self.high = low, high
        self.elements = elements
        self.order = order
        # grid has a local Gauss-Legendre quadrature basis
        self.local_basis = b.Basis1D(order=self.order)

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.elements

        # element Jacobian
        self.J = 2.0 / self.dx

        # global (whole grid) quad weights
        self.global_quads = cp.tensordot(cp.ones(elements),
                                         cp.asarray(self.local_basis.weights), axes=0)

        # arrays
        self.arr, self.device_arr = None, None
        self.create_grid()

        # spectral properties
        self.transform_matrix = None
        self.cutoff = self.elements + 1
        self.fundamental = 2.0 * np.pi / self.length
        self.wavenumbers = self.fundamental * np.arange(1 - self.cutoff, self.cutoff)
        self.device_wavenumbers = cp.asarray(self.wavenumbers)
        self.grid_phases = cp.tensordot(self.device_wavenumbers, self.device_arr, axes=0)
        self.grid_modes = cp.exp(1j * self.grid_phases)
        self.build_transform_matrix()

    def create_grid(self):
        """ Build global grid """
        # translate to [0, 1]
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        # element left boundaries (including ghost elements)
        xl = np.linspace(self.low, self.high - self.dx, num=self.elements)
        # construct coordinates
        self.arr = np.zeros((self.elements, self.order))
        for i in range(self.elements):
            self.arr[i, :] = xl[i] + self.dx * np.array(nodes_iso)
        # send to device
        self.device_arr = cp.asarray(self.arr)

    def build_transform_matrix(self):
        """ Build transform matrix, local galerkin -> global fourier """
        self.transform_matrix = cp.asarray(
            np.multiply(self.local_basis.weights[None, None, :],
                        np.exp(-1j * self.grid_phases.get())) / (self.J * self.length)
        )

    def fourier_transform(self, function, idx):
        # print(function.shape)
        # print(self.transform_matrix.shape)
        return cp.tensordot(self.transform_matrix, function, axes=([1, 2], idx))

    def inverse_fourier_transform(self, spectrum, idx):
        return cp.tensordot(spectrum, self.grid_modes, axes=(idx, [0]))


class VelocityGrid:
    def __init__(self, low, high, elements, order):
        # grid limits and elements
        self.low, self.high = low, high
        self.elements = elements
        self.order = order
        # grid has a local Gauss-Legendre quadrature basis
        self.local_basis = b.Basis1D(order=self.order)

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.elements

        # element Jacobian
        self.J = 2.0 / self.dx

        # global (whole grid) quad weights
        self.global_quads = cp.tensordot(cp.ones(elements),
                                         cp.asarray(self.local_basis.weights), axes=0)

        # arrays
        self.arr, self.device_arr = None, None
        self.create_grid()

        # spectral properties
        self.transform_matrix = None
        self.cutoff = 2 * self.elements + 1
        self.modes = np.arange(self.cutoff)
        self.device_modes = cp.asarray(self.modes)
        self.upper_grid_modes = cp.array([upper_hermite(n, self.arr) for n in range(self.cutoff)])
        self.lower_grid_modes = cp.array([lower_hermite(n, self.arr) for n in range(self.cutoff)])
        self.build_transform_matrix()
        # plt.figure()
        # for i in range(self.cutoff):
        #     plt.plot(self.arr.flatten(), self.lower_grid_modes[i, :, :].get().flatten(), 'o')
        # plt.show()

    def create_grid(self):
        """ Build global grid """
        # translate to [0, 1]
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        # element left boundaries (including ghost elements)
        xl = np.linspace(self.low, self.high - self.dx, num=self.elements)
        # construct coordinates
        self.arr = np.zeros((self.elements, self.order))
        for i in range(self.elements):
            self.arr[i, :] = xl[i] + self.dx * np.array(nodes_iso)
        # send to device
        self.device_arr = cp.asarray(self.arr)

    def build_transform_matrix(self):
        """ Build transform matrix, local galerkin -> global fourier """
        self.transform_matrix = cp.asarray(
            np.multiply(self.local_basis.weights[None, None, :],
                        self.upper_grid_modes.get()) / self.J
        )

    def hermite_transform(self, function, idx):
        return cp.tensordot(self.transform_matrix, function, axes=([1, 2], idx))

    def inverse_hermite_transform(self, spectrum, idx):
        return cp.tensordot(spectrum, self.lower_grid_modes, axes=(idx, [0]))


class PhaseSpace:
    def __init__(self, lows, highs, elements, orders):
        self.x = SpaceGrid(low=lows[0], high=highs[0], elements=elements[0], order=orders[0])
        self.v = VelocityGrid(low=lows[1], high=highs[1], elements=elements[1], order=orders[1])

    def fourier_hermite_transform(self, function):
        """ Returns Fourier-Hermite coefficients as (Fourier, Hermite) """
        return self.v.hermite_transform(
            function=self.x.fourier_transform(
                function=function, idx=[0, 1]), idx=[1, 2]).transpose()

    def invert_fourier_hermite_transform(self, spectrum):
        return cp.real(self.x.inverse_fourier_transform(
            spectrum=self.v.inverse_hermite_transform(
                spectrum=spectrum, idx=[1]), idx=[0]).transpose((2, 3, 0, 1)))


def lower_hermite(n, arr):
    return sp.hermite(n, monic=False)(arr) * np.exp(-arr ** 2.0) / np.sqrt((2.0 ** n) * np.pi * np.math.factorial(n))


def upper_hermite(n, arr):
    return sp.hermite(n, monic=False)(arr) / np.sqrt((2.0 ** n) * np.math.factorial(n))
