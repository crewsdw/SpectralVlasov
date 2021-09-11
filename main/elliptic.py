import cupy as cp
import variables as var


class Elliptic:
    def __init__(self, elements, orders):
        # init potential
        self.potential = var.Scalar1D(resolution=elements[0], order=orders[0])
        self.field = var.Scalar1D(resolution=elements[0], order=orders[0])
        # self.field_spectrum = None

    def poisson_solve(self, distribution, grid, invert=True):
        """ Solve Poisson equation for electric potential """
        # Compute zeroth moment
        distribution.zero_moment_spectral(grid=grid)

        self.potential.arr_spectral = cp.nan_to_num(
            cp.divide(distribution.zero_moment.arr_spectral,
                      grid.x.device_wavenumbers ** 2.0)
        )

        if invert:
            self.potential.inverse_fourier_transform(grid=grid)

    def compute_field(self, grid):
        # Compute electric field
        self.field.arr_nodal = cp.real(grid.x.inverse_fourier_transform(
            spectrum=cp.multiply(-1j * grid.x.device_wavenumbers, self.potential.arr_spectral),
            idx=[0]
        ))
        # print(self.field.arr_nodal.shape)
        # quit()
