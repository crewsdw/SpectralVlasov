import cupy as cp
import plotter as my_plt
import matplotlib.pyplot as plt


class Spectral:
    def __init__(self):
        self.rhs = None

    def spectral_rhs(self, distribution, grid, elliptic):
        """ Computes the spectral right-hand side of Vlasov equation """
        # Compute linear translation term
        distribution.pad_spectrum()  # add zeros for -1, N+1 modes so it's not periodic in modes
        x_translation = -1j * cp.multiply(grid.x.device_wavenumbers[:, None],
                                          distribution.hermite_translate(grid=grid))
        # Compute nonlinear term using pseudo-spectral method, E * df/dv
        df_dv = grid.invert_fourier_hermite_transform(
            spectrum=distribution.hermite_derivative(grid=grid)
        )
        elliptic.compute_field(grid=grid)
        # Transform back to fourier-hermite modes
        edfdv = cp.multiply(elliptic.field.arr_nodal[:, :, None, None], df_dv)
        v_translation = grid.fourier_hermite_transform(
            function=edfdv
        )
        # plotter = my_plt.Plotter(grid=grid)
        # plt.figure()
        # plt.plot(grid.x.arr.flatten(), elliptic.field.arr_nodal.get().flatten(), 'o')
        # plt.tight_layout()
        #
        # plt.figure()
        # plt.contourf(plotter.X, plotter.V, edfdv.get().reshape((grid.x.elements * grid.x.order,
        #                                                         grid.v.elements * grid.v.order)))
        # plt.colorbar(), plt.tight_layout()
        # plt.show()

        self.rhs = x_translation + v_translation
