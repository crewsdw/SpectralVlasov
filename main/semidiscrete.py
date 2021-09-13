import cupy as cp
import plotter as my_plt
import matplotlib.pyplot as plt


class Spectral:
    def __init__(self, alpha):
        self.rhs = None
        self.alpha = alpha

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
        v_translation = grid.fourier_hermite_transform(
            function=cp.multiply(elliptic.field.arr_nodal[:, :, None, None], df_dv)
        )
        #
        # plotter = my_plt.Plotter(grid=grid)
        # plt.figure()
        # cb = cp.linspace(cp.amin(df_dv), cp.amax(df_dv), num=100).get()
        # plt.contourf(plotter.X, plotter.V, df_dv.get().reshape(plotter.X.shape[0], plotter.V.shape[1]), cb)
        # plt.colorbar(), plt.tight_layout
        # plt.show(block=False)
        # plt.pause(0.5)
        # plt.close()
        # Collision, Lenard-Bernstein
        collision = distribution.spectral_lenard_bernstein(grid=grid)

        self.rhs = self.alpha * x_translation + v_translation / self.alpha + collision
