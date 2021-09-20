import cupy as cp
import cupyx.scipy.linalg as clg
import plotter as my_plt
import matplotlib.pyplot as plt


class ImplicitSpectral:
    def __init__(self, alpha):
        self.rhs = None
        self.alpha = alpha

    def fourier_hermite_jacobian(self, variable, grid):
        """ Computes the Jacobian of the spectral ODE """
        # convolutional_term = cp.zeros_like(variable.arr_spectral)
        convolutional_term = cp.array([clg.convolution_matrix(variable.arr_spectral[:, m-1],
                                                              n=variable.arr_spectral.shape[0],
                                                              mode='same')])


class Spectral:
    def __init__(self, alpha):
        self.rhs = None
        self.alpha = alpha

    def spectral_rhs(self, distribution, grid, elliptic):
        """ Computes the spectral right-hand side of Vlasov equation """
        self.alpha = cp.array(grid.v.alpha)
        # Compute linear translation term
        distribution.pad_spectrum()  # add zeros for -1, N+1 modes so it's not periodic in modes
        x_translation = -1j * cp.multiply(grid.x.device_wavenumbers[:, None],
                                          distribution.hermite_translate(grid=grid))
        # Compute nonlinear term using pseudo-spectral method, E * df/dv
        # df_dv = grid.invert_fourier_hermite_transform(
        #     spectrum=distribution.hermite_derivative(grid=grid)
        # )
        # elliptic.compute_field(grid=grid, nodal=True)
        # # Transform back to fourier-hermite modes
        # v_translation = grid.fourier_hermite_transform(
        #     function=cp.multiply(elliptic.field.arr_nodal[:, :, None, None], df_dv)
        # )

        # Compute nonlinear term using fully-spectral method
        df_dv_hermite = distribution.hermite_derivative(grid=grid)
        elliptic.compute_field(grid=grid, nodal=False)
        # Convolve Fourier modes
        v_translation = fourier_convolution(df_dv_hermite, elliptic.field.arr_spectral)
        # v_trans = grid.invert_fourier_hermite_transform(
        #     spectrum=v_translation
        # )
        # v_trans = grid.invert_fourier_hermite_transform(
        #     spectrum=df_dv_hermite
        # )
        #
        # plotter = my_plt.Plotter(grid=grid)
        # plt.figure()
        # cb = cp.linspace(cp.amin(v_trans), cp.amax(v_trans), num=100).get()
        # plt.contourf(plotter.X, plotter.V, v_trans.get().reshape(plotter.X.shape[0], plotter.V.shape[1]), cb)
        # plt.colorbar(), plt.tight_layout
        # plt.show(block=True)
        # plt.pause(0.5)
        # plt.close()
        # Collision, Lenard-Bernstein
        collision = distribution.spectral_lenard_bernstein(grid=grid)

        self.rhs = self.alpha * x_translation + v_translation / self.alpha + collision / self.alpha
    #
    # def fourier_convolution(self, spectrum):
    #

    def nodal_rhs(self, distribution, grid, elliptic):
        """ Computes the real-space right-hand side of Vlasov equation """
        # Compute linear translation term spectrally
        distribution.fourier_hermite_transform(grid=grid)
        distribution.pad_spectrum()  # add zeros for -1, N+1 modes so it's not periodic in modes
        x_translation = grid.invert_fourier_hermite_transform(
            spectrum=-1j * cp.multiply(grid.x.device_wavenumbers[:, None],
                                       distribution.hermite_translate(grid=grid))
        )
        # Compute nonlinear term E * df/dv on grid with each term spectrally
        df_dv = grid.invert_fourier_hermite_transform(
            spectrum=distribution.hermite_derivative(grid=grid)
        )
        elliptic.compute_field(grid=grid)
        v_translation = cp.multiply(elliptic.field.arr_nodal[:, :, None, None], df_dv)
        # Collision term (not yet)
        self.rhs = grid.v.alpha * x_translation + v_translation / grid.v.alpha


def fourier_convolution(ps_spectrum, x_spectrum):
    # works but slow
    return cp.asarray([cp.convolve(x_spectrum, ps_spectrum[:, n], mode='same')
                    for n in range(ps_spectrum.shape[1])]).transpose()
    # out = cp.zeros_like(ps_spectrum)
    # for i in range(ps_spectrum.shape[1]):
    #     out[:, i] = cp_sig.convolve(x_spectrum, ps_spectrum[:, i], mode='same')
    # # print(out.shape)
    # return out
    # x_spectrum = cp.fft.fftshift(x_spectrum, axes=0)
    # x_spectrum = cp.roll(x_spectrum, axis=0, shift=-x_spectrum.shape[0]//2+1)
    # ps_spectrum = cp.roll(ps_spectrum, axis=0, shift=-x_spectrum.shape[0]//2+1)
    # ps_spectrum = cp.fft.fftshift(ps_spectrum, axes=0)
    # plt.figure()
    # plt.plot(cp.real(x_spectrum).get(), 'o')
    # plt.plot(cp.imag(x_spectrum).get(), 'o')
    # plt.show()
    # inv1, inv2 = cp.fft.ifft(x_spectrum, axis=0), cp.fft.ifft(ps_spectrum, axis=0)
    # return cp.fft.fft(cp.multiply(inv1[:, None], inv2), axis=0)
