import cupy as cp


class Spectral:
    def __init__(self):
        self.rhs = None

    def spectral_rhs(self, distribution, grid, elliptic):
        """ Computes the spectral right-hand side of Vlasov equation """
        # for now: separate terms, can update later
        distribution.pad_spectrum()
        term1 = -1j * cp.multiply(grid.x.device_wavenumbers[:, None], distribution.hermite_translate(grid=grid))
        term2 = 1j * cp.multiply(cp.multiply(elliptic.potential.arr_spectral,
                                             grid.x.device_wavenumbers)[:, None],
                                 distribution.hermite_derivative(grid=grid))
        # print(cp.amax(cp.absolute(term1)))
        # print(cp.amax(cp.absolute(term2)))
        self.rhs = term1 + term2
