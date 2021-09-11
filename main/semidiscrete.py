import cupy as cp


class Spectral:
    def __init__(self):
        self.rhs = None

    def spectral_rhs(self, distribution, grid, elliptic):
        """ Computes the spectral right-hand side of Vlasov equation """
        # for now: separate terms, can update later
        term1 = -1j * cp.multiply(grid.x.device_wavenumbers[:, None], distribution.hermite_translate())
        term2 = -1j * cp.multiply((elliptic.potential_spectrum * grid.x.device_wavenumbers)[:, None],
                                  distribution.hermite_derivative())

        self.rhs = term1 + term2
