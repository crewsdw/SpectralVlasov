import numpy as np
import scipy.special as sp

# Gauss-Legendre nodes and weights
gl_nodes = {
    1: [0],
    2: [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
    3: [-np.sqrt(5 / 8), 0, np.sqrt(5 / 8)],
    4: [-0.861136311594052575224, -0.3399810435848562648027, 0.3399810435848562648027, 0.861136311594052575224],
    5: [-0.9061798459386639927976, -0.5384693101056830910363, 0,
        0.5384693101056830910363, 0.9061798459386639927976],
    6: [-0.9324695142031520278123, -0.661209386466264513661, -0.2386191860831969086305,
        0.238619186083196908631, 0.661209386466264513661, 0.9324695142031520278123],
    7: [-0.9491079123427585245262, -0.7415311855993944398639, -0.4058451513773971669066,
        0, 0.4058451513773971669066, 0.7415311855993944398639, 0.9491079123427585245262],
    8: [-0.9602898564975362316836, -0.7966664774136267395916, -0.5255324099163289858177, -0.1834346424956498049395,
        0.1834346424956498049395, 0.5255324099163289858177, 0.7966664774136267395916, 0.9602898564975362316836],
    9: [-0.9681602395076260898356, -0.8360311073266357942994, -0.6133714327005903973087,
        -0.3242534234038089290385, 0, 0.3242534234038089290385,
        0.6133714327005903973087, 0.8360311073266357942994, 0.9681602395076260898356],
    10: [-0.973906528517171720078, -0.8650633666889845107321, -0.6794095682990244062343,
         -0.4333953941292471907993, -0.1488743389816312108848, 0.1488743389816312108848, 0.4333953941292471907993,
         0.6794095682990244062343, 0.8650633666889845107321, 0.973906528517171720078]
}

gl_weights = {
    1: [2],
    2: [1, 1],
    3: [5 / 9, 8 / 9, 5 / 9],
    4: [0.3478548451374538573731, 0.6521451548625461426269, 0.6521451548625461426269, 0.3478548451374538573731],
    5: [0.2369268850561890875143, 0.4786286704993664680413, 0.5688888888888888888889,
        0.4786286704993664680413, 0.2369268850561890875143],
    6: [0.1713244923791703450403, 0.3607615730481386075698, 0.4679139345726910473899,
        0.46791393457269104739, 0.3607615730481386075698, 0.1713244923791703450403],
    7: [0.1294849661688696932706, 0.2797053914892766679015, 0.38183005050511894495,
        0.417959183673469387755, 0.38183005050511894495, 0.279705391489276667901, 0.129484966168869693271],
    8: [0.1012285362903762591525, 0.2223810344533744705444, 0.313706645877887287338, 0.3626837833783619829652,
        0.3626837833783619829652, 0.313706645877887287338, 0.222381034453374470544, 0.1012285362903762591525],
    9: [0.0812743883615744119719, 0.1806481606948574040585, 0.2606106964029354623187,
        0.312347077040002840069, 0.330239355001259763165, 0.312347077040002840069,
        0.260610696402935462319, 0.1806481606948574040585, 0.081274388361574411972],
    10: [0.0666713443086881375936, 0.149451349150580593146, 0.219086362515982043996,
         0.2692667193099963550912, 0.2955242247147528701739, 0.295524224714752870174, 0.269266719309996355091,
         0.2190863625159820439955, 0.1494513491505805931458, 0.0666713443086881375936]
}


class Basis1D:
    """
    Class containing basis-related methods
    Contains local basis properties
    """
    def __init__(self, order):
        # parameters
        self.order = int(order)
        self.nodes = np.array(gl_nodes.get(self.order, "nothing"))
        self.weights = np.array(gl_weights.get(self.order, "nothing"))

        # vandermonde matrix and inverse
        self.eigenvalues = self.set_eigenvalues()
        self.vandermonde = self.set_vandermonde()
        self.inv_vandermonde = self.set_inv_vandermonde()

    def set_eigenvalues(self):
        return np.array([(2.0 * s + 1) / 2.0 for s in range(self.order)])

    def set_vandermonde(self):
        return np.array([[sp.legendre(s)(self.nodes[j])
                          for j in range(self.order)]
                         for s in range(self.order)])

    def set_inv_vandermonde(self):
        return np.array([[self.weights[j] * self.eigenvalues[s] * sp.legendre(s)(self.nodes[j])
                          for j in range(self.order)]
                         for s in range(self.order)])

    # def interpolate_values(self, grid, arr):
    #     """ Determine interpolated values on any grid ("arr_fine") using the polynomial basis"""
    #     # Compute affine transformation per-element to isoparametric element
    #     xi = grid.J * (grid.arr_fine[1:-1, :] - grid.midpoints[:, None])
    #     # Legendre polynomials at transformed points
    #     ps = np.array([sp.legendre(s)(xi) for s in range(self.order)])
    #     # Interpolation polynomials at fine points
    #     ell = np.transpose(np.tensordot(self.inv_vandermonde, ps, axes=([0], [0])), [1, 0, 2])
    #     # Compute interpolated values
    #     return np.multiply(ell, arr[:, :, None]).sum(axis=1)

    # def fourier_quad(self, grid, wavenumbers, J):
    #     """
    #     Build connection coefficients, local Legendre -> global Fourier
    #     """
    #     # transform of interpolant
    #     # interpolant_transform = np.multiply(np.array(self.weights)[:, None],
    #                                         np.exp(-1j * np.tensordot(self.nodes, wavenumbers, axes=0) / J))
    #
