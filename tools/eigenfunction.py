import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

# grids
v = np.linspace(-5, 5, num=300)
vx = np.tensordot(v, np.ones_like(v), axes=0)
vy = np.tensordot(np.ones_like(v), v, axes=0)
z = vx + 1j * vy

# wavenumber
length = 4.0 * np.pi
k = 2.0 * np.pi / length
x = np.linspace(0.0, length, num=100)
# k = np.linspace(1.0e-6, 1, num=100)


def pd_function(zeta):
    """ The plasma dispersion function """
    return 1j * np.sqrt(np.pi) * np.exp(-zeta ** 2.0) * (1.0 + sp.erf(1j * zeta))


# two-stream variables
u1, u2 = -2.0, 2.0
vt = 1.0
zeta1 = (z - u1) / vt
zeta2 = (z - u2) / vt
# Shifted plasma dispersion functions and derivatives
Z1, Z2 = pd_function(zeta1), pd_function(zeta2)
dZ1, dZ2 = 1.0 + zeta1 * Z1, 1.0 + zeta2 * Z2
dZ = 0.5 * (dZ1 + dZ2)

# Dispersion function
D = 1.0 + dZ / (k ** 2.0)


# distribution gradient
def grad_max(u):
    return -(v - u) / (0.5 * (vt ** 2.0)) * np.exp(-((v - u) / vt) ** 2.0) / (vt * np.sqrt(np.pi))


df1, df2 = grad_max(u1), grad_max(u2)
df = 0.5 * (df1 + df2)


def eigenfunction(z):
    """ Compute eigenfunction with z as the eigenvalue """
    v_part = np.divide(df, (v - z))
    return np.real(1j * np.tensordot(np.exp(1j * k * x), v_part, axes=0))


eigenvalue = 3.0j
mode = eigenfunction(eigenvalue)
cb = np.linspace(np.amin(mode), np.amax(mode), num=100)

plt.figure()
plt.contour(vx, vy, np.real(D), 0)
plt.contour(vx, vy, np.imag(D), 0)

X, V = np.meshgrid(x, v, indexing='ij')

plt.figure()
plt.contourf(X, V, mode, cb)
plt.show()

