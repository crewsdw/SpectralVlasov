import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt

# grids
v = np.linspace(-5, 5, num=300)
vx = np.tensordot(v, np.ones_like(v), axes=0)
vy = np.tensordot(np.ones_like(v), v, axes=0)
z = vx + 1j * vy

# wavenumber
length = 10.0 * np.pi
k = 2.0 * np.pi / length
x = np.linspace(-length/2.0, length/2.0, num=100)
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
dZ1, dZ2 = -2.0*(1.0 + zeta1 * Z1), -2.0*(1.0 + zeta2 * Z2)
dZ = 0.5 * (dZ1 + dZ2)

# Dispersion function
D = 1.0 - dZ / (k ** 2.0)  # * np.sqrt(np.pi)


def dispersion_function_fsolve(variable):
    complex = variable[0] + 1j*variable[1]
    shifted1 = (complex - u1) / vt
    shifted2 = (complex - u2) / vt
    pdf1, pdf2 = pd_function(shifted1), pd_function(shifted2)
    d_pdf1, d_pdf2 = -2.0*(1.0 + shifted1 * pdf1), -2.0*(1.0 + shifted2 * pdf2)
    d_pdf = 0.5 * (d_pdf1 + d_pdf2)
    D_func = 1.0 - d_pdf / (k ** 2.0)
    return [np.real(D_func), np.imag(D_func)]


solution = opt.fsolve(dispersion_function_fsolve, x0=[0.0, 1.2])
print(solution)
print('The growth rate is {:0.9e}'.format(k * solution[1]))

# distribution gradient
def grad_max(u):
    return -2.0 * (v - u) / (vt ** 2.0) * np.exp(-((v - u) / vt) ** 2.0) / (vt * np.sqrt(np.pi))


df1, df2 = grad_max(u1), grad_max(u2)
df = 0.5 * (df1 + df2)


def eigenfunction(eig_here):
    """ Compute eigenfunction with z as the eigenvalue """
    v_part = np.divide(df, (v - eig_here))
    return np.real(np.tensordot(np.exp(1j * k * x), v_part, axes=0))


eigenvalue = 1.44 - 0.6j  # 3.0j
# mode = eigenfunction(eigenvalue) + eigenfunction(-1.44 - 0.6j)
mode = eigenfunction(1.2j)  # + eigenfunction(-1.44 - 0.6j)
cb = np.linspace(np.amin(mode), np.amax(mode), num=100)

plt.figure()
plt.contour(vx, vy, np.real(D), 0, colors='r')
plt.contour(vx, vy, np.imag(D), 0, colors='g')
plt.grid(True), plt.tight_layout()

X, V = np.meshgrid(x, v, indexing='ij')

# plt.figure()
# plt.contourf(X, V, df, cb)

plt.figure()
plt.contourf(X, V, mode, cb)
plt.show()

