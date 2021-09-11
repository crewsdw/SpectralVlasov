import numpy as np
# import cupy as cp
# import basis as b
import grid as g
import variables as var
import semidiscrete as sd
import elliptic as ell
import timestep as ts
import plotter as my_plt

import matplotlib.pyplot as plt

# elements and order
elements, orders = [16, 16], [10, 10]
final_time, write_time = 6.0e0, 1.0e-1

# Set up phase space grid
lows = np.array([-np.pi, -6.0])
highs = np.array([np.pi, 6.0])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, orders=orders)

# Build distribution, elliptic, etc.
distribution = var.PhaseSpaceScalar(resolutions=elements, orders=orders)
distribution.initialize(grid=grid)
distribution.fourier_hermite_transform(grid=grid)
distribution.invert_fourier_hermite_transform(grid=grid)
distribution.zero_moment_spectral(grid=grid)

# Look at Fourier-Hermite spectrum
plt.figure()
for i in range(distribution.arr_spectral.shape[0]):
    plt.plot(grid.v.modes, np.imag(distribution.arr_spectral[i, :].get()), 'o')
plt.title('imag')

plt.figure()
for i in range(distribution.arr_spectral.shape[0]):
    plt.plot(grid.v.modes, np.real(distribution.arr_spectral[i, :].get()), 'o')
plt.title('real')
plt.show()

# Set up elliptic solver
elliptic = ell.Elliptic(elements=elements, orders=orders)
elliptic.poisson_solve(distribution=distribution, grid=grid)

# Look at ic
plotter = my_plt.Plotter(grid=grid)
plotter.phasespace_scalar_contourf(ps_scalar=distribution)
plotter.spatial_scalar_plot(scalar=distribution.zero_moment, y_axis='n')
plotter.spatial_scalar_plot(scalar=elliptic.potential, y_axis='phi')
plotter.show_all()

# Set up semi-discrete
semi_discrete = sd.Spectral()

# Set up time-stepper
stepper = ts.Stepper(time_order=3, space_order=orders[0], write_time=write_time, final_time=final_time)
stepper.main_loop(distribution=distribution, grid=grid, elliptic=elliptic, semi_discrete=semi_discrete)


# Look at Fourier-Hermite spectrum
plt.figure()
for i in range(distribution.arr_spectral.shape[0]):
    plt.plot(grid.v.modes, np.imag(distribution.arr_spectral[i, :].get()), 'o')
plt.title('imag')

plt.figure()
for i in range(distribution.arr_spectral.shape[0]):
    plt.plot(grid.v.modes, np.real(distribution.arr_spectral[i, :].get()), 'o')
plt.title('real')
plt.show()

# Look at final state
distribution.invert_fourier_hermite_transform(grid=grid)
distribution.zero_moment.inverse_fourier_transform(grid=grid)
elliptic.potential.inverse_fourier_transform(grid=grid)
# plotter = my_plt.Plotter(grid=grid)
plotter.phasespace_scalar_contourf(ps_scalar=distribution)
plotter.spatial_scalar_plot(scalar=distribution.zero_moment, y_axis='n')
plotter.spatial_scalar_plot(scalar=elliptic.potential, y_axis='phi')
plotter.plot_saved_scalars(saved_array=stepper.saved_array)

plotter.show_all()
