import numpy as np
# import cupy as cp
# import basis as b
import grid as g
import variables as var
import semidiscrete as sd
import elliptic as ell
import timestep as ts
import plotter as my_plt

# elements and order
elements, orders = [20, 20], [8, 8]

# Set up phase space grid
lows = np.array([-np.pi, -5.0])
highs = np.array([np.pi, 5.0])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, orders=orders)

# Build distribution, elliptic, etc.
distribution = var.PhaseSpaceScalar(resolutions=elements, orders=orders)
distribution.initialize(grid=grid)
elliptic = ell.Elliptic(elements=elements, orders=orders)
elliptic.poisson_solve(distribution=distribution, grid=grid)

plotter = my_plt.Plotter(grid=grid)
plotter.spatial_scalar_plot(scalar=distribution.zero_moment)
plotter.show_all()
