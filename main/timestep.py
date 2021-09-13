import numpy as np
# import cupy as cp
import variables as var
import time as timer

# Courant numbers for RK-DG stability from Cockburn and Shu 2001, [time_order][space_order-1]
courant_numbers = {
    1: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    2: [1.0, 0.333],
    3: [1.256, 0.409, 0.209, 0.130, 0.089, 0.066, 0.051, 0.040, 0.033],
    4: [1.392, 0.464, 0.235, 0.145, 0.100, 0.073, 0.056, 0.045, 0.037],
    5: [1.608, 0.534, 0.271, 0.167, 0.115, 0.085, 0.065, 0.052, 0.042],
    6: [1.776, 0.592, 0.300, 0.185, 0.127, 0.093, 0.072, 0.057, 0.047],
    7: [1.977, 0.659, 0.333, 0.206, 0.142, 0.104, 0.080, 0.064, 0.052],
    8: [2.156, 0.718, 0.364, 0.225, 0.154, 0.114, 0.087, 0.070, 0.057]
}

nonlinear_ssp_rk_switch = {
    2: [[1 / 2, 1 / 2, 1 / 2]],
    3: [[3 / 4, 1 / 4, 1 / 4],
        [1 / 3, 2 / 3, 2 / 3]]
}


class Stepper:
    def __init__(self, time_order, space_order, write_time, final_time, method='spectral'):
        # Parameters
        self.time_order, self.space_order = time_order, space_order
        self.write_time, self.final_time = write_time, final_time
        # RK numbers
        self.rk_coefficients = np.array(nonlinear_ssp_rk_switch.get(self.time_order, "nothing"))
        # self.courant = courant_numbers.get(self.time_order)[self.space_order - 1]
        self.method = method

        # Simulation time init
        self.time = 0
        self.dt = 1.0e-2  # None
        self.steps_counter = 0
        self.write_counter = 1  # IC already written

        # Stored array
        self.saved_times = []
        self.saved_array = []

    def main_loop(self, distribution, grid, elliptic, semi_discrete):
        t0 = timer.time()
        print('\nBeginning main loop')
        self.saved_times += [self.time]
        self.saved_array += [distribution.arr_nodal]
        while self.time < self.final_time:
            # Take step
            # if self.method == 'spectral':
            self.spectral_rk(distribution=distribution, grid=grid, elliptic=elliptic,
                             semi_discrete=semi_discrete)
            # if self.method == 'nodal':
            # self.nodal_rk(distribution=distribution, grid=grid, elliptic=elliptic,
            #               semi_discrete=semi_discrete)
            # Update basis
            distribution.invert_fourier_hermite_transform(grid=grid)
            distribution.recompute_hermite_basis(grid=grid)
            distribution.fourier_hermite_transform(grid=grid)
            # Update time and steps counter
            self.time += self.dt
            self.steps_counter += 1
            if self.time > self.write_counter * self.write_time:
                print('\nI made it through step ' + str(self.steps_counter))
                self.write_counter += 1
                distribution.zero_moment.inverse_fourier_transform(grid=grid)
                self.saved_times += [self.time]
                self.saved_array += [distribution.arr_nodal]
                print(grid.v.alpha)
                print('The simulation time is {:0.3e}'.format(self.time))
                print('The time-step is {:0.3e}'.format(self.dt))
                print('Time since start is ' + str((timer.time() - t0) / 60.0) + ' minutes')
        print('\nAll done, total steps was ' + str(self.steps_counter))

    def nodal_rk(self, distribution, grid, elliptic, semi_discrete):
        # Set up stages
        stage0 = var.PhaseSpaceScalar(resolutions=[grid.x.elements, grid.v.elements],
                                      orders=[grid.x.order, grid.v.order])
        stage1 = var.PhaseSpaceScalar(resolutions=[grid.x.elements, grid.v.elements],
                                      orders=[grid.x.order, grid.v.order])

        # zero stage
        elliptic.poisson_solve(distribution=distribution, grid=grid, invert=False)
        semi_discrete.nodal_rhs(distribution=distribution,
                                grid=grid, elliptic=elliptic)
        stage0.arr_nodal = distribution.arr_nodal + self.dt * semi_discrete.rhs

        # first stage
        elliptic.poisson_solve(distribution=stage0, grid=grid, invert=False)
        semi_discrete.nodal_rhs(distribution=stage0,
                                grid=grid, elliptic=elliptic)
        stage1.arr_nodal = (self.rk_coefficients[0, 0] * distribution.arr_nodal +
                            self.rk_coefficients[0, 1] * stage0.arr_nodal +
                            self.rk_coefficients[0, 2] * self.dt * semi_discrete.rhs)

        # second stage, update
        elliptic.poisson_solve(distribution=stage1, grid=grid, invert=False)
        semi_discrete.nodal_rhs(distribution=stage1,
                                grid=grid, elliptic=elliptic)
        distribution.arr_nodal = (self.rk_coefficients[1, 0] * distribution.arr_nodal +
                                  self.rk_coefficients[1, 1] * stage1.arr_nodal +
                                  self.rk_coefficients[1, 2] * self.dt * semi_discrete.rhs)

    def spectral_rk(self, distribution, grid, elliptic, semi_discrete):
        # Set up stages
        stage0 = var.PhaseSpaceScalar(resolutions=[grid.x.elements, grid.v.elements],
                                      orders=[grid.x.order, grid.v.order])
        stage1 = var.PhaseSpaceScalar(resolutions=[grid.x.elements, grid.v.elements],
                                      orders=[grid.x.order, grid.v.order])

        # zero stage
        elliptic.poisson_solve(distribution=distribution, grid=grid, invert=False)
        semi_discrete.spectral_rhs(distribution=distribution,
                                   grid=grid, elliptic=elliptic)
        stage0.arr_spectral = distribution.arr_spectral + self.dt * semi_discrete.rhs

        # first stage
        elliptic.poisson_solve(distribution=stage0, grid=grid, invert=False)
        semi_discrete.spectral_rhs(distribution=stage0,
                                   grid=grid, elliptic=elliptic)
        stage1.arr_spectral = (self.rk_coefficients[0, 0] * distribution.arr_spectral +
                               self.rk_coefficients[0, 1] * stage0.arr_spectral +
                               self.rk_coefficients[0, 2] * self.dt * semi_discrete.rhs)

        # second stage, update
        elliptic.poisson_solve(distribution=stage1, grid=grid, invert=False)
        semi_discrete.spectral_rhs(distribution=stage1,
                                   grid=grid, elliptic=elliptic)
        distribution.arr_spectral = (self.rk_coefficients[1, 0] * distribution.arr_spectral +
                                     self.rk_coefficients[1, 1] * stage1.arr_spectral +
                                     self.rk_coefficients[1, 2] * self.dt * semi_discrete.rhs)
