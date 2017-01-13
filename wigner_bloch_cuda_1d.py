from wigner_moyal_cuda_1d import WignerMoyalCUDA1D

import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import cufft


class WignerBlochCUDA1D(WignerMoyalCUDA1D):
    """
    Find the Wigner function of the Boltzmann-Gibbs canonical state [rho = exp(-H/kT)]
    by second-order split-operator propagation of the Bloch equation in phase space using CUDA.
    The Hamiltonian should be of the form H = K(p) + V(x).

    This implementation is based on the algorithm described in
        D. I Bondar, A. G. Campos, R. Cabrera, H. A. Rabitz, Phys. Rev. E 93, 063304 (2016)
        (http://link.aps.org/doi/10.1103/PhysRevE.93.063304)
    """
    def __init__(self, **kwargs):
        """
        In addition to kwagrs of WignerMoyalCUDA1D.__init__ this constructor accepts:

        kT (optional)- the temperature for the Gibbs state [rho = exp(-H/kT)]
        dbeta (optional) -  inverse temperature increments for the split-operator propagation
        t_initial (optional) - if the Hamiltonian is time dependent, then the the Gibbs state will be calculated
            for the hamiltonian at t_initial (default value of zero).
        """
        if 't_initial' not in kwargs:
            kwargs.update(t_initial=0.)
            print("Warning: Initial time (t_initial) was not specified. So the default value was used t_initial = 0.")

        try:
            self.kT = kwargs['kT']
            # remove kT from kwargs so that it does not enter into self.cuda_consts
            del kwargs['kT']
        except KeyError:
            self.kT = 0.

        try:
            self.dbeta = kwargs['dbeta']
            # remove dbeta from kwargs so that it does not enter into self.cuda_consts
            del kwargs['dbeta']
        except KeyError:
            # if dbeta is not defined, just choose some value
            self.dbeta = 0.01

        if 'dt' not in kwargs:
            # Save the inverse temperature increment as dt
            kwargs.update(dt=self.dbeta)

        # Initialize parent class
        WignerMoyalCUDA1D.__init__(self, **kwargs)

        # Save the minimums of the potential (V) and kinetic (K) energy
        self.cuda_consts += "#define V_min %.15e \n" % self.get_V_min()
        self.cuda_consts += "#define K_min %.15e \n" % self.get_K_min()

        print("\n================================ Compiling Bloch expK and expV ================================\n")

        bloch_compiled = SourceModule(
            self.bloch_cuda_source.format(
                cuda_consts=self.cuda_consts, K=self.K, V=self.V,
                abs_boundary_p=self.abs_boundary_p, abs_boundary_x=self.abs_boundary_x
            )
        )

        self.bloch_expK = bloch_compiled.get_function("bloch_expK")
        self.bloch_expV = bloch_compiled.get_function("bloch_expV")

    def get_V_min(self):
        """
        Return the potential energy minimum
        """
        # fill array with values of the potential energy
        fill_compiled = SourceModule(
            self.fill_V_K.format(cuda_consts=self.cuda_consts, K=self.K, V=self.V)
        )
        fill_compiled.get_function("fill_V")(self.weighted, **self.wigner_mapper_params)

        return gpuarray.min(self.weighted).get()

    def get_K_min(self):
        """
        Return the kinetic energy minimum
        """
        # fill array with values of the kinetic energy
        fill_compiled = SourceModule(
            self.fill_V_K.format(cuda_consts=self.cuda_consts, K=self.K, V=self.V)
        )
        fill_compiled.get_function("fill_K")(self.weighted, **self.wigner_mapper_params)

        return gpuarray.min(self.weighted).get()

    @classmethod
    def get_dbeta_num_beta_steps(cls, kT, dbeta):
        """
        Calculate the number of propagation steps (num_beta_steps) and the inverse temperature step size (dbeta)
        needed to reach the Gibbs state with temperature kT
        :param kT: (float) temperature of the desired Gibbs state
        :param dbeta: (float) initial guess for the inverse temperature step size
        :return: dbeta, num_beta_steps
        """
        if kT > 0:
            # get number of dbeta steps to reach the desired Gibbs state
            num_beta_steps = 1. / (kT * dbeta)

            if round(num_beta_steps) <> num_beta_steps:
                # Changing self.dbeta so that num_beta_steps is an exact integer
                num_beta_steps = round(num_beta_steps)
                dbeta = 1. / (kT * num_beta_steps)

            num_beta_steps = int(num_beta_steps)
        else:
            # assume zero temperature
            num_beta_steps = np.inf

        return dbeta, num_beta_steps

    def get_gibbs_state(self, kT=None, dbeta=None):
        """
        Calculate the Boltzmann-Gibbs state and save it in self.wignerfunction
        :param dbeta: (float) the inverse temperature step increment
        :param kT: (float) temperature of the desired Gibbs state
        :return: self.wignerfunction
        """
        # initialize the propagation parameters
        if dbeta is None:
            dbeta = self.dbeta

        if kT is None:
            kT = self.kT

        dbeta, num_beta_steps = self.get_dbeta_num_beta_steps(kT, dbeta)
        dbeta = np.float64(dbeta)

        # Set the infinite temperature initial state
        self.set_wignerfunction(1. / np.prod(self.wignerfunction.shape))

        try:
            for k in xrange(num_beta_steps):
                # advance by one time step
                self.bloch_single_step_propagation(dbeta)

                # verify whether the obtained state is physical:
                # Purity cannot be larger than one
                current_purity = self.get_purity()
                assert current_purity < 1.

        except AssertionError:
            print(
                "Warning: Gibbs state calculations interupted because purity = %.10f > 1."
                "Current kT = %.6f" % (self.get_purity(), 1 / (k * dbeta))
            )

        return self.wignerfunction

    def get_ground_state(self, dbeta=None, abs_tol_purity=1e-12):
        """
        Obtain the ground state Wigner function with specified accuracy
        :param dbeta: (float) the inverse temperature step increment
        :param abs_tol_purity: (float) the obtained Wigner function the termination criterion
        :return: self.wignerfunction
        """
        # Initialize varaibles
        previous_energy = current_energy = np.inf
        previous_purity = current_purity = 0.

        if dbeta is None:
            dbeta = self.dbeta
        dbeta = np.float64(dbeta)

        # Set the infinite temperature initial state
        self.set_wignerfunction(1. / np.prod(self.wignerfunction.shape))

        # Allocate memory for extra copy of the density matrix
        previous_rho = self.rho.copy()

        while current_purity < (1. - abs_tol_purity) and dbeta > 1e-11:
            # advance by one time step
            self.bloch_single_step_propagation(dbeta)

            try:
                # Purity cannot be larger than one
                current_purity = self.get_purity()
                assert current_purity <= 1.

                # Check whether the state cooled
                current_energy = self.get_average(self.hamiltonian)
                assert current_energy < previous_energy

                # Verify the uncertainty principle
                assert self.get_sigma_x_sigma_p() >= 0.5

                # the current state seems to be physical, so we accept it
                previous_energy = current_energy
                previous_purity = current_purity

                # make a copy of the current state
                gpuarray._memcpy_discontig(previous_rho, self.rho)

                print(
                    "Current energy: %.5f; purity: 1 - %.2e; dbeta: %.2e"
                    % (current_energy, 1 - current_purity, dbeta)
                )
            except AssertionError:
                # the current state is unphysical,
                # revert the propagation
                gpuarray._memcpy_discontig(self.rho, previous_rho)

                # and half the step size
                dbeta *= 0.5

                # restore the original settings
                current_energy = previous_energy
                current_purity = previous_purity

        self.rho2wigner()

        return self.wignerfunction

    def bloch_single_step_propagation(self, dbeta):
        """
        Perform a single step propagation with respect to the inverse temperature via the Bloch equation.
        The final Wigner function is not normalized.
        :param dbeta: (float) the inverse temperature step size
        :return:
        """
        self.bloch_expV(self.rho, dbeta, **self.rho_mapper_params)

        cufft.fft_Z2Z(self.rho, self.rho, self.plan_Z2Z_ax0)
        cufft.fft_Z2Z(self.rho, self.rho, self.plan_Z2Z_ax1)

        self.bloch_expK(self.rho, dbeta, **self.rho_mapper_params)

        cufft.ifft_Z2Z(self.rho, self.rho, self.plan_Z2Z_ax0)
        cufft.ifft_Z2Z(self.rho, self.rho, self.plan_Z2Z_ax1)

        self.bloch_expV(self.rho, dbeta, **self.rho_mapper_params)

        self.normalize_rho_wigner()

    bloch_cuda_source = """
    /////////////////////////////////////////////////////////////////////////////
    //
    // This code closely follows WignerMoyalCUDA1D.expK_cuda_source
    //
    /////////////////////////////////////////////////////////////////////////////

    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES

    typedef pycuda::complex<double> cuda_complex;

    {cuda_consts}

    // Kinetic energy
    __device__ double K(double P, double t)
    {{
        return ({K});
    }}

    // Potential energy
    __device__ double V(double X, double t)
    {{
        return ({V});
    }}

    ////////////////////////////////////////////////////////////////////////////
    //
    // CUDA code to define the action of the kinetic energy exponent
    // onto the density matrix in the momentum representation < PP | rho | PP_prime >
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void bloch_expK(cuda_complex *W, double dbeta)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double PP = dPP * (j - 0.5 * X_gridDIM);
        const double PP_prime = dPP_prime * (i - 0.5 * P_gridDIM);

        const double phase = -0.5 * dbeta * (
            K(PP, t_initial) + K(PP_prime, t_initial) - K_min
        );

        W[indexTotal] *= exp(phase); // * ({abs_boundary_p});
    }}

    ////////////////////////////////////////////////////////////////////////////
    //
    // CUDA code to define the action of the potential energy exponent
    // onto the density matrix in the coordinate representation < XX | rho | XX_prime >
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void bloch_expV(cuda_complex *W, double dbeta)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double XX = dXX * (j - 0.5 * X_gridDIM);
        const double XX_prime = dXX_prime * (i - 0.5 * P_gridDIM);

        const double phase = -0.25 * dbeta * (
            V(XX, t_initial) + V(XX_prime, t_initial) - V_min
        );

        // sign_flip = pow(-1, i + j)
        const double sign_flip = 1. - 2. * int((i + j) % 2);

        W[indexTotal] *= sign_flip * exp(phase); // * ({abs_boundary_x});
    }}
    """

    fill_V_K = """
    ////////////////////////////////////////////////////////////////////////////
    //
    // functions to find minimum of the kinetic and potential energy
    //
    ////////////////////////////////////////////////////////////////////////////

    #include<math.h>
    #define _USE_MATH_DEFINES

    {cuda_consts}

    // Kinetic energy
    __device__ double K(double P, double t)
    {{
        return ({K});
    }}

    // Potential energy
    __device__ double V(double X, double t)
    {{
        return ({V});
    }}

    __global__ void fill_V(double *tmp)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double XX = dXX * (j - 0.5 * X_gridDIM);
        const double XX_prime = dXX_prime * (i - 0.5 * P_gridDIM);

        tmp[indexTotal] = V(XX, t_initial) + V(XX_prime, t_initial);
    }}

    __global__ void fill_K(double *tmp)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double PP = dPP * (j - 0.5 * X_gridDIM);
        const double PP_prime = dPP_prime * (i - 0.5 * P_gridDIM);

        tmp[indexTotal] = K(PP, t_initial) + K(PP_prime, t_initial);
    }}
    """

##########################################################################################
#
# Example
#
##########################################################################################

if __name__ == '__main__':

    print(WignerBlochCUDA1D.__doc__)

    import matplotlib.pyplot as plt

    # parameters for the quantum system
    params = dict(
        t=0.,
        dt=0.01,

        X_gridDIM=512,
        X_amplitude=10.,

        P_gridDIM=512,
        P_amplitude=10.,

        # Temperature of the initial state
        kT=np.random.uniform(0.1, 1.),

        # randomized parameter
        omega=np.random.uniform(1., 2.),

        # kinetic energy part of the hamiltonian
        K="0.5 * P * P",

        # potential energy part of the hamiltonian
        V="0.5 * omega * omega * X * X",

        # Hamiltonian
        H=lambda self, x, p: 0.5*(p**2 + self.omega**2 * x**2),

        # Exact analytical expression for the harmonic oscillator Gibbs state
        get_exact_gibbs=lambda self: np.tanh(0.5 * self.omega / self.kT) / np.pi * np.exp(
            -2. * np.tanh(0.5 * self.omega / self.kT) * self.H(self.X, self.P) / self.omega
        )
    )

    print("Calculating the Gibbs state...")
    gibbs_state = WignerBlochCUDA1D(**params).get_gibbs_state()

    print("Check that the obtained Gibbs state is stationary under the Wigner-Moyal propagation...")
    propagator = WignerMoyalCUDA1D(**params)
    final_state = propagator.set_wignerfunction(gibbs_state).propagate(3000).get()

    gibbs_state = gibbs_state.get()

    exact_gibbs = propagator.get_exact_gibbs()
    print(
        "\nIninity norm between analytical and numerical Gibbs states = %.2e ." %
        (np.linalg.norm(exact_gibbs.reshape(-1) - gibbs_state.reshape(-1), np.inf) * propagator.dXdP)
    )

    ##########################################################################################
    #
    #   Plot the results
    #
    ##########################################################################################

    from wigner_normalize import WignerSymLogNorm

    # save common plotting parameters
    plot_params = dict(
        origin='lower',
        extent=[propagator.X.min(), propagator.X.max(), propagator.P.min(), propagator.P.max()],
        cmap='seismic',
        # make a logarithmic color plot (see, e.g., http://matplotlib.org/users/colormapnorms.html)
        norm=WignerSymLogNorm(linthresh=1e-14, vmin=-0.01, vmax=0.1)
    )
    plt.subplot(131)

    plt.title("The Gibbs state (initial state)")
    plt.imshow(gibbs_state.real, **plot_params)
    plt.colorbar()
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('$p$ (a.u.)')

    plt.subplot(132)

    plt.title("The exact Gibbs state")
    plt.imshow(exact_gibbs, **plot_params)
    plt.colorbar()
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('$p$ (a.u.)')

    plt.subplot(133)

    plt.title("The Gibbs state after propagation")
    plt.imshow(final_state.real, **plot_params)
    plt.colorbar()
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('$p$ (a.u.)')

    plt.show()
