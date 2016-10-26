import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from fractions import gcd
from types import MethodType, FunctionType
import cufft

import skcuda.linalg as cu_linalg
cu_linalg.init()


class PaddedWignerCUDA1D:
    """
    The second-order split-operator propagator for the Wigner function W(x, p, t)
    with the time-dependent Hamiltonian H = K(p, t) + V(x, t) using CUDA.

    The Wigner function is obtained by padded Wigner transforming the (rectangular) density matrix,
    which is propagated via the von Neumann equation by assuming the periodic boundaries.

    This implementation stores the Wigner function as a 2D real gpu array.
    """
    def __init__(self, **kwargs):
        """
        The following parameters are to be specified
            X_gridDIM - the coordinate grid size
            X_amplitude - maximum value of the coordinates

            t (optional) - initial value of time (default t = 0)
            consts (optional) - a string of the C code declaring the constants
            functions (optional) -  a string of the C code declaring auxiliary functions
            V - a string of the C code specifying potential energy. Coordinate (X) and time (t) variables are declared.
            K - a string of the C code specifying kinetic energy. Momentum (P) and time (t) variables are declared.
            diff_V (optional) - a string of the C code specifying the potential energy derivative w.r.t. X
                                    for the Ehrenfest theorem calculations
            diff_K (optional) - a string of the C code specifying the kinetic energy derivative w.r.t. P
                                    for the Ehrenfest theorem calculations
            dt - time step

            abs_boundary_p (optional) - a string of the C code specifying function of PP and PP_prime,
                                    which will be applied to the density matrix at each propagation step
            abs_boundary_x (optional) - a string of the C code specifying function of XX and XX_prime,
                                    which will be applied to the density matrix at each propagation step

            max_thread_block (optional) - the maximum number of GPU processes to be used (default 512)
        """
        # save all attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        # Check that all attributes were specified
        try:
            self.X_gridDIM
        except AttributeError:
            raise AttributeError("Coordinate grid size (X_gridDIM) was not specified")

        assert 2 ** int(np.log2(self.X_gridDIM)) == self.X_gridDIM, \
            "Coordinate grid size (X_gridDIM) must be a power of two"

        try:
            self.X_amplitude
        except AttributeError:
            raise AttributeError("Coordinate grid range (X_amplitude) was not specified")

        try:
            self.V
        except AttributeError:
            raise AttributeError("Potential energy (V) was not specified")

        try:
            self.K
        except AttributeError:
            raise AttributeError("Momentum dependence (K) was not specified")

        try:
            self.dt
        except AttributeError:
            raise AttributeError("Time-step (dt) was not specified")

        try:
            self.t
            del kwargs['t']
        except AttributeError:
            print("Warning: Initial time (t) was not specified, thus it is set to zero.")
            self.t = 0.

        # Save the current value of t as the initial time
        kwargs.update(t_initial=self.t)

        self.t = np.float64(self.t)

        ##########################################################################################
        #
        # Generating grids
        #
        ##########################################################################################

        # get coordinate and momentum step sizes
        self.dX = 2. * self.X_amplitude / self.X_gridDIM

        # coordinate grid
        self.X = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX, self.X_gridDIM)

        # momentum grid
        self.P = np.fft.fftshift(np.fft.fftfreq(self.X.size, self.dX / (2. * np.pi)))
        self.dP = self.P[1] - self.P[0]

        # Padding for Wigner function (from one-side)
        self.wigner_padding = int(np.ceil(
            0.5 * (np.sqrt(2.) - 1.) * self.X.size
        ))

        # number of points for Wigner function
        self.XX_gridDIM = self.X.size + 2 * self.wigner_padding

        self.X_wigner = self.dX / np.sqrt(2.) * (np.arange(self.XX_gridDIM) - self.XX_gridDIM // 2)
        self.dX_wigner = self.X_wigner[1] - self.X_wigner[0]

        PP = np.fft.fftshift(np.fft.fftfreq(self.XX_gridDIM, self.dX / (2. * np.pi)))
        self.dPP = PP[1] - PP[0]

        self.P_wigner = PP / np.sqrt(2.)
        self.dP_wigner = self.P_wigner[1] - self.P_wigner[0]

        self.dXdP_wigner = self.dX_wigner * self.dP_wigner

        ##########################################################################################
        #
        # Save CUDA constants
        #
        ##########################################################################################

        kwargs.update(
            dX=self.dX,
            dP=self.dP,

            XX_gridDIM=self.XX_gridDIM,
            dXX=self.dX,
            dPP=self.dPP,

            # These constants are useful for constructing the absorbing boundaries
            P_max=self.P.max(),
            X_max=self.X.max(),
        )

        self.cuda_consts = ""

        # Convert real constants into CUDA code
        for name, value in kwargs.items():
            if isinstance(value, int):
                self.cuda_consts += "#define %s %d\n" % (name, value)
            elif isinstance(value, float):
                self.cuda_consts += "#define %s %.15e\n" % (name, value)

        # Append user defined constants, if specified
        try:
            self.cuda_consts += self.consts
        except AttributeError:
            pass

        ##########################################################################################
        #
        # Absorbing boundaries in different representations
        #
        ##########################################################################################

        try:
            self.abs_boundary_x
        except AttributeError:
            self.abs_boundary_x = "1."

        try:
            self.abs_boundary_p
        except AttributeError:
            self.abs_boundary_p = "1."

        ##########################################################################################
        #
        # Allocate memory for the wigner function and the denisty matrix
        #
        ##########################################################################################

        # Allocate the Wigner function
        self.wignerfunction = gpuarray.zeros((self.X_wigner.size, self.X_wigner.size), np.complex128)

        # Allocate the density matrix
        self.rho = gpuarray.zeros((self.X.size, self.X.size), np.complex128)

        ##########################################################################################
        #
        # Set-up CUDA FFT
        #
        ##########################################################################################

        self.plan_wigner_ax0 = cufft.Plan_Z2Z_2D_Axis0(self.wignerfunction.shape)
        self.plan_wigner_ax1 = cufft.Plan_Z2Z_2D_Axis1(self.wignerfunction.shape)

        self.plan_rho_2D = cufft.PlanZ2Z(self.rho.shape)

        ##########################################################################################
        #
        #   Define block and grid parameters for CUDA kernel
        #
        ##########################################################################################

        #  Make sure that self.max_thread_block is defined
        # i.e., the maximum number of GPU processes to be used (default 512)
        try:
            self.max_thread_block
        except AttributeError:
            self.max_thread_block = 512

        # If the X grid size is smaller or equal to the max number of CUDA threads
        # then use all self.X_gridDIM processors
        # otherwise number of processor to be used is the greatest common divisor of these two attributes
        size_x = self.X.size
        nproc = (size_x if size_x <= self.max_thread_block else gcd(size_x, self.max_thread_block))

        # CUDA block and grid for functions that act on the whole denisty matrix
        self.rho_mapper_params = dict(
            block=(nproc, 1, 1),
            grid=(size_x // nproc, size_x)
        )

        # CUDA block and grid for functions that act on the whole Wigner function
        size_x = self.XX_gridDIM
        nproc = (size_x if size_x <= self.max_thread_block else gcd(size_x, self.max_thread_block))

        self.wigner_mapper_params = dict(
            block=(nproc, 1, 1),
            grid=(size_x // nproc, size_x)
        )

        ##########################################################################################
        #
        # Generate CUDA functions applying the exponents
        #
        ##########################################################################################

        # Append user defined functions
        try:
            self.cuda_consts += self.functions
        except AttributeError:
            pass

        print("\n================================ Compiling wigner propagators ================================\n")

        wigner_cuda_compiled = SourceModule(
            self.wigner_cuda_source.format(
                cuda_consts=self.cuda_consts,
                K=self.K, V=self.V,
                abs_boundary_x=self.abs_boundary_x, abs_boundary_p=self.abs_boundary_p
            )
        )

        self.phase_shearX = wigner_cuda_compiled.get_function("phase_shearX")
        self.phase_shearY = wigner_cuda_compiled.get_function("phase_shearY")
        self.sign_flip = wigner_cuda_compiled.get_function("sign_flip")

        self.expV = wigner_cuda_compiled.get_function("expV")
        self.expK = wigner_cuda_compiled.get_function("expK")

        self.blackmanX = wigner_cuda_compiled.get_function("blackmanX")
        self.blackmanY = wigner_cuda_compiled.get_function("blackmanY")

        """
        ##########################################################################################
        #
        #   Initialize facility for calculating expectation values of the current wave function
        #   see the implementation of self.get_average
        #
        ##########################################################################################

        # This array is used for expectation value calculation
        self.weighted = gpuarray.zeros(self.wavefunction.shape, np.float64)

        # hash table of cuda compiled functions that calculate an average of specified observable
        self._compiled_observable = dict()

        ##########################################################################################
        #
        #   Ehrenfest theorems (optional)
        #
        ##########################################################################################

        try:
            # Check whether the necessary terms are specified to calculate the Ehrenfest theorems
            self.diff_K
            self.diff_V

            # Lists where the expectation values of X and P
            self.X_average = []
            self.P_average = []

            # Lists where the right hand sides of the Ehrenfest theorems for X and P
            self.X_average_RHS = []
            self.P_average_RHS = []

            # List where the expectation value of the Hamiltonian will be calculated
            self.hamiltonian_average = []

            # Flag requesting tha the Ehrenfest theorem calculations
            self.isEhrenfest = True

        except AttributeError:
            # Since self.diff_V and self.diff_K are not specified,
            # the Ehrenfest theorem will not be calculated
            self.isEhrenfest = False
        """

        self.print_memory_info()

    @classmethod
    def print_memory_info(cls):
        """
        Print the CUDA memory info
        :return:
        """
        print(
            "\n\n\t\tGPU memory Total %.2f GB\n\t\tGPU memory Free %.2f GB\n" % \
            tuple(np.array(pycuda.driver.mem_get_info()) / 2. ** 30)
        )

    def set_rho(self, new_rho):
        """
        Set the initial density matrix
        :param new_rho: 2D numpy array, 2D GPU array containing the density matrix,
                    a string of the C code specifying the initial condition,
                    a python function of the form F(self, x, x_prime), or a float number
                    Coordinates (X) and momentum (X_prime) variables are declared.
        :return: self
        """
        if isinstance(new_rho, (np.ndarray, gpuarray.GPUArray)):
            # perform the consistency checks
            assert new_rho.shape == self.rho.shape, "The grid sizes does not match"

            # copy density matrix
            self.rho[:] = new_rho.astype(np.complex128)

        elif isinstance(new_rho, FunctionType):
            # user supplied the function which will return the density matrix
            self.rho[:] = new_rho(self, self.X[np.newaxis, :], self.X[:, np.newaxis])

        elif isinstance(new_rho, str):
            # user specified C code
            print("\n================================ Compiling init_rho ================================\n")
            init_rho = SourceModule(
                self.init_rho_source.format(cuda_consts=self.cuda_consts, new_rho_func=new_rho),
            ).get_function("Kernel")
            init_rho(self.rho, **self.rho_mapper_params)

        elif isinstance(new_rho, (float, complex)):
            # user specified a constant
            self.rho.fill(np.complex128(new_rho))
        else:
            raise NotImplementedError("new_rho must be either function or numpy.array")

        # normalize
        self.rho /= cu_linalg.trace(self.rho) * self.dX

        return self

    def get_wigner(self):
        """
        Transform the density matrix saved in self.rho into the unormalized Wigner function
        :return: self.wignerfunction
        """

        self.wignerfunction.fill(np.float64(0.))

        # Make a copy of the density matrix
        self.wignerfunction[
            self.wigner_padding:(self.wigner_padding + self.X_gridDIM),
            self.wigner_padding:(self.wigner_padding + self.X_gridDIM)
        ] = self.rho

        # Step 1: Rotate by +45 degrees
        # Shear X
        cufft.fft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_wigner_ax1)
        self.phase_shearX(self.wignerfunction, **self.wigner_mapper_params)
        cufft.ifft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_wigner_ax1)

        # Shear Y
        cufft.fft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_wigner_ax0)
        self.phase_shearY(self.wignerfunction, **self.wigner_mapper_params)
        cufft.ifft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_wigner_ax0)

        # Shear X
        cufft.fft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_wigner_ax1)
        self.phase_shearX(self.wignerfunction, **self.wigner_mapper_params)
        cufft.ifft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_wigner_ax1)

        # Step 2: FFt the Blokhintsev function
        self.sign_flip(self.wignerfunction, **self.wigner_mapper_params)
        cufft.ifft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_wigner_ax0)
        self.sign_flip(self.wignerfunction, **self.wigner_mapper_params)

        self.wignerfunction /= gpuarray.sum(self.wignerfunction).get().real * self.dXdP_wigner

        return self.wignerfunction

    def propagate(self, steps=1):
        """
        Time propagate the density matrix saved in self.rho
        :param steps: number of self.dt time increments to make
        :return: self.wignerfunction
        """
        for _ in xrange(steps):
            # increment current time
            self.t += self.dt

            # advance by one time step
            self.single_step_propagation()

            # normalize
            self.rho /= cu_linalg.trace(self.rho) * self.dX

            # calculate the Ehrenfest theorems
            #self.get_Ehrenfest(self.t)

        return self

    def single_step_propagation(self):
        """
        Perform a single step propagation. The final density matrix function is not normalized.
        :return: self.rho
        """
        self.expV(self.rho, self.t, **self.rho_mapper_params)

        cufft.fft_Z2Z(self.rho, self.rho, self.plan_rho_2D)

        self.expK(self.rho, self.t, **self.rho_mapper_params)

        cufft.ifft_Z2Z(self.rho, self.rho, self.plan_rho_2D)
        #self.rho /= self.rho.shape[0] * self.rho.shape[1]

        self.expV(self.rho, self.t, **self.rho_mapper_params)

        return self.rho

    wigner_cuda_source = """
    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES

    typedef pycuda::complex<double> cuda_complex;

    {cuda_consts}

    ////////////////////////////////////////////////////////////////////////////
    //
    // CUDA code to apply phase factors to perform X and Y shearing
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void phase_shearX(cuda_complex *rho)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * XX_gridDIM;

        const double PP = dPP * ((j + XX_gridDIM / 2) % XX_gridDIM - 0.5 * XX_gridDIM);
        const double XX_prime = dXX * (i - 0.5 * XX_gridDIM);

        // perform rotation by theta: const double a = tan(0.5 * theta);
        const double a = tan(M_PI / 8.);
        const double phase = -a * PP * XX_prime;

        rho[indexTotal] *= cuda_complex(cos(phase), sin(phase)) / double(XX_gridDIM);

    }}

    __global__ void phase_shearY(cuda_complex *rho)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * XX_gridDIM;

        const double XX = dXX * (j - 0.5 * XX_gridDIM);
        const double PP_prime = dPP * ((i + XX_gridDIM / 2) % XX_gridDIM - 0.5 * XX_gridDIM);

        // perform rotation by theta: const double b = -sin(theta);
        const double b = -sin(M_PI / 4.);
        const double phase = -b * PP_prime * XX;

        rho[indexTotal] *= cuda_complex(cos(phase), sin(phase)) / double(XX_gridDIM);
    }}

    ////////////////////////////////////////////////////////////////////////////
    //
    //  Blackman filters
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void blackmanX(cuda_complex *rho)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * XX_gridDIM;

        const double Cj = j * 2.0 * M_PI / (XX_gridDIM - 1.);

        rho[indexTotal] *= 0.42 - 0.5 * cos(Cj) + 0.08 * cos(2.0 * Cj);
    }}

    __global__ void blackmanY(cuda_complex *rho)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * XX_gridDIM;

        const double Ci = i * 2.0 * M_PI / (XX_gridDIM - 1.);

        rho[indexTotal] *= 0.42 - 0.5 * cos(Ci) + 0.08 * cos(2.0 * Ci);
    }}

    ////////////////////////////////////////////////////////////////////////////
    //
    // CUDA code to multiply with (-1)^i in order for FFT
    // to approximate the Fourier integral over theta
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void sign_flip(cuda_complex *rho)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * XX_gridDIM;

        // rho *= pow(-1, i)
        rho[indexTotal] *= 1 - 2 * int(i % 2);
    }}

    ////////////////////////////////////////////////////////////////////////////
    //
    //  Propagator
    //
    ////////////////////////////////////////////////////////////////////////////

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

    // Absorbing potential in X
    __device__ double abs_boundary_x(double X)
    {{
        return ({abs_boundary_x});
    }}

    // Absorbing potential in P
    __device__ double abs_boundary_p(double P)
    {{
        return ({abs_boundary_p});
    }}

    ////////////////////////////////////////////////////////////////////////////
    //
    // CUDA code to define the action of the kinetic energy exponent
    // onto the density matrix in the momentum representation < P | rho | P_prime >
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void expK(cuda_complex *rho, double t)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        // fft shifting momentum
        const double P = dP * ((j + X_gridDIM / 2) % X_gridDIM - 0.5 * X_gridDIM);
        const double P_prime = dP * ((i + X_gridDIM / 2) % X_gridDIM - 0.5 * X_gridDIM);

        const double phase = -dt * (K(P, t + 0.5 * dt) - K(P_prime, t + 0.5 * dt));

        rho[indexTotal] *= cuda_complex(cos(phase), sin(phase)) * abs_boundary_p(P) * abs_boundary_p(P_prime);
    }}

    ////////////////////////////////////////////////////////////////////////////
    //
    // CUDA code to define the action of the potential energy exponent
    // onto the density matrix in the coordinate representation < X | rho | X_prime >
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void expV(cuda_complex *rho, double t)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double X_prime = dX * (i - 0.5 * X_gridDIM);

        const double phase = -0.5 * dt * (V(X, t + 0.5 * dt) - V(X_prime, t + 0.5 * dt));

        rho[indexTotal] *= cuda_complex(cos(phase), sin(phase)) * abs_boundary_x(X) * abs_boundary_x(X_prime);
    }}
    """

    init_rho_source = """
    // CUDA code to initialize the denisty matrix
    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES

    typedef pycuda::complex<double> cuda_complex;

    {cuda_consts}

    __global__ void Kernel(cuda_complex *rho)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double X_prime = dX * (i - 0.5 * X_gridDIM);

        rho[indexTotal] = ({new_rho_func});
    }}
    """

##########################################################################################
#
# Example
#
##########################################################################################

if __name__ == '__main__':

    print(PaddedWignerCUDA1D.__doc__)

    # load tools for creating animation
    import sys
    import matplotlib

    if sys.platform == 'darwin':
        # only for MacOS
        matplotlib.use('TKAgg')

    import matplotlib.animation
    import matplotlib.pyplot as plt


    class VisualizeDynamicsPhaseSpace:
        """
        Class to visualize the Wigner function function dynamics in phase space.
        """

        def __init__(self, fig):
            """
            Initialize all propagators and frame
            :param fig: matplotlib figure object
            """
            #  Initialize systems
            self.set_quantum_sys()

            #################################################################
            #
            # Initialize plotting facility
            #
            #################################################################

            self.fig = fig

            ax = fig.add_subplot(121)

            ax.set_title('Wigner function, $W(x,p,t)$')
            extent = [
                self.quant_sys.X_wigner.min(), self.quant_sys.X_wigner.max(),
                self.quant_sys.P_wigner.min(), self.quant_sys.P_wigner.max()
            ]

            # import utility to visualize the wigner function
            from wigner_normalize import WignerNormalize

            # generate empty plot
            self.img = ax.imshow(
                [[]],
                extent=extent,
                origin='lower',
                interpolation='nearest',
                cmap='seismic',
                norm=WignerNormalize(vmin=-0.1, vmax=0.1)
            )

            self.fig.colorbar(self.img)

            ax.set_xlabel('$x$ (a.u.)')
            ax.set_ylabel('$p$ (a.u.)')

            #ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])

            ax = self.fig.add_subplot(122)

            self.lines_wigner_marginal, = ax.plot(
                [self.quant_sys.X_wigner.min(), self.quant_sys.X_wigner.max()], [0, 0.3],
                'r', label='Wigner marginal'
            )
            self.lines_rho, = ax.plot(
                [self.quant_sys.X.min(), self.quant_sys.X.max()], [0, 0.3],
                'b', label='Density matrix diagonal'
            )
            ax.legend()

            ax.set_xlabel('X (a.u.)')
            ax.set_ylabel('Probability')

        def set_quantum_sys(self):
            """
            Initialize quantum propagator
            :param self:
            :return:
            """
            # Create propagator
            self.quant_sys = RhoWignerCUDA1D(
                t=0.,
                dt=0.01,

                X_gridDIM=1024,
                X_amplitude=10.,

                # randomized parameter
                omega_square=np.random.uniform(2., 6.),

                # randomized parameters for initial condition
                sigma=np.random.uniform(0.5, 4.),
                p0=np.random.uniform(-1., 1.),
                x0=np.random.uniform(-1., 1.),

                # kinetic energy part of the hamiltonian
                K="0.5 * P * P",

                # potential energy part of the hamiltonian
                V="0.0", #"0.5 * omega_square * X * X",

                # these functions are used for evaluating the Ehrenfest theorems
                diff_K="P",
                diff_V="omega_square * X"
            )

            # set randomised initial condition
            self.quant_sys.set_rho(
                "exp(-sigma * pow(X - x0, 2) + cuda_complex(0., p0) * X) *"
                "exp(-sigma * pow(X_prime - x0, 2) - cuda_complex(0., p0) * X_prime)"
            )

        def empty_frame(self):
            """
            Make empty frame and reinitialize quantum system
            :param self:
            :return: image object
            """
            self.img.set_array([[]])
            self.lines_wigner_marginal.set_data([], [])
            self.lines_rho.set_data([], [])
            return self.img, self.lines_wigner_marginal, self.lines_rho

        def __call__(self, frame_num):
            """
            Draw a new frame
            :param frame_num: current frame number
            :return: image objects
            """
            # propagate the wave function and then get the Wigner function
            W = self.quant_sys.propagate(10).get_wigner().get()

            self.img.set_array(W.real)

            x_marginal = W.sum(axis=0).real
            x_marginal *= self.quant_sys.dP_wigner

            self.lines_wigner_marginal.set_data(self.quant_sys.X_wigner, x_marginal)
            self.lines_rho.set_data(
                self.quant_sys.X, self.quant_sys.rho.get().diagonal().real
            )

            return self.img, self.lines_wigner_marginal, self.lines_rho


    fig = plt.gcf()
    visualizer = VisualizeDynamicsPhaseSpace(fig)
    animation = matplotlib.animation.FuncAnimation(
        fig, visualizer, frames=np.arange(100), init_func=visualizer.empty_frame, repeat=True, blit=True
    )

    plt.show()

    """
    # extract the reference to quantum system
    quant_sys = visualizer.quant_sys

    # Analyze how well the energy was preserved
    h = np.array(quant_sys.hamiltonian_average)
    print(
        "\nHamiltonian is preserved within the accuracy of %f percent" % ((1. - h.min() / h.max()) * 100)
    )

    #################################################################
    #
    # Plot the Ehrenfest theorems after the animation is over
    #
    #################################################################

    # generate time step grid
    dt = quant_sys.dt
    times = dt * np.arange(len(quant_sys.X_average)) + dt

    plt.subplot(131)
    plt.title("The first Ehrenfest theorem verification")

    plt.plot(times, np.gradient(quant_sys.X_average, dt), 'r-', label='$d\\langle x \\rangle/dt$')
    plt.plot(times, quant_sys.X_average_RHS, 'b--', label='$\\langle p \\rangle$')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(132)
    plt.title("The second Ehrenfest theorem verification")

    plt.plot(times, np.gradient(quant_sys.P_average, dt), 'r-', label='$d\\langle p \\rangle/dt$')
    plt.plot(times, quant_sys.P_average_RHS, 'b--', label='$\\langle -\\partial V/\\partial x \\rangle$')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(133)
    plt.title('Hamiltonian')
    plt.plot(times, h)
    plt.xlabel('time $t$ (a.u.)')

    plt.show()
    """
