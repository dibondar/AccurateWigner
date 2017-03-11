import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel
from pycuda.tools import dtype_to_ctype
import numpy as np
from fractions import gcd
from types import MethodType, FunctionType
import cufft

import skcuda.linalg as cu_linalg
cu_linalg.init()


class SchrodingerWignerCUDA1D:
    """
    The second-order split-operator propagator for the Wigner function W(x, p, t)
    with the time-dependent Hamiltonian H = K(p, t) + V(x, t) using CUDA.

    The Wigner function is obtained by Wigner transforming the wave function,
    which is propagated via the Schrodinger equation.

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

        self.X_wigner = self.X / np.sqrt(2.)
        self.dXX = self.X_wigner[1] - self.X_wigner[0]

        self.P_wigner = self.P / np.sqrt(2.)
        self.dPP = self.P_wigner[1] - self.P_wigner[0]

        self.wigner_dXdP = self.dX * self.dP

        ##########################################################################################
        #
        # Save CUDA constants
        #
        ##########################################################################################

        kwargs.update(
            dX=self.dX,
            dP=self.dP,

            # These constants are useful for constructing the absorbing boundaries
            P_max=self.P.max(),
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
        size_x = self.X_gridDIM
        nproc = (size_x if size_x <= self.max_thread_block else gcd(size_x, self.max_thread_block))

        # CUDA block and grid for functions that act on the whole wave function
        self.wavefunction_mapper_params = dict(
            block=(nproc, 1, 1),
            grid=(size_x // nproc, 1)
        )

        # CUDA block and grid for functions that act on the whole Wigner function
        self.rho_mapper_params = self.wigner_mapper_params = dict(
            block=(nproc, 1, 1),
            grid=(size_x // nproc, self.X_gridDIM)
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

        schrodinger_wigner_compiled = SourceModule(
            self.schrodinger_wigner_cuda_source.format(
                cuda_consts=self.cuda_consts,
                K=self.K, V=self.V,
                abs_boundary_x=self.abs_boundary_x, abs_boundary_p=self.abs_boundary_p
            )
        )

        self.psi2rho = schrodinger_wigner_compiled.get_function("psi2rho")

        self.phase_shearX = schrodinger_wigner_compiled.get_function("phase_shearX")
        self.phase_shearY = schrodinger_wigner_compiled.get_function("phase_shearY")
        self.sign_flip = schrodinger_wigner_compiled.get_function("sign_flip")

        self.expV = schrodinger_wigner_compiled.get_function("expV")
        self.expK = schrodinger_wigner_compiled.get_function("expK")

        self.bloch_expV = schrodinger_wigner_compiled.get_function("bloch_expV")
        self.bloch_expK = schrodinger_wigner_compiled.get_function("bloch_expK")

        self.blackmanX = schrodinger_wigner_compiled.get_function("blackmanX")
        self.blackmanY = schrodinger_wigner_compiled.get_function("blackmanY")

        ##########################################################################################
        #
        # Allocate memory for the wigner function and the wave function
        #
        ##########################################################################################

        # Allocate the Wigner function
        #self.wignerfunction = gpuarray.zeros((self.X.size, self.X.size), np.complex128)

        # Allocate the density matrix
        self.wavefunction = gpuarray.zeros((1, self.X.size), np.complex128)

        ##########################################################################################
        #
        # Set-up CUDA FFT
        #
        ##########################################################################################

        #self.plan_Z2Z_ax0 = cufft.Plan_Z2Z_2D_Axis0(self.wignerfunction.shape)
        #self.plan_Z2Z_ax1 = cufft.Plan_Z2Z_2D_Axis1(self.wignerfunction.shape)

        self.plan_Z2Z_psi = cufft.Plan_Z2Z_2D_Axis1(self.wavefunction.shape)

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

    def set_wavefunction(self, new_wave_func):
        """
        Set the initial wave function
        :param new_wave_func: 1D numpy array, 1D GPU array contaning the wave function,
                    a string of the C code specifying the initial condition,
                    a python function of the form F(self, X), or a float number
                    Coordinate (X) is declared.
        :return: self
        """
        if isinstance(new_wave_func, (np.ndarray, gpuarray.GPUArray)):
            # perform the consistency checks
            assert new_wave_func.shape == self.wavefunction.shape, \
                "The grid sizes does not match with the wave function"

            # copy wigner function
            self.wavefunction[:] = new_wave_func.astype(np.complex128)

        elif isinstance(new_wave_func, FunctionType):
            # user supplied the function which will return the wave function
            self.wavefunction[:] = new_wave_func(self, self.X)

        elif isinstance(new_wave_func, str):
            # user specified C code
            print("\n================================ Compiling init_wavefunc ================================\n")
            init_wigner = SourceModule(
                self.init_wavefunction_cuda_source.format(cuda_consts=self.cuda_consts, new_wave_func=new_wave_func),
            ).get_function("Kernel")
            init_wigner(self.wavefunction, **self.wavefunction_mapper_params)

        elif isinstance(new_wave_func, (float, complex)):
            # user specified a constant
            self.wavefunction.fill(np.complex128(new_wave_func))
        else:
            raise NotImplementedError("new_wave_func must be either function or numpy.array")

        # normalize
        self.wavefunction /= cu_linalg.norm(self.wavefunction) * np.sqrt(self.dX)

        return self

    def single_step_propagation(self):
        """
        Perform a single step propagation. The final wave function is not normalized.
        :return: self.wavefunction
        """
        self.expV(self.wavefunction, self.t, **self.wavefunction_mapper_params)

        cufft.fft_Z2Z(self.wavefunction, self.wavefunction, self.plan_Z2Z_psi)
        self.expK(self.wavefunction, self.t, **self.wavefunction_mapper_params)
        cufft.ifft_Z2Z(self.wavefunction, self.wavefunction, self.plan_Z2Z_psi)

        self.expV(self.wavefunction, self.t, **self.wavefunction_mapper_params)

        return self.wavefunction

    def propagate(self, steps=1):
        """
        Time propagate the density matrix saved in self.rho
        :param steps: number of self.dt time increments to make
        :return: self
        """
        for _ in xrange(steps):
            # increment current time
            self.t += self.dt

            # advance by one time step
            self.single_step_propagation()

            # calculate the Ehrenfest theorems
            self.get_Ehrenfest(self.t)

            # normalize
            self.wavefunction /= cu_linalg.norm(self.wavefunction) * np.sqrt(self.dX)

        return self

    def get_average(self, observable):
        """
        Return the expectation value of the observable with respect to the current wave function
        :param observable: (str)
        :return: float
        """
        # Compile the corresponding cuda functions, if it has not been done
        try:
            func = self._compiled_observable[observable]
        except KeyError:
            print("\n============================== Compiling [%s] ==============================\n" % observable)
            func = self._compiled_observable[observable] = SourceModule(
                self.weighted_func_cuda_code.format(cuda_consts=self.cuda_consts, func=observable),
            ).get_function("Kernel")

        # Execute underlying function
        func(self.wavefunction, self.weighted, self.t, **self.wavefunction_mapper_params)

        return gpuarray.sum(self.weighted).get()

    def get_Ehrenfest(self, t):
        """
        Calculate observables entering the Ehrenfest theorems at time
        :param t: current time
        :return: coordinate and momentum densities, if the Ehrenfest theorems were calculated; otherwise, return None
        """
        if self.isEhrenfest:

            #########################################################################
            #
            #   Working in the coordinate representation
            #
            #########################################################################

            # Normalize the wave function as required in self.weighted_func_cuda_code
            self.wavefunction /= cu_linalg.norm(self.wavefunction)

            # save the current value of <X>
            self.X_average.append(self.get_average("X"))

            # save the current value of <-diff_V>
            self.P_average_RHS.append(-self.get_average(self.diff_V))

            # save the potential energy
            self.hamiltonian_average.append(self.get_average(self.V))

            #########################################################################
            #
            #   Working in the momentum representation
            #
            #########################################################################

            cufft.fft_Z2Z(self.wavefunction, self.wavefunction, self.plan_Z2Z_psi)

            # Normalize the wave function as required in self.weighted_func_cuda_code
            self.wavefunction /= cu_linalg.norm(self.wavefunction)

            # save the current value of <diff_K>
            self.X_average_RHS.append(self.get_average(self.diff_K))

            # save the current value of <P>
            self.P_average.append(self.get_average("P"))

            # add the expectation value for the kinetic energy
            self.hamiltonian_average[-1] += self.get_average(self.K)

            # going back to the coordinate representation
            cufft.ifft_Z2Z(self.wavefunction, self.wavefunction, self.plan_Z2Z_psi)

    def get_energy(self):
        """
        Calculate the expectation value of the Hamiltonian
        :return:
        """
        # Normalize the wave function as required in self.weighted_func_cuda_code
        self.wavefunction /= cu_linalg.norm(self.wavefunction)

        av_V = self.get_average(self.V)

        # go to the momentum representation
        cufft.fft_Z2Z(self.wavefunction, self.wavefunction, self.plan_Z2Z_psi)

        # Normalize the wave function as required in self.weighted_func_cuda_code
        self.wavefunction /= cu_linalg.norm(self.wavefunction)

        av_K = self.get_average(self.K)

        # go back to the coordinate representation
        cufft.ifft_Z2Z(self.wavefunction, self.wavefunction, self.plan_Z2Z_psi)

        return av_V + av_K

    def set_ground_state(self, nsteps=10000):
        """
        Obtain the ground state wave function by the imaginary time propagation
        :param nsteps: number of the imaginary time steps to take
        :return: self
        """
        self.set_wavefunction(1.)

        for _ in xrange(nsteps):
            self.bloch_expV(self.wavefunction, **self.wavefunction_mapper_params)

            cufft.fft_Z2Z(self.wavefunction, self.wavefunction, self.plan_Z2Z_psi)
            self.bloch_expK(self.wavefunction, **self.wavefunction_mapper_params)
            cufft.ifft_Z2Z(self.wavefunction, self.wavefunction, self.plan_Z2Z_psi)

            self.bloch_expV(self.wavefunction, **self.wavefunction_mapper_params)

            self.wavefunction /= cu_linalg.norm(self.wavefunction) * np.sqrt(self.dX)

        return self

    def projectout_stationary_states(self, wavefunction):
        """
        Project out the stationary states from wavefunction
        :param wavefunction: provided wavefunction
        :return: wavefunction
        """
        ##########################################################################################
        #
        #   Making sure that all the functions are initialized
        #
        ##########################################################################################
        try:
            self.vdot
            self.projectout
        except AttributeError:

            wavefunction_type = dict(wave_type = dtype_to_ctype(self.wavefunction.dtype))

            # set the vdot function (scalar product with complex conjugation)
            self.vdot = ReductionKernel(
                self.wavefunction.dtype,
                neutral="0.".format(**wavefunction_type),
                reduce_expr="a + b",
                map_expr="conj(bra[i]) * ket[i]",
                arguments="const {wave_type} *bra, const {wave_type} *ket".format(**wavefunction_type)
            )

            self.projectout = ElementwiseKernel(
                "{wave_type} *psi, const {wave_type} *phi, const {wave_type} C".format(**wavefunction_type),
                "psi[i] -= C * phi[i]"
            )

        # normalize
        wavefunction /= cu_linalg.norm(wavefunction)

        # calculate all projections
        projs = [self.vdot(psi, wavefunction).get() for psi in self.stationary_states]

        # project out the stationary states
        for psi, proj in zip(self.stationary_states, projs):
            self.projectout(wavefunction, psi, proj)

        # normalize
        wavefunction /= cu_linalg.norm(wavefunction)

        return wavefunction

    def get_stationary_states(self, nstates, nsteps=10000):
        """
        Obtain stationary states via the imaginary time propagation
        :param nstates: number of states to obtaine.
                if nstates = 1, only the ground state is obtained. if  nstates = 2,
                the ground and first exited states are obtained.
        :param nsteps: number of the imaginary time steps to take
        :return:self
        """
        # initialize the list where the stationary states will be saved
        self.stationary_states = []

        even = True

        for n in xrange(nstates):

            # initialize the wavefunction
            if even:
                self.set_wavefunction(1.)
            else:
                self.wavefunction[:] = self.X.reshape(self.wavefunction.shape).astype(self.wavefunction.dtype)

            even = not even

            for _ in xrange(nsteps):
                self.bloch_expV(self.wavefunction, **self.wavefunction_mapper_params)

                cufft.fft_Z2Z(self.wavefunction, self.wavefunction, self.plan_Z2Z_psi)
                self.bloch_expK(self.wavefunction, **self.wavefunction_mapper_params)
                cufft.ifft_Z2Z(self.wavefunction, self.wavefunction, self.plan_Z2Z_psi)

                self.bloch_expV(self.wavefunction, **self.wavefunction_mapper_params)

                # project out all previous stationary states
                self.projectout_stationary_states(self.wavefunction)

            # save obtained approximation to the stationary state
            self.stationary_states.append(self.wavefunction.copy())

        return self

    def get_wigner(self):
        """
        Transform the density matrix saved in self.rho into the unormalized Wigner function
        :return: self.wignerfunction
        """
        # Create the density matrix out of the wavefunction
        self.psi2rho(self.wavefunction, self.wignerfunction, **self.wigner_mapper_params)

        # Step 1: Rotate by +45 degrees
        # Shear X
        cufft.fft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_Z2Z_ax1)
        self.phase_shearX(self.wignerfunction, **self.wigner_mapper_params)
        cufft.ifft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_Z2Z_ax1)

        # Shear Y
        cufft.fft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_Z2Z_ax0)
        self.phase_shearY(self.wignerfunction, **self.wigner_mapper_params)
        cufft.ifft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_Z2Z_ax0)

        # Shear X
        cufft.fft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_Z2Z_ax1)
        self.phase_shearX(self.wignerfunction, **self.wigner_mapper_params)
        cufft.ifft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_Z2Z_ax1)

        # Step 2: FFt the Blokhintsev function
        self.sign_flip(self.wignerfunction, **self.wigner_mapper_params)
        cufft.ifft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_Z2Z_ax0)
        self.sign_flip(self.wignerfunction, **self.wigner_mapper_params)

        # normalize
        self.wignerfunction /= gpuarray.sum(self.wignerfunction).get().real * self.wigner_dXdP

        return self.wignerfunction

    def get_wigner_blackman(self):
        """
        Transform the density matrix saved in self.rho into the unormalized Wigner function
        :return: self.wignerfunction
        """
        # Create the density matrix out of the wavefunction
        self.psi2rho(self.wavefunction, self.wignerfunction, **self.wigner_mapper_params)

        # Step 1: Rotate by +45 degrees
        # Shear X
        self.blackmanX(self.wignerfunction, **self.wigner_mapper_params)
        cufft.fft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_Z2Z_ax1)
        self.phase_shearX(self.wignerfunction, **self.wigner_mapper_params)
        cufft.ifft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_Z2Z_ax1)

        # Shear Y
        self.blackmanY(self.wignerfunction, **self.wigner_mapper_params)
        cufft.fft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_Z2Z_ax0)
        self.phase_shearY(self.wignerfunction, **self.wigner_mapper_params)
        cufft.ifft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_Z2Z_ax0)

        # Shear X
        self.blackmanX(self.wignerfunction, **self.wigner_mapper_params)
        cufft.fft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_Z2Z_ax1)
        self.phase_shearX(self.wignerfunction, **self.wigner_mapper_params)
        cufft.ifft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_Z2Z_ax1)

        # Step 2: FFt the Blokhintsev function
        self.blackmanY(self.wignerfunction, **self.wigner_mapper_params)
        self.sign_flip(self.wignerfunction, **self.wigner_mapper_params)
        cufft.ifft_Z2Z(self.wignerfunction, self.wignerfunction, self.plan_Z2Z_ax0)
        self.sign_flip(self.wignerfunction, **self.wigner_mapper_params)

        # normalize
        self.wignerfunction /= gpuarray.sum(self.wignerfunction).get().real * self.wigner_dXdP

        return self.wignerfunction

    schrodinger_wigner_cuda_source = """
    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES

    typedef pycuda::complex<double> cuda_complex;

    {cuda_consts}

    ////////////////////////////////////////////////////////////////////////////
    //
    //  Wave function to density matrix
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void psi2rho(const cuda_complex* wavefunction, cuda_complex* rho)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        rho[indexTotal] = wavefunction[j] * conj(wavefunction[i]);
    }}

    ////////////////////////////////////////////////////////////////////////////
    //
    // CUDA code to apply phase factors to perform X and Y shearing
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void phase_shearX(cuda_complex *rho)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double P = dP * ((j + X_gridDIM / 2) % X_gridDIM - 0.5 * X_gridDIM);
        const double X_prime = dX * (i - 0.5 * X_gridDIM);

        // perform rotation by theta: const double a = tan(0.5 * theta);
        const double a = tan(M_PI / 8.);
        const double phase = -a * P * X_prime;

        rho[indexTotal] *= cuda_complex(cos(phase), sin(phase)) / double(X_gridDIM);

    }}

    __global__ void phase_shearY(cuda_complex *rho)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double P_prime = dP * ((i + X_gridDIM / 2) % X_gridDIM - 0.5 * X_gridDIM);

        // perform rotation by theta: const double b = -sin(theta);
        const double b = -sin(M_PI / 4.);
        const double phase = -b * P_prime * X;

        rho[indexTotal] *= cuda_complex(cos(phase), sin(phase)) / double(X_gridDIM);
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
        const size_t indexTotal = j + i * X_gridDIM;

        const double Cj = j * 2.0 * M_PI / (X_gridDIM - 1.);

        rho[indexTotal] *= 0.42 - 0.5 * cos(Cj) + 0.08 * cos(2.0 * Cj);
    }}

    __global__ void blackmanY(cuda_complex *rho)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double Ci = i * 2.0 * M_PI / (X_gridDIM - 1.);

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
        const size_t indexTotal = j + i * X_gridDIM;

        // rho *= pow(-1, i)
        rho[indexTotal] *= 1 - 2 * int(i % 2);
    }}

    ////////////////////////////////////////////////////////////////////////////
    //
    //  Real time propagator
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

    ////////////////////////////////////////////////////////////////////////////
    //
    // CUDA code to define the action of the kinetic energy exponent
    // onto the wavefunction
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void expK(cuda_complex *wavefunction, double t)
    {{
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;

        // fft shifting momentum
        const double P = dP * ((j + X_gridDIM / 2) % X_gridDIM - 0.5 * X_gridDIM);

        const double phase = -dt * K(P, t);

        wavefunction[j] *= cuda_complex(cos(phase), sin(phase)) * ({abs_boundary_p});
    }}

    ////////////////////////////////////////////////////////////////////////////
    //
    // CUDA code to define the action of the potential energy exponent
    // onto the wavefunction
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void expV(cuda_complex *wavefunction, double t)
    {{
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;

        const double X = dX * (j - 0.5 * X_gridDIM);

        const double phase = -0.5 * dt * V(X, t);

        wavefunction[j] *= cuda_complex(cos(phase), sin(phase)) * ({abs_boundary_x});
    }}

    ////////////////////////////////////////////////////////////////////////////
    //
    //  Imaginary time propagator, e.g., to find the ground state
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void bloch_expK(cuda_complex *wavefunction)
    {{
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;

        const double P = dP * ((j + X_gridDIM / 2) % X_gridDIM - 0.5 * X_gridDIM);

        const double phase = -dt * K(P, t_initial);

        wavefunction[j] *= exp(phase);
    }}

    __global__ void bloch_expV(cuda_complex *wavefunction)
    {{
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;

        const double X = dX * (j - 0.5 * X_gridDIM);

        const double phase = -0.5 * dt * V(X, t_initial);

        wavefunction[j] *= exp(phase);
    }}
    """

    init_wavefunction_cuda_source = """
    // CUDA code to initialize the wave function
    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES

    typedef pycuda::complex<double> cuda_complex;

    {cuda_consts}

    __global__ void Kernel(cuda_complex *wavefunction)
    {{
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;

        const double X = dX * (j - 0.5 * X_gridDIM);

        wavefunction[j] = ({new_wave_func});
    }}
    """

    weighted_func_cuda_code = """
    // CUDA code to calculate
    //      weighted = abs(psi)**2 * func.
    // This is used in self.get_average
    // weighted.sum() is the average of func over the wave function
    //
    // Note that the normalization condition of psi is np.sum(np.abs(psi)**2) = 1

    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES

    typedef pycuda::complex<double> cuda_complex;

    {cuda_consts}

    __global__ void Kernel(const cuda_complex *wavefunction, double *weighted, double t)
    {{
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double P = dP * ((j + X_gridDIM / 2) % X_gridDIM - 0.5 * X_gridDIM);

        weighted[j] = pow(abs(wavefunction[j]), 2) * ({func});
    }}
    """

##########################################################################################
#
# Example
#
##########################################################################################

if __name__ == '__main__':

    print(SchrodingerWignerCUDA1D.__doc__)

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

            ax = fig.add_subplot(111)

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
                cmap='seismic',
                norm=WignerNormalize(vmin=-0.01, vmax=0.1)
            )

            self.fig.colorbar(self.img)

            ax.set_xlabel('$x$ (a.u.)')
            ax.set_ylabel('$p$ (a.u.)')

            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])

        def set_quantum_sys(self):
            """
            Initialize quantum propagator
            :param self:
            :return:
            """
            # Create propagator
            self.quant_sys = SchrodingerWignerCUDA1D(
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
                V="0.5 * omega_square * X * X",

                # these functions are used for evaluating the Ehrenfest theorems
                diff_K="P",
                diff_V="omega_square * X"
            )

            # set randomised initial condition
            self.quant_sys.set_wavefunction(
                "exp(-sigma * pow(X - x0, 2) + cuda_complex(0., p0) * X)"
            )

        def empty_frame(self):
            """
            Make empty frame and reinitialize quantum system
            :param self:
            :return: image object
            """
            self.img.set_array([[]])
            return self.img,

        def __call__(self, frame_num):
            """
            Draw a new frame
            :param frame_num: current frame number
            :return: image objects
            """
            # propagate the wave function and then get the Wigner function
            self.img.set_array(
                self.quant_sys.propagate(50).get_wigner().get().real
            )

            return self.img,


    fig = plt.gcf()
    visualizer = VisualizeDynamicsPhaseSpace(fig)
    animation = matplotlib.animation.FuncAnimation(
        fig, visualizer, frames=np.arange(100), init_func=visualizer.empty_frame, repeat=True, blit=True
    )

    plt.show()

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

