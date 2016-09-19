import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from fractions import gcd
from types import MethodType, FunctionType
import cufft


class WignerMoyalCUDA1D:
    """
    The second-order split-operator propagator for the Moyal equation for the Wigner function W(x, p, t)
    with the time-dependent Hamiltonian H = K(p, t) + V(x, t) using CUDA.

    This implementation stores the Wigner function as a 2D real gpu array.
    """
    def __init__(self, **kwargs):
        """
        The following parameters are to be specified
            X_gridDIM - the coordinate grid size
            X_amplitude - maximum value of the coordinates
            P_gridDIM - the momentum grid size
            P_amplitude - maximum value of the momentum
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

        assert 2**int(np.log2(self.X_gridDIM)) == self.X_gridDIM, \
            "Coordinate grid size (X_gridDIM) must be a power of two"

        try:
            self.P_gridDIM
        except AttributeError:
            raise AttributeError("Momentum grid size (P_gridDIM) was not specified")

        assert 2**int(np.log2(self.P_gridDIM)) == self.P_gridDIM, \
            "Momentum grid size (P_gridDIM) must be a power of two"

        try:
            self.X_amplitude
        except AttributeError:
            raise AttributeError("Coordinate grid range (X_amplitude) was not specified")

        try:
            self.P_amplitude
        except AttributeError:
            raise AttributeError("Momentum grid range (P_amplitude) was not specified")

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

        self.t = np.float64(self.t)

        ##########################################################################################
        #
        # Generating grids
        #
        ##########################################################################################

        # get coordinate and momentum step sizes
        self.dX = 2. * self.X_amplitude / self.X_gridDIM
        self.dP = 2. * self.P_amplitude / self.P_gridDIM

        self.dXdP = self.dX * self.dP

        # coordinate grid
        self.X = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX, self.X_gridDIM)

        # momentum grid
        self.P = np.linspace(-self.P_amplitude, self.P_amplitude - self.dP, self.P_gridDIM)

        #
        self.PP_prime = np.sqrt(2.) * self.P
        self.dPP_prime = self.PP_prime[1] - self.PP_prime[0]

        self.XX_prime = np.fft.fftshift(np.fft.fftfreq(self.PP_prime.size, self.dPP_prime / (2. * np.pi)))
        self.dXX_prime = self.XX_prime[1] - self.XX_prime[0]

        self.XX = np.sqrt(2.) * self.X
        self.dXX = self.XX[1] - self.XX[0]

        self.PP = np.fft.fftfreq(self.XX.size, self.dXX / (2. * np.pi))
        self.dPP = self.PP[1] - self.PP[0]

        self.P = self.P[:, np.newaxis]
        self.X = self.X[np.newaxis, :]

        self.XX = self.XX[np.newaxis, :]
        self.PP = self.PP[np.newaxis, :]

        self.XX_prime = self.XX_prime[:, np.newaxis ]
        self.PP_prime = self.PP_prime[:, np.newaxis]

        ##########################################################################################
        #
        # Save CUDA constants
        #
        ##########################################################################################

        kwargs.update(
            dX=self.dX,
            dP=self.dP,

            dXX=self.dXX,
            dXX_prime=self.dXX_prime,

            dPP=self.dPP,
            dPP_prime=self.dPP_prime,

            # These constants are useful for constructing the absorbing boundaries
            XX_max=self.XX.max(),
            XX_prime_max=self.XX_prime.max(),

            PP_max=self.PP.max(),
            PP_prime_max=self.PP_prime.max()
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

        # CUDA block and grid for functions that act on the whole Wigner function
        self.wigner_mapper_params = dict(
            block=(nproc, 1, 1),
            grid=(size_x // nproc, self.P_gridDIM)
        )

        self.rho_mapper_params = self.wigner_mapper_params

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

        ##########################################################################################
        #
        # Allocate memory for the wigner function in the theta x and p lambda representations
        # by reusing the memory
        #
        ##########################################################################################

        # Allocate the Wigner function in the X and P representation
        self.wignerfunction = gpuarray.zeros((self.P.size, self.X.size), np.complex128)

        # Allocate the density matrix
        self.rho = gpuarray.zeros_like(self.wignerfunction)

        ##########################################################################################
        #
        # Set-up CUDA FFT
        #
        ##########################################################################################

        self.plan_Z2Z_ax0 = cufft.Plan_Z2Z_2D_Axis0(self.rho.shape)
        self.plan_Z2Z_ax1 = cufft.Plan_Z2Z_2D_Axis1(self.rho.shape)

        ##########################################################################################
        #
        #   Initialize facility for calculating expectation values of the cuurent wigner function
        #   see the implementation of self.get_average
        #
        ##########################################################################################

        # This array is used for expectation value calculation
        self.weighted = gpuarray.zeros(self.wignerfunction.shape, np.float64)

        # hash table of cuda compiled functions that calculate an average of specified observable
        self._compiled_observable = dict()

        ##########################################################################################
        #
        #   Ehrenfest theorems (optional)
        #
        ##########################################################################################

        self.hamiltonian = self.K + ' + ' + self.V

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

    def get_average(self, observable):
        """
        Return the expectation value of the observable with respcte to the current wigner function
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
        func(self.wignerfunction, self.weighted, self.t, **self.wigner_mapper_params)

        return gpuarray.sum(self.weighted).get() * self.dXdP

    def get_purity(self):
        """
        Return the purity of the current Wigner function, 2*np.pi*np.sum(W**2)*dXdP
        :return: float
        """
        return 2. * np.pi * gpuarray.dot(self.wignerfunction, self.wignerfunction).get().real * self.dXdP

    def get_sigma_x_sigma_p(self):
        """
        Return the product of standart deviation of coordinate and momentum,
        the LHS of the Heisenberg uncertainty principle:
            sigma_p * sigma_p >= 0.5
        :return: float
        """
        return np.sqrt(
            (self.get_average("X * X") - self.get_average("X") ** 2)
            * (self.get_average("P * P") - self.get_average("P") ** 2)
        )

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

    def set_wignerfunction(self, new_wigner_func):
        """
        Set the initial Wigner function
        :param new_wigner_func: 2D numpy array, 2D GPU array contaning the wigner function,
                    a string of the C code specifying the initial condition,
                    a python function of the form F(self, x, p), or a float number
                    Coordinate (X) and momentum (P) variables are declared.
        :return: self
        """
        if isinstance(new_wigner_func, (np.ndarray, gpuarray.GPUArray)):
            # perform the consistency checks
            assert new_wigner_func.shape == self.wignerfunction.shape, \
                "The grid sizes does not match with the Wigner function"

            # copy wigner function
            self.wignerfunction[:] = new_wigner_func.astype(np.complex128)

        elif isinstance(new_wigner_func, FunctionType):
            # user supplied the function which will return the Wigner function
            self.wignerfunction[:] = new_wigner_func(self, self.X, self.P)

        elif isinstance(new_wigner_func, str):
            # user specified C code
            print("\n================================ Compiling init_wigner ================================\n")
            init_wigner = SourceModule(
                self.init_wigner_cuda_source.format(cuda_consts=self.cuda_consts, new_wigner_func=new_wigner_func),
            ).get_function("Kernel")
            init_wigner(self.wignerfunction, **self.wigner_mapper_params)

        elif isinstance(new_wigner_func, (float, complex)):
            # user specified a constant
            self.wignerfunction.fill(np.complex128(new_wigner_func))
        else:
            raise NotImplementedError("new_wigner_func must be either function or numpy.array")

        # normalize
        self.wignerfunction /= gpuarray.sum(self.wignerfunction).get() * self.dXdP

        # Find the underlying density matrix
        self.wigner2rho()

        return self

    def wigner2rho(self):
        """
        Transform the Wigner function saved in self.wignerfunction into the density matrix
        :return: self.rho
        """
        # Make a copy of the wogner function
        gpuarray._memcpy_discontig(self.rho, self.wignerfunction)

        ####################################################################################
        #
        # Step 1: Perform the FFT over the Wigner function
        # using method from
        #   D. H. Bailey and P. N. Swarztrauber, SIAM J. Sci. Comput. 15, 1105 (1994)
        #   (http://epubs.siam.org/doi/abs/10.1137/0915067)
        #
        ####################################################################################
        # """
        self.sign_flip(self.rho, **self.rho_mapper_params)
        cufft.fft_Z2Z(self.rho, self.rho, self.plan_Z2Z_ax0)
        self.sign_flip(self.rho, **self.rho_mapper_params)

        ####################################################################################
        #
        # Step 2: Perform the -45 degrees rotation of the density matrix
        # using method from
        #   K. G. Larkin,  M. A. Oldfield, H. Klemm, Optics Communications, 139, 99 (1997)
        #   (http://www.sciencedirect.com/science/article/pii/S0030401897000977)
        #
        ####################################################################################

        # Shear X
        cufft.ifft_Z2Z(self.rho, self.rho, self.plan_Z2Z_ax1)
        self.phase_shearX(self.rho, **self.rho_mapper_params)
        cufft.fft_Z2Z(self.rho, self.rho, self.plan_Z2Z_ax1)

        # Shear Y
        cufft.ifft_Z2Z(self.rho, self.rho, self.plan_Z2Z_ax0)
        self.phase_shearY(self.rho, **self.rho_mapper_params)
        cufft.fft_Z2Z(self.rho, self.rho, self.plan_Z2Z_ax0)

        # Shear X
        cufft.ifft_Z2Z(self.rho, self.rho, self.plan_Z2Z_ax1)
        self.phase_shearX(self.rho, **self.rho_mapper_params)
        cufft.fft_Z2Z(self.rho, self.rho, self.plan_Z2Z_ax1)

        return self.rho

    def rho2wigner_(self):
        """
        Transform the density matrix saved in self.rho into the unormalized Wigner function
        :return: self.wignerfunction
        """
        # Make a copy of the density matrix
        gpuarray._memcpy_discontig(self.wignerfunction, self.rho)

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

        return self.wignerfunction

    def rho2wigner_blackman(self):
        """
        Transform the density matrix saved in self.rho into the unormalized Wigner function
        :return: self.wignerfunction
        """
        # Make a copy of the density matrix
        gpuarray._memcpy_discontig(self.wignerfunction, self.rho)

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

        return self.wignerfunction

    # Which Wigner transform to use
    rho2wigner = rho2wigner_

    def single_step_propagation(self):
        """
        Perform a single step propagation. The final density matrix function is not normalized.
        :return: self.rho
        """
        self.expV(self.rho, self.t, **self.rho_mapper_params)

        cufft.fft_Z2Z(self.rho, self.rho, self.plan_Z2Z_ax0)
        cufft.fft_Z2Z(self.rho, self.rho, self.plan_Z2Z_ax1)

        self.expK(self.rho, self.t, **self.rho_mapper_params)

        cufft.ifft_Z2Z(self.rho, self.rho, self.plan_Z2Z_ax0)
        cufft.ifft_Z2Z(self.rho, self.rho, self.plan_Z2Z_ax1)
        self.rho /= self.rho.shape[0] * self.rho.shape[1]

        self.expV(self.rho, self.t, **self.rho_mapper_params)

        return self.rho

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

            # calculate the Ehrenfest theorems
            self.get_Ehrenfest(self.t)

        if not self.isEhrenfest:
            self.normalize_rho_wigner()

        return self.wignerfunction

    def get_Ehrenfest(self, t):
        """
        Calculate observables entering the Ehrenfest theorems at time
        :param t: current time
        :return: coordinate and momentum densities, if the Ehrenfest theorems were calculated; otherwise, return None
        """
        if self.isEhrenfest:
            self.normalize_rho_wigner()

            # save the current value of <X>
            self.X_average.append(self.get_average("X"))

            # save the current value of <diff_K>
            self.X_average_RHS.append(self.get_average(self.diff_K))

            # save the current value of <P>
            self.P_average.append(self.get_average("P"))

            # save the current value of <-diff_V>
            self.P_average_RHS.append(-self.get_average(self.diff_V))

            # save the current expectation value of energy
            self.hamiltonian_average.append(self.get_average(self.hamiltonian))

    def normalize_rho_wigner(self):
        """
        Perform the Wigner transform and then normalize the wigner function and density matrix
        :return: self
        """
        # Recover the Wigner function
        self.rho2wigner()

        # Normalize the Wigner function and density matrix
        norm = gpuarray.sum(self.wignerfunction).get().real * self.dXdP

        self.wignerfunction /= norm
        self.rho /= norm

        return self

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
        const size_t indexTotal = j + i * X_gridDIM;

        const double PP = dPP * ((j + X_gridDIM / 2) % X_gridDIM - 0.5 * X_gridDIM);
        const double XX_prime = dXX_prime * (i - 0.5 * P_gridDIM);

        // perform rotation by theta: const double a = tan(0.5 * theta);
        const double a = tan(M_PI / 8.);
        const double phase = -a * PP * XX_prime;

        rho[indexTotal] *= cuda_complex(cos(phase), sin(phase)) / double(X_gridDIM);

    }}

    __global__ void phase_shearY(cuda_complex *rho)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double XX = dXX * (j - 0.5 * X_gridDIM);
        const double PP_prime = dPP_prime * ((i + P_gridDIM / 2) % P_gridDIM - 0.5 * P_gridDIM);

        // perform rotation by theta: const double b = -sin(theta);
        const double b = -sin(M_PI / 4.);
        const double phase = -b * PP_prime * XX;

        rho[indexTotal] *= cuda_complex(cos(phase), sin(phase)) / double(P_gridDIM);
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

        const double Ci = i * 2.0 * M_PI / (P_gridDIM - 1.);

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

    ////////////////////////////////////////////////////////////////////////////
    //
    // CUDA code to define the action of the kinetic energy exponent
    // onto the density matrix in the momentum representation < PP | rho | PP_prime >
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void expK(cuda_complex *rho, double t)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        // fft shifting momentum
        const double PP = dPP * ((j + X_gridDIM / 2) % X_gridDIM - 0.5 * X_gridDIM);
        const double PP_prime = dPP_prime * ((i + P_gridDIM / 2) % P_gridDIM - 0.5 * P_gridDIM);

        const double phase = -dt * (K(PP, t) - K(PP_prime, t));

        rho[indexTotal] *= cuda_complex(cos(phase), sin(phase)) * ({abs_boundary_p});
    }}

    ////////////////////////////////////////////////////////////////////////////
    //
    // CUDA code to define the action of the potential energy exponent
    // onto the density matrix in the coordinate representation < XX | rho | XX_prime >
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void expV(cuda_complex *rho, double t)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double XX = dXX * (j - 0.5 * X_gridDIM);
        const double XX_prime = dXX_prime * (i - 0.5 * P_gridDIM);

        const double phase = -0.5 * dt * (V(XX, t + 0.5 * dt) - V(XX_prime, t + 0.5 * dt));

        rho[indexTotal] *= cuda_complex(cos(phase), sin(phase)) * ({abs_boundary_x});
    }}
    """

    init_wigner_cuda_source = """
    // CUDA code to initialize the wigner function
    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES

    typedef pycuda::complex<double> cuda_complex;

    {cuda_consts}

    __global__ void Kernel(cuda_complex *W)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double P = dP * (i - 0.5 * P_gridDIM);

        W[indexTotal] = ({new_wigner_func});
    }}
    """

    weighted_func_cuda_code = """
    // CUDA code to calculate
    //      weighted = W(X, P, t) * func(X, P, t).
    // This is used in self.get_average
    // weighted.sum()*dX*dP is the average of func(X, P, t) over the Wigner function
    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES

    typedef pycuda::complex<double> cuda_complex;

    {cuda_consts}

    __global__ void Kernel(const cuda_complex *W, double *weighted, double t)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double P = dP * (i - 0.5 * P_gridDIM);

        weighted[indexTotal] = W[indexTotal].real() * ({func});
    }}
    """

##########################################################################################
#
# Example
#
##########################################################################################

if __name__ == '__main__':

    print(WignerMoyalCUDA1D.__doc__)

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
            extent = [self.quant_sys.X.min(), self.quant_sys.X.max(), self.quant_sys.P.min(), self.quant_sys.P.max()]

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

        def set_quantum_sys(self):
            """
            Initialize quantum propagator
            :param self:
            :return:
            """
            # Create propagator
            self.quant_sys = WignerMoyalCUDA1D(
                t=0.,
                dt=0.01,

                X_gridDIM=1024,
                X_amplitude=10.,

                P_gridDIM=512,
                P_amplitude=10.,

                # randomized parameter
                omega_square=np.random.uniform(2., 6.),

                # randomized parameters for initial condition
                sigma=np.random.uniform(0.5, 4.),
                p0=np.random.uniform(-1., 1.),
                x0=np.random.uniform(-1., 1.),

                # smoothing parameter for absorbing boundary
                #alpha=0.01,

                # kinetic energy part of the hamiltonian
                K="0.5 * P * P",

                # potential energy part of the hamiltonian
                V="0.5 * omega_square * X * X",

                # these functions are used for evaluating the Ehrenfest theorems
                diff_K="P",
                diff_V="omega_square * X"
            )

            # set randomised initial condition
            self.quant_sys.set_wignerfunction(
                "exp(-sigma * pow(X - x0, 2) - (1.0 / sigma) * pow(P - p0, 2))"
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
            # propagate the wigner function
            self.img.set_array(self.quant_sys.propagate(50).get().real)

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
