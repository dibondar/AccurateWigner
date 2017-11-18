import numpy as np
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix
from types import MethodType, FunctionType # this is used to dynamically add method to class


class MUBQBandStructure:
    """
    Calculate the band structure for quantum Hamiltonian, H(x,p) = K(p) + V(x),
    using mutually unbiased bases (MUB).
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - specifying the grid size
            X_amplitude - maximum value of the coordinates
            V(x) - potential energy (as a function)
            K(p) - momentum dependent part of the hamiltonian (as a function)
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
            raise AttributeError("Grid size (X_gridDIM) was not specified")

        try:
            self.X_amplitude
        except AttributeError:
            raise AttributeError("Coordinate range (X_amplitude) was not specified")

        try:
            self.V
        except AttributeError:
            raise AttributeError("Potential energy (V) was not specified")

        try:
            self.K
        except AttributeError:
            raise AttributeError("Momentum dependence (K) was not specified")

        # get coordinate step size
        self.dX = 2.*self.X_amplitude / self.X_gridDIM

        # generate coordinate range
        self.X_range = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX , self.X_gridDIM)

        r = np.arange(self.X_gridDIM)
        self.sign_flip = (-1)**(r[:, np.newaxis] + r[np.newaxis, :])

        # generate momentum range as it corresponds to FFT frequencies
        self.P_range = np.arange(self.X_gridDIM, dtype=np.float) - self.X_gridDIM / 2
        self.P_range *= 2. * np.pi / (self.X_gridDIM * self.dX)

    def get_hamiltonian(self, k):
        """
        Return the hamiltonian matrix in the coordinate representation corresponding to the bloch vector (k)
        :param k: (float) block vector
        :return: 2D numpy.array
        """
        # Construct the momentum dependent part
        hamiltonian = np.diag(self.K(self.P_range + k))
        hamiltonian *= self.sign_flip
        hamiltonian = fftpack.ifft(hamiltonian, axis=0, overwrite_x=True)
        hamiltonian = fftpack.fft(hamiltonian, axis=1, overwrite_x=True)
        hamiltonian *= self.sign_flip

        # Add diagonal potential energy
        hamiltonian += np.diag(self.V(self.X_range))

        return hamiltonian

    def get_band_structure(self, k, n):
        """
        Calculate band structure of specified quantum system
        :param k: (numpy.array) range of k vectors for which the band structure are to be calculated
        :param n: (int) how many bands to calculate
        :return: 2d numpy.array with size = (n, len(k))
        """
        bands = np.array([
            linalg.eigvalsh(self.get_hamiltonian(_))[:n] for _ in k
        ])
        return bands.T

##########################################################################################
#
# Example
#
##########################################################################################

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    print(MUBQBandStructure.__doc__)

    sys_params = dict(
        X_gridDIM=256,

        # the lattice constant is 2 * X_amplitude
        X_amplitude=4.,

        # Lattice height
        V0=0.37,

        # the kinetic energy
        K=lambda self, p: 0.5 * p ** 2,

        # Mathieu-type periodic system
        V=lambda self, x: -self.V0 * (1 + np.cos(np.pi * (x + self.X_amplitude) / self.X_amplitude))
    )

    # initialize the system
    qsys = MUBQBandStructure(**sys_params)

    # how many eV is in 1 a.u. of energy
    au2eV = 27.

    # range of bloch vectors to compute the band structure
    k_ampl = np.pi / qsys.X_amplitude
    K = np.linspace(-0.5, 0.5, qsys.X_gridDIM)

    for epsilon in qsys.get_band_structure(K, 4):
        plt.plot(K, au2eV * epsilon, '-')

    plt.title("Reproduction of Fig. 1 from M. Wu et al. Phys. Rev A 91, 043839 (2015)")
    plt.xlabel("$k$ (units of $2\pi/ a_0$)")
    plt.ylabel('$\\varepsilon(k)$ (eV)')

    plt.show()
