import numpy as np
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix
from band_structure_1d import MUBQBandStructure

##########################################################################################
#
#   NOTE: The Bloch_Gibbs propagator implemented incorrectly
#
##########################################################################################

class MUBGibbsBand(MUBQBandStructure):
    """
    Calculate the Gibbs satte for band structure for quantum Hamiltonian, H(x,p) = K(p) + V(x),
    using mutually unbiased bases (MUB).
    """
    def get_gibbs(self, k_val, kT, n=-1):
        """
        Return the Gibbs state rho = exp(-H/kT) by diagonalizing the Hamiltonian
        :param k_val: (float) quasimomentum (Bloch vector)
        :param n: number of bands to include
        :param kT: temperature
        :return:
        """
        vals, vecs = linalg.eigh(self.get_hamiltonian(k_val))

        # extract only first n eigenvectors
        vecs = vecs[:,:n]
        vals = vals[:n]

        # According to the Bloch theorem, we need to multiply by the plane wave with quasimomentum
        vecs *= np.exp(1j * k_val * self.X_range[:, np.newaxis])

        # rho[a,b] = sum over c of vecs[a,c] * conj(vecs[b,c]) * exp(-E[c]/kT)
        rho_gibbs = np.einsum('ac,bc,c', vecs, vecs.conj(), np.exp(-vals / kT))

        # normalize the obtained gibbs state
        rho_gibbs /= rho_gibbs.trace()

        return rho_gibbs

    def get_gibbs_bloch(self, k_val, kT, dbeta=0.005):
        """
        Get Gibbs state by solving the Bloch equation
        :param k_val: (float) quasimomentum (Bloch vector)
        :param kT: (float) temperature
        :return:
        """
        # get number of dbeta steps to reach the desired Gibbs state
        num_beta_steps = 1. / (kT * dbeta)

        if round(num_beta_steps) <> num_beta_steps:
            # Changing dbeta so that num_beta_steps is an exact integer
            num_beta_steps = np.ceil(num_beta_steps)
            dbeta = 1. / (kT * num_beta_steps)

        num_beta_steps = int(num_beta_steps)

        # Pre-calculate the exponents
        X = self.X_range[:, np.newaxis]
        X_prime = self.X_range[np.newaxis, :]

        expV = self.sign_flip * np.exp(-0.25 * dbeta * (self.V(X) + self.V(X_prime)))
        expK = np.exp(-0.5 * dbeta * (
            self.K(self.P_range[:, np.newaxis] - k_val) + self.K(self.P_range[np.newaxis, :] - k_val)
        ))

        # initiate the Gibbs state with the infinite temperature state
        rho_gibb = np.eye(self.X_gridDIM, dtype=np.complex)

        # propagate the state in beta
        for _ in xrange(num_beta_steps):
            rho_gibb *= expV

            rho_gibb = fftpack.ifft(rho_gibb, overwrite_x=True, axis=0)
            rho_gibb = fftpack.fft(rho_gibb, overwrite_x=True, axis=1)

            rho_gibb *= expK

            rho_gibb = fftpack.fft(rho_gibb, overwrite_x=True, axis=0)
            rho_gibb = fftpack.ifft(rho_gibb, overwrite_x=True, axis=1)

            rho_gibb *= expV
            rho_gibb /= rho_gibb.trace()

        # Factorize in the quasimomentum
        rho_gibb *= np.exp(1j * k_val * (X - X_prime))

        return rho_gibb

    def get_energy(self, rho):
        """
        Calculate the expectation value of the hamiltonian over the denisty matrix rho
        :param rho: (numpy.array) denisty matrix
        :return: float
        """
        rho = rho.copy()

        # normalize the state
        rho /= rho.trace()

        # expectation value of the potential energy
        average_V = np.dot(rho.diagonal().real, self.V(self.X_range))

        # going into the momentum representation
        rho *= self.sign_flip
        rho_p = fftpack.ifft(rho, overwrite_x=True, axis=0)
        rho_p = fftpack.fft(rho_p, overwrite_x=True, axis=1)
        rho_p *= self.sign_flip

        rho_p /= rho_p.trace()

        # expectation value of the kinetic energy
        average_K = np.dot(rho_p.diagonal().real, self.K(self.P_range))

        return average_K + average_V

##########################################################################################
#
# Example
#
##########################################################################################

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    print(MUBGibbsBand.__doc__)

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

    def is_physicial(rho):
        """
        check for physicallity of the density matrix rho
        """
        p = linalg.eigvalsh(rho)
        if not np.allclose(p[p < 0], 0) and not np.allclose(rho.diagonal().imag, 0):
            print("WARNING: Obtained Gibbs denisty matrix i not a positively defined matrix")

    # initialize the system
    qsys = MUBGibbsBand(**sys_params)

    # range of bloch vectors to compute the band structure
    #k_ampl = np.pi / qsys.X_amplitude
    #K = np.linspace(-0.5 * k_ampl, 0.5 * k_ampl, 200)

    #plt.subplot(121)

    quasimomentum = 0.
    kT = 0.5

    gibbs = qsys.get_gibbs(quasimomentum, kT, n=1)
    is_physicial(gibbs)
    print("Energy of the Gibbs state via MUB: %f (a.u.)" % qsys.get_energy(gibbs))

    #gibbs_bloch = qsys.get_gibbs_bloch(quasimomentum, kT)
    #is_physicial(gibbs_bloch)
    #print("Energy of the Gibbs state via Bloch: %f (a.u.)" % qsys.get_energy(gibbs_bloch))

    #print("Difference between Gibs via MUB and via Bloch = %1.2e" % np.linalg.norm(gibbs - gibbs_bloch))

    ##########################################################################################
    #
    # Begin CUDA check
    #
    ##########################################################################################

    from rho_bloch_cuda_1d import RhoBlochCUDA1D

    # some adapdation for CUDA
    sys_params.update(
        V="-V0 * (1. + cos(M_PI * (X + X_amplitude) / X_amplitude))",
        K="0.5 * P * P",
        quasimomentum=quasimomentum,
        dt=0.01,
    )

    qsys_cuda = RhoBlochCUDA1D(**sys_params)
    qsys_cuda.get_ground_state(dbeta=1e-3)
    gibbs_cuda = qsys_cuda.propagate(8000).rho.get()

    #gibbs_cuda = qsys_cuda.get_gibbs_state(kT=kT).get()
    gibbs_cuda /= gibbs_cuda.trace()

    print(
        "Difference between Gibs via MUB and via Bloch CUDA = %1.2e" \
            % np.linalg.norm(gibbs - gibbs_cuda)
    )

    ##########################################################################################
    #
    # End CUDA check
    #
    ##########################################################################################

    plt.subplot(221)
    plt.title("Gibbs state")
    plt.imshow(gibbs.real, origin='lower')
    plt.colorbar()

    plt.subplot(222)
    plt.title("Gibbs state via Bloch")
    plt.imshow(gibbs_cuda.real, origin='lower')
    plt.colorbar()

    plt.subplot(223)
    plt.title("Gibbs state")
    plt.imshow(gibbs.imag, origin='lower')
    plt.colorbar()

    plt.subplot(224)
    plt.title("Gibbs state via Bloch")
    plt.imshow(gibbs_cuda.imag, origin='lower')
    plt.colorbar()

    # plt.plot(qsys.X_range, gibbs.diagonal().real,label='Gibbs')
    # plt.plot(qsys.X_range, gibbs_bloch.diagonal().real, label='Gibbs via Bloch')
    # plt.legend(loc='upper center')

    # plt.subplot(132)
    # plt.title("Gibbs bloch")
    # plt.imshow(np.real(gibbs_bloch), origin='lower')
    # plt.colorbar()
    #
    # plt.subplot(122)
    # for epsilon in qsys.get_band_structure():
    #     plt.plot(K, epsilon)
    #
    # print("Mininum energy: %f (a.u.)" % qsys.get_band_structure().min())
    #
    # plt.title("Reproduction of Fig. 1 from M. Wu et al. Phys. Rev A 91, 043839 (2015)")
    # plt.xlabel("$k$ (a.u.)")
    # plt.ylabel('$\\varepsilon(k)$ (a.u.)')

    plt.show()
