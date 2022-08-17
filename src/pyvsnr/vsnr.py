"""
2D-VSNR algorithm working on CPU and GPU, based on the simplified MATLAB code
issued from : https://www.math.univ-toulouse.fr/~weiss/PageCodes.html

Original algorithm : Jerome FEHRENBACH, Pierre WEISS
Developper : Patrick QUEMERE
"""
import numpy as np
import matplotlib.pyplot as plt

from pyvsnr.filters import gabor_filter
from pyvsnr.utils import pad_centered

# check GPU environment
try:
    import cupy as cp  # check cupy has been installed

    cp.cuda.runtime.getDeviceCount()  # check a GPU is reachable
    GPU_ENV = True
except:
    GPU_ENV = False


class VSNR:  # pylint: disable=I0011,C0103
    """
    Class dedicated to vsnr calculation on CPU and GPU

    .. note:
    To ease original MATLAB / Python code comparison most of the variable names
    have been kept during the code transcription.
    Accordingly, PEP8 formatting compatibility is not respected.
    """

    def __init__(self, shape):
        """
        VSNR object instantiation

        Parameters
        ----------
        shape: tuple of 2 ints
            Shape of the images to work with
        """
        self.shape = shape

        self.alphas = []
        self.filters = []

        self.filters_fft = None
        self.q = None
        self.lambdas = None
        self.lambdas_b = None

        self.tau = None
        self.sigma = None

        self.cvg_criterias = None

        # GPU/CPU variable and associated numerical library
        self.is_gpu = None
        self.xp = None

    def add_filter(self, alpha, name='gabor', sigma=None, theta=0.):
        """
        Add filter to the VSNR object

        Parameters
        ----------
        alpha: float
            vsnr parameter associated to the filter
        name: str, optional
            Filter type to choose among Gabor filter and horizontal or vertical
            Dirac filter : 'gabor', 'dirac_h', 'dirac_v' respectively
        sigma: tuple of 2 floats, optional
            Filter sizes associated to a Gabor filter in the x and y directions
            respectively
        theta: float, optional
            Rotation angle (in degree) to consider in the case of a Gabor filter
        """
        if name == "gabor":
            vsnr_filter = gabor_filter(sigma, theta=theta)
            vsnr_filter *= 0.01 / vsnr_filter.max()  # renormalization
            vsnr_filter = pad_centered(vsnr_filter, self.shape, value=0.)
        elif name == 'dirac_h':
            vsnr_filter = np.zeros(self.shape)
            vsnr_filter[0, :] = 1. / self.shape[0]
        elif name == 'dirac_v':
            vsnr_filter = np.zeros(self.shape)
            vsnr_filter[:, 0] = 1. / self.shape[1]
        else:
            names = ['gabor', 'dirac_h', 'dirac_v']
            raise IOError(f"name should be defined among {names}")

        self.alphas.append(alpha)
        self.filters.append(vsnr_filter)

    def initialize(self, is_gpu=True):
        """
        vsnr computation initialization (to do before eval() method call)

        Parameters
        ----------
        is_gpu: bool, optional
            Keyword to order GPU computation (if possible)
       """
        if is_gpu and GPU_ENV:
            print("VSNR will run on GPU ...")
            self.xp = cp
            self.is_gpu = True
            # arrays transposition (host to device)
            self.filters = self.xp.asarray(self.filters)
            self.alphas = self.xp.asarray(self.alphas)
        else:
            print("VSNR will run on CPU ...")
            self.xp = np

        # variables names contraction (to ease the usage)
        xp, fft = self.xp, self.xp.fft.rfft2

        # arrays initialization
        self.q = xp.zeros((2, self.shape[0], self.shape[1]))
        self.lambdas = xp.zeros_like(self.filters)
        self.lambdas_b = xp.zeros_like(self.filters)
        d1 = xp.zeros(self.shape)
        d1[-1, 0] = 1
        d1[0, 0] = -1
        d2 = xp.zeros(self.shape)
        d2[0, -1] = 1
        d2[0, 0] = -1

        # fourier transforms computation and storage
        self.filters_fft = []
        for filter in self.filters:
            self.filters_fft.append(fft(filter))
        d1_fft = fft(d1)
        d2_fft = fft(d2)

        # vsnr parameters initialization
        H = xp.zeros(self.filters_fft[0].shape)
        for filter_fft in self.filters_fft:
            H += xp.abs(filter_fft) ** 2
        norm = xp.sqrt(xp.max(H * (xp.abs(d1_fft ** 2) + xp.abs(d2_fft ** 2))))
        self.tau = self.sigma = 1. / norm

    def eval(self, u0, maxit=100, cvg_threshold=1e-4):
        """
        vsnr computation

        Parameters
        ----------
        u0: numpy.ndarray((m, n))
            2D array corresponding to the image to deal with
        maxit: int, optional
            Maximum iterations of the iterative processing
        cvg_threshold: float, optional
            Convergence criteria value used to stop the iterative processing

        Returns
        -------
        u: numpy.ndarray((m, n))
            The corrected image
        """
        msg = f"image shape {u0.shape} passed to vsnr.eval() differs from shape"
        msg += f" {self.shape} passed to the vsnr object instantiation"
        assert (u0.shape == self.shape), msg

        # variables names contraction (to ease the usage)
        xp, fft, ifft = self.xp, self.xp.fft.rfft2, self.xp.fft.irfft2

        # arrays transposition (host to device)
        if self.is_gpu:
            u0 = xp.asarray(u0)

        # iterative process variables initialization
        nit = 0
        u = u_old = u0
        cvg_criteria = 1000.
        self.cvg_criterias = []
        Astarq = xp.zeros_like(self.filters)
        nfilters = len(self.filters)

        while nit < maxit and cvg_criteria > cvg_threshold:

            self.q[0] += self.sigma * self.drond1(u)
            self.q[1] += self.sigma * self.drond2(u)

            norm = xp.sqrt(self.q[0] ** 2 + self.q[1] ** 2)
            self.q /= xp.maximum(norm, 1)

            q_grad_fft = fft(self.drond1T(self.q[0]) + self.drond2T(self.q[1]))

            for k in range(nfilters):
                Astarq[k] = ifft(xp.conj(self.filters_fft[k]) * q_grad_fft,
                                 Astarq[k].shape)

            lambdas_u = self.lambdas - self.tau * Astarq

            for k in range(nfilters):
                lambdas_u[k] = self.Prox_Phi(lambdas_u[k], self.tau,
                                             self.alphas[k])
            theta = 1.
            self.lambdas_b = lambdas_u + theta * (lambdas_u - self.lambdas)
            self.lambdas = lambdas_u

            # Correction term computation
            du = xp.zeros_like(u0)
            for k in range(nfilters):
                du += ifft(fft(self.lambdas_b[k]) * self.filters_fft[k],
                           du.shape)

            # Current denoised image estimation
            u = u0 + du

            # cvg_criteria : Max(||u_(n+1)-u_(n)||)
            cvg_criteria = float(xp.max(xp.abs(u - u_old)))
            self.cvg_criterias.append(cvg_criteria)

            nit += 1
            u_old = u

        # arrays transposition (device to host)
        if self.is_gpu:
            u = xp.asnumpy(u)

        return u

    def drond1(self, u):  # pylint: disable=C0116
        shape = self.shape
        res = self.xp.zeros(shape)
        res[:shape[0] - 1, :] = u[1:shape[0], :] - u[:shape[0] - 1, :]
        return res

    def drond1T(self, u):  # pylint: disable=C0116
        shape = self.shape
        res = self.xp.zeros(shape)
        res[1:shape[0] - 1, :] = u[:shape[0] - 2, :] - u[1:shape[0] - 1, :]
        res[0, :] = -u[0, :]
        res[shape[0] - 1, :] = u[shape[0] - 2, :]
        return res

    def drond2(self, u):  # pylint: disable=C0116
        shape = self.shape
        res = self.xp.zeros(shape)
        res[:, :shape[1] - 1] = u[:, 1:shape[1]] - u[:, :shape[1] - 1]
        return res

    def drond2T(self, u):  # pylint: disable=C0116
        shape = self.shape
        res = self.xp.zeros(shape)
        res[:, 1:shape[1] - 1] = u[:, :shape[1] - 2] - u[:, 1:shape[1] - 1]
        res[:, 0] = -u[:, 0]
        res[:, shape[1] - 1] = u[:, shape[1] - 2]
        return res

    def Prox_Phi(self, x, tau, alpha):
        """
        This function solves :

        .. math::
         argmin_{|y|<=C} \tau \alpha ||y||_1 + 1/2 ||M(y-x)||_2^2
        """
        C = 100.
        y = self.xp.maximum(self.xp.abs(x) - tau * alpha, 0)
        y *= self.xp.sign(x) / (self.xp.maximum(self.xp.abs(y) / C, 1))
        return y

    def plot_cvg_criteria(self):
        """
        Plot cvg_criteria calculated during the iterative processing

        Returns
        -------
        fig: a matplotib.pyplot.figure
        """
        fig = plt.figure()
        plt.plot(self.cvg_criterias)
        plt.title("Convergence criteria evolution")
        plt.xlabel("# iteration [n]")
        plt.ylabel(r"$Max(||u_{(n+1)}-u_{(n)}||)$")
        return fig
