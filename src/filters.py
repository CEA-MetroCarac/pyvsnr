"""
Filters definition
"""

import numpy as np


def gabor_filter(sigma, theta=0., lambda_=0., psi=0.,
                 normalization=True):
    r"""
    Return a Gabor filter

    Parameters
    ----------
    sigma: tuple of 2 floats
        Standard deviations of the Gaussian envelope in the x and y directions
        respectively
    theta: float, optional
        Rotation angle (in clockwise) to apply [in degrees]
    lambda_: float, optional
        Wavelength of the sinusoidal factor
    psi: float, optional
        Phase offset
    normalization: bool, optional
        Activation key for filter normalization

    Returns
    -------
    gbf: numpy.ndarray((m, n))
        Gabor filter in physical space (real part) defined on a :math:`3-\sigma`
        support
    """
    assert (sigma[0] > 0. and sigma[1] > 0.)

    sigma_x, sigma_y = sigma
    theta_rad = np.deg2rad(theta)

    # Bounding box
    nstds = 3  # Number of standard deviation (3.sigma -> 99.97% of the signal)
    xmax = max(abs(nstds * sigma_x * np.sin(theta_rad)),
               abs(nstds * sigma_y * np.cos(theta_rad)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.cos(theta_rad)),
               abs(nstds * sigma_y * np.sin(theta_rad)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    x, y = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    n = max(xmax, ymax)

    # Rotation
    x_theta = x * np.cos(theta_rad) + y * np.sin(theta_rad)
    y_theta = -x * np.sin(theta_rad) + y * np.cos(theta_rad)

    x_compo = (x_theta / sigma_x) ** 2
    y_compo = (y_theta / sigma_y) ** 2

    gbf = np.exp(-(x_compo + y_compo) / 2.)
    gbf *= np.cos(2 * np.pi / n * lambda_ * x_theta + psi)

    if normalization:
        gbf /= (2. * np.pi * sigma_x * sigma_y)

    return gbf
