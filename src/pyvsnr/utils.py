import numpy as np

def create_gabor(n0, n1, level, sigmax, sigmay, angle, phase, lambda_, xp):
    """Create a Gabor filter."""
    psi = xp.zeros((n0, n1), dtype=xp.float32)

    theta = xp.radians(angle)
    offset_x = (n1 / 2.0) + 1.0
    offset_y = (n0 / 2.0) + 1.0
    phase = xp.radians(phase)
    nn = xp.pi / xp.sqrt(sigmax * sigmay)

    i = xp.arange(n1)
    j = xp.arange(n0)
    x, y = xp.meshgrid(i, j)

    x = offset_x - x
    y = offset_y - y
    x_theta = x * xp.cos(theta) + y * xp.sin(theta)
    y_theta = y * xp.cos(theta) - x * xp.sin(theta)

    val = xp.exp(
        -0.5 * ((x_theta / sigmax) ** 2 + (y_theta / sigmay) ** 2)
    ) * xp.cos((x_theta * lambda_ / sigmax) + phase)
    psi = level * val / nn

    return psi.flatten()

def pad_centered(arr, shape_ref, value=0):
    """
    Return a centered ND-array with a surrounding padding 'value'

    Parameters
    ----------
    arr: numpy.ndarray()
        Array to handle
    shape_ref:
        Final array shape to reach
    value: float, optional
        Value used for the padding

    Returns
    -------
    arr_pad: numpy.ndarray()
        The resulting padded array
    """
    assert len(shape_ref) == len(arr.shape)

    dim = len(shape_ref)
    arr_pad = arr.copy()
    for k in range(dim):
        # gap between shape_ref and shape_max to pad
        gap = shape_ref[k] - arr.shape[k]
        gap2 = gap // 2

        # swap axes to work on axis=0
        arr_pad = np.swapaxes(arr_pad, 0, k)

        # padding
        if gap >= 0:
            width = (gap2, gap - gap2)
            if dim > 1:
                width = (width,) + (dim - 1) * ((0, 0),)
            arr_pad = np.pad(arr_pad, width, constant_values=value)
        # cutting
        else:
            arr_pad = arr_pad[-gap2 : -gap2 + shape_ref[k], ...]

        # return to original axis
        arr_pad = np.swapaxes(arr_pad, 0, k)

    return arr_pad


def curtains_addition(img_ref, seed=None, amplitude=0.2, sigma=(3, 40), angle=0, threshold=0.999, norm=True):
    """
    Add curtains effects in a image

    Parameters
    ----------
    img_ref: numpy.ndarray((m, n))
        Original image
    seed: float, optional
        Seed associated to randomized noise (for stripes definition)
    amplitude: float, optional
        Positive relative amplitude associated to the strides.
    sigma: tuple of 2 floats, optional
        Pixel sizes of the spots (curtains) in x and y directions respectively
    theta: float, optional
        Spot orientation (angle (in clockwise) [in degrees])
    threshold: float, optional
        Parameters to select more or less curtains positions. The higher the
        threshold is, the less positions there are

    Returns
    -------
    img: numpy.ndarray((m, n))
        Image with curtains
    """
    assert amplitude >= 0.0

    np.random.seed(seed)

    n0, n1 = img_ref.shape
    sigmax, sigmay = sigma

    # relative to absolute noise amplitude conversion
    vmin, vmax = img_ref.min(), img_ref.max()

    amplitude *= vmax

    # curtains definition (from gabor filter) and location
    psi = create_gabor(
        n0, n1, 0.2, sigmax, sigmay, angle=angle, phase=0, lambda_=0.0, xp=np
    ).reshape(n0, n1)

    psi *= 0.01 / psi.max()  # renormalization
    psi = pad_centered(psi, img_ref.shape, value=0)

    position = np.random.random(img_ref.shape)
    position = (position > threshold).astype(float)
    noise = np.fft.irfft2(np.fft.rfft2(position) * np.fft.rfft2(psi))

    noise *= amplitude / noise.max()

    # if dark_curtains:
    #     noise *= -1.

    img = img_ref + noise
    if norm:
        img = np.clip(img, vmin, vmax)

    return img


def stripes_addition(img_base, amplitude, seed=None, norm=True):
    """ Add stripes defects in a image """
    np.random.seed(seed)

    noisy_img = img_base.copy()
    vmin, vmax = noisy_img.min(), noisy_img.max()

    if norm:
        noisy_img = noisy_img / vmax

    for i in range(img_base.shape[0]):
        noise = amplitude * (np.random.random() - 0.5)
        noisy_img[i] += noise

    if norm:
        noisy_img = np.clip(noisy_img, 0, 1)

    return noisy_img


def add_gaussian_noise(img, scale=0.1):
    """ Add gaussian noise in a image """

    vmin, vmax = img.min(), img.max()
    noise =  np.random.normal(loc=0, scale=scale, size=img.shape)
    noisy_img = img + noise

    noisy_img = np.clip(noisy_img,vmin,vmax)

    return noisy_img