def test_core():
    import numpy as np
    from pyvsnr import vsnr2d

    img = np.random.random((100, 100))  # Input image
    filters = [{'name':'Dirac', 'noise_level':0.35}]  # List of filters

    img_corr_py = vsnr2d(img, filters, algo="numpy") # output is a 2D array (100, 100)

    # check if the output is a 2D array
    assert img_corr_py.shape == (100, 100)

    # check if the output has changed
    assert np.any(img_corr_py != img)