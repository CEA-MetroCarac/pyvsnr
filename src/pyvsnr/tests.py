"""
unittest
"""
import unittest
import numpy as np

from pyvsnr.examples import ex_camera


class TestVSNR(unittest.TestCase):
    """
    Test VSNR algorithm
    """

    def test_ex_camera_stripes(self):
        """ Test VSNR algorithm on stripes removal """

        img_corr = ex_camera('stripes', show_plots=False)

        self.assertAlmostEqual(np.sum(img_corr), 131985.69656241007)

    def test_ex_camera_curtains(self):
        """ Test VSNR algorithm on curtains removal """

        img_corr = ex_camera('curtains', show_plots=False)

        self.assertAlmostEqual(np.sum(img_corr), 124973.38026424834)
