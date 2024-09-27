import numpy as np
import skimage

from tests import IMATestCase


class TestCreateHistogramEqualizationLut(IMATestCase):

    def test_create_histogram_equalization_lut(self):
        expected_lut = np.array(
            [   0,   0,   2,   5,  10,  17,  28,  40,
               54,  67,  77,  85,  90,  94,  98, 103,
              112, 125, 142, 164, 186, 207, 225, 237,
              246, 250, 253, 254, 254, 254, 254, 255],
            dtype = np.uint8
        )
        a = np.arange(32)
        h = 600 * np.exp(-((a - 8) ** 2) / 4**2) + 1000 * np.exp(-((a - 20) ** 2) / 4**2)
        lut = self.params['create_histogram_equalization_lut_fn'](h)
        self.assertEqual(lut.dtype, np.uint8)
        self.assertArraysClose(lut, expected_lut, atol=1.0, rtol=0.0)


class TestEqualizeHistogram(IMATestCase):

    def test_equalize_histogram(self):
        gray = skimage.util.img_as_ubyte(skimage.io.imread(self.rpath('data/rentgen.bmp'), as_gray=True))
        expected_equ = skimage.util.img_as_ubyte(skimage.exposure.equalize_hist(gray))
        equ = self.params['equalize_histogram_fn'](gray)
        self.assertEqual(equ.dtype, np.uint8)
        self.assertArraysClose(equ, expected_equ, atol=1.0, rtol=0.0)


class TestEqualizeHistogramRGBIndep(IMATestCase):

    def test_equalize_histogram_rgb_indep(self):
        expected_equ = skimage.io.imread(self.rpath('tests/test_data/histogram_equalization-fruits_equalized_indep.png'))
        rgb = skimage.io.imread(self.rpath('data/fruits.jpg'))
        equ = self.params['equalize_histogram_rgb_indep_fn'](rgb)
        self.assertEqual(equ.dtype, np.uint8)
        self.assertArraysClose(equ, expected_equ, atol=1.0, rtol=0.0)


class TestEqualizeHistogramRGBGray(IMATestCase):

    def test_equalize_histogram_rgb_gray(self):
        expected_equ = skimage.io.imread(self.rpath('tests/test_data/histogram_equalization-fruits_equalized_gray.png'))
        rgb = skimage.io.imread(self.rpath('data/fruits.jpg'))
        equ = self.params['equalize_histogram_rgb_gray_fn'](rgb)
        self.assertEqual(equ.dtype, np.uint8)
        self.assertArraysClose(equ, expected_equ, atol=1.0, rtol=0.0)


class TestCreateLUT(IMATestCase):

    def test_create_lut(self):
        expected_luts = np.loadtxt(
            '../tests/test_data/histogram_equalization-target_luts.csv',
            dtype=np.uint8,
            delimiter=','
        )
        luts = np.stack([
            self.params['create_lut_fn']([(0, 255), (255, 0)]),
            self.params['create_lut_fn']([(0, 0), (50, 10), (200, 245), (255, 255)]),
            self.params['create_lut_fn']([(0, 0), (100, 0), (101, 255), (255, 255)]),
            self.params['create_lut_fn']([(50, 150), (128, 30), (200, 255)]),
        ])
        self.assertEqual(luts.dtype, np.uint8)
        self.assertArraysClose(luts, expected_luts, atol=1.0, rtol=0.0)
