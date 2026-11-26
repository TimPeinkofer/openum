import unittest
import numpy as np
# Angenommen, deine Funktionen sind hier definiert oder importiert
# from dein_modul import cooley_tuckey, cooley_tuckey_ND

class TestCooleyTuckey(unittest.TestCase):

    def test_cooley_tuckey_1d_real(self):
        """Testet die 1D-FFT mit reellen Zahlen."""
        x = np.array([1.0, 2.0, 1.0, -1.0, 1.5, 2.5, 3.5, 4.5])
        expected = np.fft.fft(x)
        result = cooley_tuckey(x)
        np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

    def test_cooley_tuckey_1d_complex(self):
        """Testet die 1D-FFT mit komplexen Zahlen."""
        x = np.array([1+1j, 2-1j, 3+2j, 4-3j])
        expected = np.fft.fft(x)
        result = cooley_tuckey(x)
        np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

    def test_cooley_tuckey_ND_2d(self):
        """Testet die ND-FFT auf einem 2D-Array (4x4)."""
        np.random.seed(42)
        matrix = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
        expected = np.fft.fftn(matrix)
        result = cooley_tuckey_ND(matrix)
        np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

    def test_cooley_tuckey_ND_3d(self):
        """Testet die ND-FFT auf einem 3D-Array (2x4x8)."""
        np.random.seed(42)
        matrix = np.random.rand(2, 4, 8)
        expected = np.fft.fftn(matrix)
        result = cooley_tuckey_ND(matrix)
        np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

    def test_cooley_tuckey_ND_value_error(self):
        """Testet, ob der ValueError bei falschen Dimensionen ausgelöst wird."""
        matrix = np.ones((3, 4)) # 3 ist keine Zweierpotenz
        with self.assertRaises(ValueError) as context:
            cooley_tuckey_ND(matrix)
        
        self.assertTrue("power of 2" in str(context.exception))

if __name__ == '__main__':
    unittest.main()