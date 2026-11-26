import unittest
from unittest.mock import patch
import numpy as np
import sys
import os

# Add the path to the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'openum')))

from openum.opti import gradient_descent


class TestGradientDescent(unittest.TestCase):

    def test_1d_quadratic(self):
        """Testet das Finden des Minimums für f(x) = x^2 + 2x + 1 (Minimum bei x = -1)."""
        f = lambda x: x**2 + 2 * x + 1
        df = lambda x: 2 * x + 2

        x_min, history = gradient_descent(
            f, df, x0=4.0, learning_rate=0.1, max_iter=1000, plotchoose=False, tol=1e-6
        )

        self.assertAlmostEqual(float(x_min), -1.0, places=4)
        self.assertGreater(len(history), 1)

    def test_2d_paraboloid(self):
        """Testet das Finden des Minimums für f(x, y) = x^2 + y^2 (Minimum bei [0, 0])."""
        f = lambda v: v[0]**2 + v[1]**2
        grad_f = lambda v: [2 * v[0], 2 * v[1]]

        x_min, history = gradient_descent(
            f, grad_f, x0=[3.0, -4.0], learning_rate=0.1, max_iter=1000, plotchoose=False, tol=1e-6
        )

        np.testing.assert_allclose(x_min, [0.0, 0.0], atol=1e-4)
        self.assertEqual(history.shape[1], 2)

    def test_already_at_minimum(self):
        """Wenn x0 bereits das Minimum ist, sollte sofort abgebrochen werden."""
        f = lambda x: x**2
        df = lambda x: 2 * x

        x_min, history = gradient_descent(
            f, df, x0=0.0, learning_rate=0.1, max_iter=1000, plotchoose=False, tol=1e-6
        )

        self.assertAlmostEqual(float(x_min), 0.0, places=6)
        self.assertEqual(len(history), 1)

    def test_max_iter_limit(self):
        """Stellt sicher, dass max_iter eingehalten wird."""
        f = lambda x: x**2
        df = lambda x: 2 * x

        _, history = gradient_descent(
            f, df, x0=10.0, learning_rate=0.001, max_iter=15, plotchoose=False, tol=1e-12
        )

        # Startpunkt + 15 Iterationen = 16 Einträge
        self.assertEqual(len(history), 16)

    @patch("matplotlib.pyplot.show")
    def test_plotchoose_does_not_crash(self, mock_show):
        """Prüft, ob die Plot-Funktion im 1D-Fall fehlerfrei aufgerufen wird (CI-sicher)."""
        f = lambda x: x**2
        df = lambda x: 2 * x

        gradient_descent(
            f, df, x0=2.0, learning_rate=0.2, max_iter=10, plotchoose=True, tol=1e-6
        )
        mock_show.assert_called_once()


if __name__ == "__main__":
    unittest.main()