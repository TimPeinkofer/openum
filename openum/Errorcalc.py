from . import numint, numdiff
from typing import Callable
from numpy.typing import ArrayLike, NDArray
import numpy as np

class IntegrationandDiffMethodError:
    def __init__(self, method, a: float, b: float, func:Callable):
        """
        Initialisiert die Integrationsmethode.

        Args:
            method (function): Die Integrationsmethode (z. B. Newton_cotes oder Trapezoidal)
            a (float): Untere Grenze des Integrals
            b (float): Obere Grenze des Integrals
            func (function): Funktion, die integriert werden soll
        """
        self.method = method
        self.n = 1000  # Standardanzahl der Unterintervalle
        self.a = a
        self.b = b
        self.func = func
        self.exact = None

    def calculate_exact(self):
        """
        Führt die numerische Integration durch und speichert das Ergebnis.

        Returns:
            float: Das Ergebnis der numerischen Integration
        """
        self.exact = self.method(self.n, self.a, self.b, self.func)
        return self.exact

    def calculate_error(self, approx_value: float) -> float:
        """
        Berechnet den Fehler der numerischen Integration, falls der exakte Wert bekannt ist.

        Args:
            approx_value (float): Der approximierte Wert des Integrals

        Returns:
            float: Der Fehler der numerischen Integration
        """
        if self.exact is None:
            self.calculate_exact()
        return abs(approx_value - self.exact)


def error(method, a: float, b: float, func:Callable, approx_result: float) -> float:
    """
    Berechnet den Fehler der numerischen Integration.

    Args:
        method: Methode, die für die Berechnung des approximierten Werts verwendet wird
        a (float): Untere Grenze des Integrals
        b (float): Obere Grenze des Integrals
        func: Funktion, die integriert werden muss
        approx_result (float): Der approximierte Wert des Integrals
    
    Returns:
        error (float): Fehler der numerischen Integration
    """
    err_func = IntegrationandDiffMethodError(method, a, b, func)
    exact_value = err_func.calculate_exact()  # Calculate exact value
    error_value = err_func.calculate_error(approx_result)  # Calculate the error
    return error_value


class FFTMethodError:
    def __init__(self, input_data: ArrayLike, is_nd: bool = False):
        """
        Initialisiert die FFT-Fehlerberechnung.

        Args:
            input_data (ArrayLike): Das Eingabearray für die FFT
            is_nd (bool): True, wenn es sich um ein mehrdimensionales Array handelt
        """
        self.input_data = np.asarray(input_data, dtype=complex)
        self.is_nd = is_nd
        self.exact = None

    def calculate_exact(self) -> NDArray[np.complex128]:
        """
        Berechnet die exakte FFT mithilfe von numpy als Referenz.
        """
        if self.is_nd:
            self.exact = np.fft.fftn(self.input_data)
        else:
            self.exact = np.fft.fft(self.input_data)
        return self.exact

    def calculate_error(self, approx_result: NDArray[np.complex128], metric: str = 'max') -> float:
        """
        Berechnet die Abweichung zwischen deiner FFT und der exakten FFT.

        Args:
            approx_result (NDArray): Das Ergebnis deiner cooley_tuckey Funktion
            metric (str): 'max' für maximalen Fehler, 'mse' für mittleren quadratischen Fehler

        Returns:
            float: Der berechnete Fehler
        """
        if self.exact is None:
            self.calculate_exact()
            
        # Berechnet die absoluten Differenzen zwischen den komplexen Einträgen
        diff = np.abs(self.exact - approx_result)
        
        if metric == 'max':
            return float(np.max(diff))
        elif metric == 'mse':
            return float(np.mean(diff**2))
        else:
            raise ValueError("Unbekannte Metrik. Wähle 'max' oder 'mse'.")


def fft_error(input_data: ArrayLike, approx_result: NDArray[np.complex128], is_nd: bool = False) -> float:
    """
    Berechnet den maximalen Fehler einer approximierten FFT.

    Args:
        input_data (ArrayLike): Das ursprüngliche Eingabearray
        approx_result (NDArray): Das von deiner Methode berechnete FFT-Array
        is_nd (bool): True für mehrdimensionale FFT
    
    Returns:
        float: Der maximale Fehler
    """
    err_func = FFTMethodError(input_data, is_nd)
    exact_value = err_func.calculate_exact()
    error_value = err_func.calculate_error(approx_result, metric='max')
    return error_value