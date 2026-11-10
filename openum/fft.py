import numpy as np
from numpy.typing import ArrayLike, NDArray

def cooley_tuckey(x: ArrayLike) -> NDArray[np.complex128]:
    """
    Compute fast Fourier transform using the Cooley-Tukey algorithm.
    
    
    Parameters:
    x: array_like
        Input array of complex numbers.
    
    Returns:
    result: array_like
        The FFT of the input array.
    """

    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    if N <= 1:
        return x

    even = cooley_tuckey(x[::2])
    odd = cooley_tuckey(x[1::2])
    T = np.exp(-2j * np.pi * np.arange(N) / N)
    result = np.concatenate([even + T[:N // 2] * odd, even + T[N // 2:] * odd])
    return result


def cooley_tuckey_ND(matrix: ArrayLike) -> NDArray[np.complex128]:
    """
    Compute fast Fourier transform using the Cooley-Tukey algorithm.
    
    
    Parameters:
    matrix: array_like
        Input array of complex numbers.
    
    Returns:
    result_ND: array_like
        The FFT of the input array.
    """

    matrix = np.asarray(matrix, dtype=complex)

    for axis in range(matrix.ndim):
        result_ND = np.empty_like(matrix)
        # Prüfung auf Zweierpotenz
        if matrix.shape[axis] & (matrix.shape[axis] - 1) != 0:
            raise ValueError(
                f"Input array must have a length that is a power of 2 along each axis. "
                f"Failed at axis {axis} with length {matrix.shape[axis]}."
            )    
        else:
            # 1D-FFT auf genau diese Achse anwenden und das Array überschreiben
            result_ND = np.apply_along_axis(cooley_tuckey, axis, matrix)

    
    
    return result_ND




