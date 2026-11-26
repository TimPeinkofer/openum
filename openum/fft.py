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
    result_ND = np.asarray(matrix, dtype=complex)

    for axis in range(result_ND.ndim):
        if result_ND.shape[axis] == 0 or (result_ND.shape[axis] & (result_ND.shape[axis] - 1)) != 0:
            raise ValueError(
                f"Input array must have a length that is a power of 2 along each axis. "
                f"Failed at axis {axis} with length {result_ND.shape[axis]}."
            )
        # Auf result_ND anwenden, nicht auf matrix
        result_ND = np.apply_along_axis(cooley_tuckey, axis, result_ND)

    return result_ND
