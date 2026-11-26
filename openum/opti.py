import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(f, grad_f, x0, learning_rate, max_iter, plotchoose, tol=1e-6):
    """
    Finds a local minimum of a function using gradient descent.

    Args:
        f: Objective function f(x).
        grad_f: Gradient (derivative) of the function grad_f(x).
        x0: Initial guess (float for 1D or list/ndarray for nD).
        learning_rate: Step size (alpha).
        max_iter: Maximum number of iterations.
        plotchoose: Boolean to plot the descent (in the 1D case).
        tol: Stopping criterion (norm of the gradient).

    Returns:
        tuple: (x_min, history) - Found minimum and history of steps.
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]

    for i in range(max_iter):
        grad = np.array(grad_f(x), dtype=float)

        # Stopping criterion: gradient is close enough to 0
        if np.linalg.norm(grad) < tol:
            break

        # Actual update step
        x = x - learning_rate * grad
        history.append(x.copy())

    history = np.array(history)

    # Integrated plot for the 1D case
    if plotchoose and x.ndim == 0:
        _plot_1d_descent(f, history)

    return x, history


def _plot_1d_descent(f, history):
    """Helper function to visualize the 1D gradient descent."""
    x_min_plot = min(history) - 1.0
    x_max_plot = max(history) + 1.0
    x_vals = np.linspace(x_min_plot, x_max_plot, 400)
    y_vals = [f(val) for val in x_vals]
    hist_y = [f(val) for val in history]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label="f(x)", color="blue")
    plt.plot(history, hist_y, 'ro--', label="Descent steps", markersize=5)
    plt.scatter(history[-1], hist_y[-1], color="green", s=100, zorder=5, label="Found minimum")
    plt.title("Gradient Descent")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()