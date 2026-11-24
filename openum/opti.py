import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, grad_f, x0, learning_rate, max_iter,plotchoose, tol=1e-6):
    """
    Findet ein lokales Minimum einer Funktion mittels Gradientenabstieg.

    Args:
        f: Zielfunktion f(x).
        grad_f: Gradient (Ableitung) der Funktion grad_f(x).
        x0: Startpunkt (float für 1D oder Liste/ndarray für nD).
        learning_rate: Schrittweite (Alpha).
        max_iter: Maximale Anzahl an Iterationen.
        tol: Abbruchkriterium (Norm des Gradienten).
        plotchoose: Boolean, um den Abstieg (im 1D-Fall) zu plotten.

    Returns:
        tuple: (x_min, history) - Gefundenes Minimum und Verlauf der Schritte.
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]

    for i in range(max_iter):
        grad = np.array(grad_f(x), dtype=float)

        # Abbruchkriterium: Gradient ist nah genug an 0
        if np.linalg.norm(grad) < tol:
            break

        # Eigentlicher Update-Schritt
        x = x - learning_rate * grad
        history.append(x.copy())

    history = np.array(history)

    # Integrierter Plot für den 1D-Fall (didaktisch ideal)
    if plotchoose and x.ndim == 0:
        _plot_1d_descent(f, history)

    return x, history


def _plot_1d_descent(f, history):
    """Hilfsfunktion zur Visualisierung des 1D-Gradientenabstiegs."""
    x_min_plot = min(history) - 1.0
    x_max_plot = max(history) + 1.0
    x_vals = np.linspace(x_min_plot, x_max_plot, 400)
    y_vals = [f(val) for val in x_vals]
    hist_y = [f(val) for val in history]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label="f(x)", color="blue")
    plt.plot(history, hist_y, 'ro--', label="Abstiegsschritte", markersize=5)
    plt.scatter(history[-1], hist_y[-1], color="green", s=100, zorder=5, label="Gefundenes Minimum")
    plt.title("Gradientenabstieg (Gradient Descent)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
