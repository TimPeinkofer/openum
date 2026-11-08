import openum.Interpolation as interp
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("--- openum Example: Cubic Spline Interpolation ---")
    
    # 1. Definiere die Funktion, die interpoliert werden soll
    def func(x):
        return np.sin(x)

    # 2. Stützstellen (Grid Points) erzeugen
    x_data = np.linspace(0, 2 * np.pi, 6)
    n = len(x_data) - 1
    y_data = func(x_data)

    # 3. Spline-Momente (zweite Ableitungen) über openum berechnen
    matrix, vec, solution = interp.cubic_splines(n, x_data, func)

    if solution is None:
        print("Fehler bei der Spline-Berechnung.")
        return

    # 4. Hilfsfunktion, um den Spline zwischen den Punkten auszuwerten
    def evaluate_spline(x_eval, x_nodes, y_nodes, M):
        y_eval = np.zeros_like(x_eval)
        for j, x in enumerate(x_eval):
            # Finde das richtige Intervall für x
            for i in range(len(x_nodes) - 1):
                if x_nodes[i] <= x <= x_nodes[i + 1]:
                    h = x_nodes[i + 1] - x_nodes[i]
                    # Kubische Spline-Gleichung mit den Momenten M (solution)
                    term1 = (M[i] / (6 * h)) * (x_nodes[i + 1] - x)**3
                    term2 = (M[i + 1] / (6 * h)) * (x - x_nodes[i])**3
                    term3 = (y_nodes[i] - (M[i] * h**2) / 6) * ((x_nodes[i + 1] - x) / h)
                    term4 = (y_nodes[i + 1] - (M[i + 1] * h**2) / 6) * ((x - x_nodes[i]) / h)
                    y_eval[j] = term1 + term2 + term3 + term4
                    break
        return y_eval

    # 5. Feines Grid für den Plot erzeugen und Spline auswerten
    x_fine = np.linspace(0, 2 * np.pi, 100)
    y_interp = evaluate_spline(x_fine, x_data, y_data, solution)
    y_exact = func(x_fine)

    # Plot erstellen
    plt.figure(figsize=(10, 6))
    plt.plot(x_fine, y_exact, 'k--', alpha=0.5, label="Exakte Funktion (sin(x))")
    plt.plot(x_fine, y_interp, 'b-', label="openum Cubic Spline")
    plt.plot(x_data, y_data, 'ro', label="Stützstellen (Nodes)")
    
    plt.title("Cubic Spline Interpolation mit openum")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()