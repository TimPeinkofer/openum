import openum.Interpolation as interp
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("--- openum Example: Interpolation ---")
    
    # Given data points (e.g., measured values)
    x_data = [0, 1, 2, 3, 4, 5]
    y_data = [1, 2.5, 2, 4.5, 3, 5]

    # Create a fine grid for a smooth curve
    x_fine = np.linspace(0, 5, 100)
    
    # Calculate the interpolated values 
    # (Adjust 'lagrange' to match the function in your Interpolation.py)
    y_interp = [interp.lagrange(x_data, y_data, x) for x in x_fine]

    # Create plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_fine, y_interp, label="openum Interpolation", color="blue")
    plt.plot(x_data, y_data, 'ro', label="Original data points")
    
    plt.title("Interpolation with openum")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()