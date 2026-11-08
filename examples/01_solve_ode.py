import openum.ODE as ode

def main():
    print("--- openum Example: Ordinary Differential Equations ---")
    
    # Definition of the differential equation: dy/dt = -0.5 * y
    def f(t, y):
        return -0.5 * y

    # Initial conditions
    t0 = 0.0          # Start time
    y0 = 100.0        # Initial value
    t_end = 10.0      # End time
    step_size = 0.1   # Step size

    # Calculation (adjust the function name to match your ODE.py, e.g., euler or runge_kutta)
    t_values, y_values = ode.euler_method(f, t0, y0, t_end, step_size)

    # Print results
    print(f"Initial value at time {t0}: {y0}")
    print(f"Calculated value at time {t_end}: {y_values[-1]:.4f}")

if __name__ == "__main__":
    main()