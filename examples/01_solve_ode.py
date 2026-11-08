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
    
    # openum expects the number of steps (n) instead of the step size
    n = int((t_end - t0) / step_size)

    # Calculation using Runge-Kutta 4th order from your ODE module
    # Signature: runge_kutta(x_start, x_end, y_0, n, f, plotchoose)
    t_values, y_values = ode.runge_kutta(t0, t_end, y0, n, f, plotchoose=True)

    # Print results
    print(f"Initial value at time {t0}: {y0}")
    print(f"Calculated value at time {t_end}: {y_values[-1]:.4f}")

if __name__ == "__main__":
    main()