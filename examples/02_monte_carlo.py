import openum.MC as mc

def main():
    print("--- openum Example: Monte Carlo Integration ---")
    
    # The function to integrate
    def func(x):
        return x**2

    # Integration limits
    a = 0.0
    b = 2.0
    samples = 50000

    # Calculation (adjust the function name to match your MC.py)
    result = mc.monte_carlo_integration(func, a, b, num_samples=samples)

    print(f"Function: f(x) = x^2")
    print(f"Interval: [{a}, {b}]")
    print(f"Number of samples: {samples}")
    print(f"Estimated integral: {result:.4f}")
    print(f"Exact value: 2.6666...")

if __name__ == "__main__":
    main()