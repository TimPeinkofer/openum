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

    # Calculation (Parameters: n, a, b, func)
    result, error = mc.Monte_Carlo(samples, a, b, func)

    print(f"Function: f(x) = x^2")
    print(f"Interval: [{a}, {b}]")
    print(f"Number of samples: {samples}")
    print(f"Estimated integral: {result:.4f}")
    print(f"Estimated error: ±{error:.4f}")
    print(f"Exact value: 2.6667")

if __name__ == "__main__":
    main()