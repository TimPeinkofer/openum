# Welcome to openum

**openum** is a Python library for numerical mathematics and scientific computing. It provides accessible and efficient algorithms for solving ordinary and partial differential equations, calculating integrals, finding roots, and applying stochastic methods.

## Installation

You can install `openum` directly from GitHub (or via PyPI, once published):

```bash
pip install openum
```

*For local development and testing, please refer to our [Contributing Guidelines](https://github.com/timpeinkofer/openum/blob/main/CONTRIBUTING.md).*

## Quick Start

Here is a simple example showing how to use the Monte Carlo module to estimate an integral:

```python
import openum.MC as mc

# Define the mathematical function
def f(x):
    return x**2

# Calculate the integral from 0 to 1
result, error = Monte_Carlo(10000, 0, 1, f)

print(f"Estimated Integral: {result}")
print(f"Estimated Error: {error}")
```

## Features & Modules

Explore the API documentation for our core modules:

- **[Boundary Value Problems (BVP)](bvp.md)**
- **[Ordinary Differential Equations (ODE)](ode.md)**
- **[Partial Differential Equations (PDE)](pde.md)**
- **[Monte Carlo Methods (MC)](mc.md)**
- **[Numerical Integration](numint.md)** & **[Differentiation](numdiff.md)**
- **[Non-linear](nonlineq.md)** & **[Linear Equations](numlinequ.md)**
- **[Interpolation](interpolation.md)**
- **[Numerical Eigenvalues](numeigenv.md)**
- **[Error Calculation](errorcalc.md)**

## License & Security

`openum` is open-source software. Please refer to the `LICENSE.md` in the repository for more details. For security-related information, see our `SECURITY.md` policy.
