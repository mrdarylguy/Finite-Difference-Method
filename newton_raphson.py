'''
This script approximates roots of a function to within a given error tolerance

'''
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**3 - 2*x**2 + 5*x + 3


x = np.linspace(-4, 4, 100)
plt.title("cubic polynomial: x**3 - 2*x**2 + 5*x + 3")
plt.plot(x, f(x))
plt.axhline(0)
plt.grid(True)
plt.savefig("cubic_polynomial.png")
plt.show()

f_prime = lambda x: 3*x**2 - 4*x + 5

initial_guess = -0.5


newton_raphson = initial_guess - (f(initial_guess) / f_prime(initial_guess))

def recursive_newton_raphson(f, df, x0, tol):
    if abs(f(x0)) < tol:
        return x0
    else:
        return recursive_newton_raphson(f, df, x0 - f(x0)/df(x0), tol)

estimate = recursive_newton_raphson(f, f_prime, 0.5, 1e-6)


print('Newton Raphson value =', newton_raphson) #-0.483871
print('Estimate =', estimate) #-0.4837523
