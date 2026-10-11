# Simple fitting example from the SciPy documentation (https://docs.scipy.org/doc/scipy/tutorial/optimize.html)

import numpy as np
from scipy.optimize import least_squares

# Model for an enzymatic reaction (https://doi.org/10.1016/0025-5564(68)90006-0)
def model(x, u): 
    return (x[0] * (u**2 + u * x[1])) / (u**2 + u * x[2] + x[3])

# Jacobien of the model function (with x as the only variable)
def jac(x, u, y):
    J = np.empty((u.size, x.size))
    num = u**2 + u * x[1]
    den = u**2 + u * x[2] + x[3]
    J[:,0] = num / den
    J[:,1] = x[0] * u / den
    J[:,2] = - x[0] * num / den**2 * u
    J[:,3] = - x[0] * num / den**2
    return J

# Set the values of u and y, initial value, and the constraints on x
u = np.array([4.0, 2.0, 1.0, 5.0e-1, 2.5e-1, 1.67e-1, 1.25e-1, 1.0e-1, 8.33e-2, 7.14e-2, 6.25e-2])
y = np.array([1.957e-1, 1.947e-1, 1.735e-1, 1.6e-1, 8.44e-2, 6.27e-2, 4.56e-2, 3.42e-2, 3.23e-2, 2.35e-2, 2.46e-2])
x0 = np.array([2.5, 3.9, 4.15, 3.9])
xmin = 0
xmax = 100

# Find the calue of x minimizing the L2 norm of the model
res = least_squares(lambda x, u, y: model(x, u) - y, x0, jac=jac, bounds=(xmin,xmax), args=(u,y), verbose=1)
print(f"Optimal valie of x: {res.x}")

# Plot the fitted model
import matplotlib.pyplot as plt
umin = 0
umax = 5
u_test = np.linspace(umin, umax)
y_test = model(res.x, u_test)
plt.plot(u, y, 'o', markersize=4, label='data')
plt.plot(u_test, y_test, label='fitted model')
plt.xlim(u_test[0], u_test[-1])
plt.xlabel('u')
plt.ylabel('y')
plt.grid()
plt.legend(loc='lower right')
plt.show()
