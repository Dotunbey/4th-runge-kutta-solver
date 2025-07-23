import sympy as sp
import numpy as np
def f_system(x_val, y_vals, order, f_func):
    """
    Return the derivatives [y0', y1', ..., y_{n-1}'] for the system of first-order ODEs.

    Parameters:
        x_val : current x value
        y_vals : list of current values [y0, y1, ..., y_{n-1}]
        order : the order of the ODE
        f_func : the lambda function for the system of ODEs

    Returns:
        derivs : array of derivatives [y0', y1', ..., y_{n-1}']
    """
    derivs = np.zeros(order)

    # y0' = y1, y1' = y2, ..., y_{n-2}' = y_{n-1}
    for i in range(order-1):
        derivs[i] = y_vals[i+1]

    # Last equation: y_{n-1}' = f(x, y0, ..., y_{n-1})
    derivs[order-1] = f_func(x_val, *y_vals)

    return derivs

def runge_kutta(f_func, order, x0, initial_conditions, h, x_end):
    """
    Solve the system of first-order ODEs using the 4th-order Runge-Kutta method.

    Parameters:
        f_func : function that returns the system of derivatives
        order : the order of the ODE (e.g., 2 for y'')
        x0 : initial x value
        initial_conditions : initial values for y(0), y'(0), ..., y^(n-1)(0)
        h : step size
        x_end : end value of x

    Returns:
        xs : array of x values
        ys : array of solution values for each step
    """
    N_steps = int((x_end - x0) / h)  # number of RK4 steps
    xs = np.zeros(N_steps + 1)
    ys = np.zeros((order, N_steps + 1))

    xs[0] = x0
    ys[:, 0] = initial_conditions

    # RK4 loop
    for i in range(N_steps):
        x_curr = xs[i]
        y_curr = ys[:, i]

        # Compute RK4 slopes k1..k4 using the system function
        k1 = f_system(x_curr, y_curr, order, f_func)
        k2 = f_system(x_curr + 0.5*h, y_curr + 0.5*h*k1, order, f_func)
        k3 = f_system(x_curr + 0.5*h, y_curr + 0.5*h*k2, order, f_func)
        k4 = f_system(x_curr + h,     y_curr + h*k3, order, f_func)

        # Update solution for next step
        ys[:, i+1] = y_curr + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        xs[i+1] = x_curr + h

    return xs, ys
