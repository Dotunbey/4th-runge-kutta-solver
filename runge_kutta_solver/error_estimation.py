import sympy as sp
import numpy as np
def estimate_rk4_error(f_func, order, x0, y0, h, x_end):
    """
    Estimate the global error for any order IVP using RK4 and Richardson Extrapolation.

    Parameters:
        f_func: RHS function from reduced first-order system.
        order: Order of the original ODE (int).
        x0: Initial value of x.
        y0: List of initial values [y, y', ..., y^{n-1}].
        h: Step size.
        x_end: Final x value.

    Returns:
        xs: x-values grid
        y0_values: Solution for the primary function y(x)
        error_estimates: Estimated global error for y(x)
    """
    # Get RK4 solution with step size h
    xs_h, ys_h = runge_kutta(f_func, order, x0, y0, h, x_end)

    # Get RK4 solution with step size h/2 (more accurate)
    xs_h2, ys_h2 = runge_kutta(f_func, order, x0, y0, h/2, x_end)

    # Align both solutions: pick every second point from h/2 solution
    aligned_ys_h2 = ys_h2[:, ::2]  # Matches same x-grid as xs_h

    # Estimate error using only the y0 component (the function itself, not derivatives)
    y0_h = ys_h[0]         # y at step h
    y0_h2 = aligned_ys_h2[0]  # y at step h/2 (downsampled)

    # Richardson extrapolation formula for RK4: error ≈ (y_h2 - y_h) / (2^4 - 1)
    error = np.abs(y0_h2 - y0_h) / 15.0

    return xs_h, y0_h, error
