import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def reduce_to_first_order(f_expr, order):
    """
    Reduces an nth-order ODE to a system of first-order ODEs.

    Parameters:
        f_expr : sympy expression for the nth-order ODE
        order  : the order of the ODE (e.g., 2 for y'')

    Returns:
        f_func  : a function representing the system of first-order ODEs
        Y_symbols : list of first-order variables [y, y', ..., y^(n-1)]
    """
    # Create symbols Y0, Y1, ..., Y_{order-1} representing y, y', ..., y^(n-1)
    Y_symbols = sp.symbols(' '.join([f'Y{i}' for i in range(order)]), seq=True)

    # Build a substitution map: y(x)->Y0, y'(x)->Y1, ..., y^(n-1)(x)->Y_{n-1}
    subs_map = {sp.Function('y')(x): Y_symbols[0]}
    for i in range(1, order):
        subs_map[sp.Derivative(sp.Function('y')(x), x, i)] = Y_symbols[i]

    # Substitute into the ODE to express the highest derivative in terms of Y_symbols
    f_sub = f_expr.subs(subs_map)

    # Lambdify to generate the function for numerical evaluation
    f_func = sp.lambdify((x,) + Y_symbols, f_sub, 'numpy')

    return f_func, Y_symbols
