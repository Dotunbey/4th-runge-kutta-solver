from sympy import symbols, Function
from runge_kutta_solver.reduction import reduce_to_first_order
from runge_kutta_solver.rk4_solver import runge_kutta
from runge_kutta_solver.error_estimation import estimate_rk4_error

x = symbols('x')
y = Function('y')
f_expr = 1 + 2*x*y(x) - x**2*y(x).diff(x)  # Example: y″ = 1 + 2xy − x²y′
order = 2
f_func, _ = reduce_to_first_order(f_expr, order)

x0 = 0.0
y0 = [1.0, 0.0]
h = 0.1
x_end = 1.0

xs, ys, err = estimate_rk4_error(f_func, order, x0, y0, h, x_end)

for xi, yi, ei in zip(xs, ys, err):
    print(f"x = {xi:.2f}, y ≈ {yi:.6f}, error ≈ {ei:.2e}")
