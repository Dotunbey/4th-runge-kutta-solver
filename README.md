# 4th-runge-kutta-solver
# General Runge-Kutta Solver for Higher-Order ODEs

This repository provides a symbolic and numerical implementation of the classical 4th-order Runge-Kutta method for solving arbitrary **n-th order ODEs**.

## ✨ Features

- Automatically reduces higher-order ODEs to a system of first-order ODEs using `sympy`
- Numerically solves using the classic RK4 method (`numpy`)
- Provides error estimation using Richardson extrapolation
- Works for 1st, 2nd, 3rd, 4th... n-th order ODEs

## 📦 Requirements

```bash
pip install -r requirements.txt
