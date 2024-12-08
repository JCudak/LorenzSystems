from scipy.integrate import solve_ivp
import numpy as np

def lorenz_system(t, state, sigma, beta, rho):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def disturbed_lorenz_system(t, state, sigma, beta, rho, eps, delta, ni):
    xv, yv, zv, xh, yh, zh = state

    dxv_dt = sigma * (yv - xv) + eps*zh
    dyv_dt = xv * (rho - zv) - yv
    dzv_dt = xv*yv - beta*zv + delta*(xh + ni)

    dxh_dt = sigma * (yh - xh)
    dyh_dt = xh * (rho - zh) - yh
    dzh_dt = xh * yh - beta * zh

    return [dxv_dt, dyv_dt, dzv_dt, dxh_dt, dyh_dt, dzh_dt]


def yang_system(t, state, alpha, beta, gamma):
    x, y, z = state
    dx_dt = alpha * (y - x)
    dy_dt = gamma * x - x * z
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def chen_system(t, state, alpha, beta, gamma):
    x, y, z = state
    dx_dt = alpha*(y - x)
    dy_dt = (gamma - alpha - z)*x + gamma*y
    dz_dt = x*y - beta*z
    return [dx_dt, dy_dt, dz_dt]

def lu_system(t, state, alpha, beta, gamma):
    x, y, z = state
    dx_dt = alpha*(y - x)
    dy_dt = -x*z + gamma*y
    dz_dt = x*y - beta*z
    return [dx_dt, dy_dt, dz_dt]

def unified_system(t, state, alpha, beta, gamma):
    x, y, z = state
    dx_dt = (25*alpha + 10)*(y - x)
    dy_dt = (28 - 35*alpha)*x - x*z + (29*alpha - 1)*y
    dz_dt = x*y - (alpha + 8)/3*z
    return [dx_dt, dy_dt, dz_dt]


def solve_system(system_func, initial_state, params, t_span=(0, 50), num_points=10000):
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    solution = solve_ivp(system_func, t_span, initial_state, args=params, t_eval=t_eval)

    return solution
