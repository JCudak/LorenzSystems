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



def lorenz_supermodel_creator(params1, params2, params3):
    sigma1, beta1, rho1 = params1
    sigma2, beta2, rho2 = params2
    sigma3, beta3, rho3 = params3

    def inner(t, state, c12, c13, c23, c21, c31, c32):
        x1, y1, z1, x2, y2, z2, x3, y3, z3 = state

        dx1dt, dy1dt, dz1dt = lorenz_system(t, [x1, y1, z1], sigma1, beta1, rho1)
        dx2dt, dy2dt, dz2dt = lorenz_system(t, [x2, y2, z2], sigma2, beta2, rho2)
        dx3dt, dy3dt, dz3dt = lorenz_system(t, [x3, y3, z3], sigma3, beta3, rho3)

        dx1dt += c12*(x2-x1) + c13*(x3-x1)
        dx2dt += c21*(x1-x2) + c23*(x3-x2)
        dx3dt += c31*(x1-x3) + c32*(x2-x3)

        return [dx1dt, dy1dt, dz1dt, dx2dt, dy2dt, dz2dt, dx3dt, dy3dt, dz3dt]
    
    return inner


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
