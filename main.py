import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from SALib.sample import sobol
from SALib.analyze.sobol import analyze as sobol_analyze

def lorenz_system(t, state, sigma, beta, rho):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def yang_system(t, state, alpha, beta, gamma):
    x, y, z = state
    dx_dt = alpha * (y - x)
    dy_dt = gamma * x - x * z
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def solve_system(system_func, initial_state, params, t_span=(0, 50), num_points=10000):
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    solution = solve_ivp(system_func, t_span, initial_state, args=params, t_eval=t_eval)
    return solution

def plot_3d_solution(solution, title='3D Plot of System', colors='blue'):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(solution.y[0], solution.y[1], solution.y[2], lw=0.7, color=colors)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

initial_state_lorenz = [1.0, 1.0, 1.0]
params_lorenz = (10.0, 8.0 / 3.0, 28.0)  # sigma, beta, rho

initial_state_yang = [1.5, 1.5, 1.5]
params_yang = (10.0, 8.0 / 3.0, 16.0)  # alpha, beta, gamma

solution_lorenz = solve_system(lorenz_system, initial_state_lorenz, params_lorenz)
plot_3d_solution(solution_lorenz, title='3D Plot of the Lorenz System')

solution_yang = solve_system(yang_system, initial_state_yang, params_yang)
plot_3d_solution(solution_yang, title='3D Plot of the Yang System', colors='green')


problem_lorenz = {
    'num_vars': 3,
    'names': ['sigma', 'beta', 'rho'],
    'bounds': [[0.5, 20.0], [0.5, 5.0], [20.0, 50.0]]  # Zakresy parametrów
}

problem_yang = {
    'num_vars': 3,
    'names': ['alpha', 'beta', 'gamma'],
    'bounds': [[0.5, 20.0], [0.5, 5.0], [5.0, 30.0]]  # Zakresy parametrów
}

param_values_lorenz = sobol.sample(problem_lorenz, 32)
param_values_yang = sobol.sample(problem_yang, 32)

def run_simulation_for_samples(system_func, param_values, initial_state):
    results = []
    for params in param_values:
        print(f"Solving for parameters: {params}")
        solution = solve_system(system_func, initial_state, tuple(params))
        results.append(solution.y[0, -1])  # Zapisujemy np. końcową wartość x
    return np.array(results)

results_lorenz = run_simulation_for_samples(lorenz_system, param_values_lorenz, initial_state_lorenz)

results_yang = run_simulation_for_samples(yang_system, param_values_yang, initial_state_yang)

sobol_indices_lorenz = sobol_analyze(problem_lorenz, results_lorenz)

sobol_indices_yang = sobol_analyze(problem_yang, results_yang)

print("Sobol Indices for Lorenz System:")
print(f"First-order indices: {sobol_indices_lorenz['S1']}")
print(f"Total-order indices: {sobol_indices_lorenz['ST']}")

print("\nSobol Indices for Yang System:")
print(f"First-order indices: {sobol_indices_yang['S1']}")
print(f"Total-order indices: {sobol_indices_yang['ST']}")
