from systems import *
from plotting import *
import numpy as np
import pypesto
from scipy.integrate import solve_ivp
from pypesto import optimize

def rms_error(model_solution, true_solution):
    return np.sqrt(np.mean((model_solution - true_solution) ** 2))




def assimilate_data(system_func, initial_state, observed_data, t_span, initial_params, bounds):

    def model(t, y, params):
        return system_func(t, y, *params)

    def objective_fun(params):
        solution = solve_ivp(model, t_span, initial_state, args=(params,),
                             t_eval=np.linspace(t_span[0], t_span[1], len(observed_data)))
        model_output = solution.y[0]

        return rms_error(model_output, observed_data)

    objective = pypesto.Objective(fun=objective_fun)

    lb = np.array([b[0] for b in bounds])  # lower bounds
    ub = np.array([b[1] for b in bounds])  # upper bounds

    problem = pypesto.Problem(objective=objective, lb=lb, ub=ub)

    optimizer = optimize.ScipyOptimizer()

    result = optimize.minimize(problem=problem, optimizer=optimizer, n_starts=10, filename=None)

    return result.optimize_result[0].x

if __name__ == "__main__":
    systems = [
        {
            "name": "Lorenz",
            "system_func": lorenz_system,
            "base_params": (10.0, 8.0 / 3.0, 28.0),
            "start_params": (9.5, 2.5, 25),
            "bounds": [(5, 20), (0.5, 5), (20, 50)],
            "initial_state": [1.0, 1.0, 1.0]
        },
        {
            "name": "Yang",
            "system_func": yang_system,
            "base_params": (10, 8. / 3., 16),
            "start_params": (9, 2.5, 15),
            "bounds": [(5, 20), (2, 5), (10, 20)],
            "initial_state": [1.5, 1.5, 1.5]
        },
        {
            "name": "Chen",
            "system_func": chen_system,
            "base_params": (35, 3, 28),
            "start_params": (30, 2.5, 25),
            "bounds": [(20, 40), (2, 5), (20, 35)],
            "initial_state": [1.0, 1.0, 1.0]
        },
        {
            "name": "Lu",
            "system_func": lu_system,
            "base_params": (36, 3, 20),
            "start_params": (32, 2.5, 18),
            "bounds": [(20, 40), (2, 5), (10, 25)],
            "initial_state": [1.0, 1.0, 1.0]
        }
    ]

    for system in systems:
        print(f"Processing {system['name']} system")

        base_solution = solve_system(system["system_func"], system["base_params"], system["initial_state"])

        t_train_span = (10, 30)
        train_indices = (base_solution.t >= t_train_span[0]) & (base_solution.t <= t_train_span[1])
        training_data = base_solution.y[:, train_indices]
        training_times = base_solution.t[train_indices]

        fitted_params = assimilate_data(system["system_func"], system["initial_state"], training_data[0],
                                        t_train_span, system["start_params"], system["bounds"])
        print(f"Fitted Parameters for {system['name']}: {fitted_params}")

        fitted_solution = solve_system(system["system_func"], fitted_params, system["initial_state"])

        rms_forward = rms_error(fitted_solution.y[:, fitted_solution.t >= 30],
                                base_solution.y[:, base_solution.t >= 30])
        rms_backward = rms_error(fitted_solution.y[:, fitted_solution.t <= 10],
                                 base_solution.y[:, base_solution.t <= 10])
        print(f"RMS Forward (30-50): {rms_forward}")
        print(f"RMS Backward (0-10): {rms_backward}")

        plot_trajectories(base_solution.t, base_solution.y[0], fitted_solution.y[0],
                          f"{system['name']} - x Trajectory")
        plot_3d_solution(base_solution, title=f"{system['name']} - Base Model", colors="blue")
        plot_3d_solution(fitted_solution, title=f"{system['name']} - Fitted Model", colors="orange")