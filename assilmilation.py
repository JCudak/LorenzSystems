from systems import *
from plotting import *
import numpy as np
import pypesto
from scipy.integrate import solve_ivp
from pypesto import optimize


def rms_error(model_solution, true_solution):
    return np.sqrt(np.mean((model_solution - true_solution) ** 2))


def calculate_rms_over_time(base_solution, fitted_solution):
    """
    Calculate the RMS error over time between the base solution and the fitted solution.
    Each RMS value is computed for the corresponding time step.
    """
    # Difference between base and fitted solutions
    if base_solution.shape[0] == 3 or fitted_solution.shape[0] == 3:
        differences = base_solution[:3] - fitted_solution[:3]
    else:
        differences = base_solution - fitted_solution
    # RMS calculated across variables (rows) for each time step (columns)
    rms_values = np.sqrt(np.mean(differences ** 2, axis=0))
    return rms_values


def assimilate_data(system_func, initial_state, observed_data, t_span, bounds, solver_params):
    def model(t, y, params):
        return system_func(t, y, *params)

    def objective_fun(params):
        t_eval = np.linspace(t_span[0], t_span[1], len(observed_data[0]))
        if solver_params is not None:
            solution = solve_ivp(model, t_span, initial_state, args=(params,),
                                 t_eval=t_eval, method=solver_params[0], min_step=solver_params[1])
        else:
            solution = solve_ivp(model, t_span, initial_state, args=(params,), t_eval=t_eval)
        model_output = solution.y

        return rms_error(model_output, observed_data)

    objective = pypesto.Objective(fun=objective_fun)

    lb = np.array([b[0] for b in bounds])  # lower bounds
    ub = np.array([b[1] for b in bounds])  # upper bounds

    problem = pypesto.Problem(objective=objective, lb=lb, ub=ub)

    optimizer = optimize.ScipyOptimizer()

    result = optimize.minimize(problem=problem, optimizer=optimizer, n_starts=10, filename=None)

    return result.optimize_result[0].x


def main():
    systems = [
        # {
        #     "name": "Lorenz",
        #     "ground_truth_func": lorenz_system,
        #     "solver_params": None,
        #     "base_params": [(10.0, 8.0 / 3.0, 28.0), (9.9, 2.5, 27.5)],
        #     "bounds": [(5, 20), (0.5, 5), (20, 50)],
        #     "gt_initial_state": [1.0, 1.0, 1.0]
        # },
        # {
        #     "name": "Yang",
        #     "ground_truth_func": yang_system,
        #     "solver_params": None,
        #     "base_params": [(10, 8. / 3., 16), (9.9, 2.5, 15.9)],
        #     "bounds": [(5, 20), (2, 5), (10, 20)],
        #     "gt_initial_state": [1.5, 1.5, 1.5]
        # },
        # {
        #     "name": "Chen",
        #     "ground_truth_func": chen_system,
        #     "solver_params": None,
        #     "base_params": [(34.5, 2.7, 28.0), (34.55, 2.8, 29.0)],
        #     "bounds": [(33.0, 38.0), (2.0, 5.5), (26.0, 31.0)],
        #     "gt_initial_state": [1.0, 1.0, 1.0]
        # },
        # {
        #     "name": "Lu",
        #     "ground_truth_func": lu_system,
        #     "solver_params": None,
        #     "base_params": [(36, 3, 20), (35.5, 2.5, 19.5)],
        #     "bounds": [(32.0, 39.0), (1.5, 5.5), (15.0, 24.5)],
        #     "gt_initial_state": [1.0, 1.0, 1.0]
        # },
        {
            "name": "Distorded Lorenz System",
            "ground_truth_func": disturbed_lorenz_system,
            "surrogate_func": lorenz_system,
            "base_params": [(10.0, 8. / 3., 28.0, 1.0, 5.0, 1.0), (10.0, 8. / 3., 28.0, 1.0, 5.0, 1.0)],
            "bounds": [(5, 15), (20, 40.0), (0.5, 5), (0.5, 5), (3, 10), (0.5, 5)],
            "gt_initial_state": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "surrogate_state_len": 3,
        }
    ]

    sample_params = [
        # ((30, 40), 2),
        ((30, 40), 5),
        ((30, 40), 10),
        ((30, 40), 40)
    ]

    for system in systems:
        print(f"Processing {system['name']} system")

        ground_truth_func = system['ground_truth_func']
        base_params = system["base_params"][0]
        initial_state = system['gt_initial_state']

        t_span = (0, 90)
        t_eval = np.linspace(*t_span, 10000)
        base_solution = solve_ivp(ground_truth_func, t_span, initial_state, args=base_params, t_eval=t_eval)

        fig = plot_3d_solution(base_solution, title=f"{system['name']} - 3D Phase Space")
        save_plot(fig, "Surrogated_" + system['name'], (0, 90), 0, "3D_Phase_Space")
        for std_dev in [0.5, 2, 5]:
            for t_sample_span, sampling_points in sample_params:
                print(f"Sampling for t_sample_span = {t_sample_span}, sampling_points = {sampling_points}, std_dev = {std_dev}")

                t_sample = np.linspace(t_sample_span[0], t_sample_span[1], sampling_points)
                # print("t_sample", t_sample)
                # print("base sol", base_solution.t.shape, base_solution.y.shape)
                sampled_data = [np.interp(t_sample, base_solution.t, base_solution.y[i]) for i in
                                range(base_solution.y.shape[0])]

                # print("lol0", sampled_data)
                noisy_sampled_data = [data + np.random.normal(0, std_dev, size=len(data))
                                      for data in sampled_data]
                # print("lol1", noisy_sampled_data)
                sampled_points = [(t_sample, noisy_sampled_data[i]) for i in range(len(noisy_sampled_data))]
                # print("lol", sampled_data)

                # Plot sampled data
                fig = plot_dynamic_variation(base_solution.t, base_solution.y, samples=sampled_points,
                                             title=f"{system['name']} - Sampled Dynamics for {t_sample_span[0]}-{t_sample_span[1]}")
                save_plot(fig, "Surrogated_" + system['name'], t_sample_span, sampling_points, "Sampled_Dynamics")
                # Fit model to sampled data

                fitted_params = assimilate_data(ground_truth_func, initial_state, noisy_sampled_data, t_sample_span,
                                                system['bounds'], system.get('solver_params'))
                print(f"Fitted Parameters for {system['name']}: {fitted_params}")

                # Predict forward and backward

                fitted_solution = solve_ivp(ground_truth_func, t_span, initial_state, args=fitted_params, t_eval=t_eval)
                fig = plot_dynamic_variation(fitted_solution.t, fitted_solution.y, samples=sampled_points,
                                             title=f"{system['name']} - Fitted Sampled Dynamics for {t_sample_span[0]}-{t_sample_span[1]}")
                save_plot(fig, "Fitted_surrogated_" + system['name'], t_sample_span, sampling_points, "Sampled_Dynamics")

                to_train_func = system.get('surrogate_func', ground_truth_func)
                surrogated_params = assimilate_data(to_train_func, initial_state[:3], noisy_sampled_data[:3], t_sample_span,
                                                    system['bounds'][:3], system.get('solver_params'))

                surrogated_solution = solve_ivp(to_train_func, t_span, initial_state[:3], args=surrogated_params,
                                                t_eval=t_eval)
                print(f"Surrogated system fitted parameters: {surrogated_params}")

                print("RMS - base/fitted: ",rms_error(base_solution.y[:3, :], fitted_solution.y[:3, :]))
                print("RMS - base/surrogated: ",rms_error(base_solution.y[:3, :], surrogated_solution.y[:3, :]))
                print("RMS - fitted/surrogated: ",rms_error(fitted_solution.y[:3, :], surrogated_solution.y[:3, :]))


if __name__ == "__main__":
    main()
