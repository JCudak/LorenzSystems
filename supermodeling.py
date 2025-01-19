from systems import *
from plotting import *
import numpy as np
import pypesto
from scipy.integrate import solve_ivp
from pypesto import optimize
import random


def rms_error(model_solution, true_solution):
    return np.sqrt(np.mean((model_solution - true_solution) ** 2))
    
def assimilate_data(system_func, initial_state, observed_data, t_eval, bounds, solver_params=None, result=None, maxiter=None):
    def model(t, y, params):
        return system_func(t, y, *params)

    def objective_fun(params):
        t_span = t_eval[0], t_eval[-1]
        if solver_params is not None:
            solution = solve_ivp(model, t_span, initial_state, args=(params,),
                                t_eval=t_eval, method=solver_params[0], min_step=solver_params[1])
        else:
            solution = solve_ivp(model, t_span, initial_state, args=(params,), t_eval=t_eval)
        model_output = solution.y[0]

        return rms_error(model_output, observed_data)

    objective = pypesto.Objective(fun=objective_fun)

    lb = np.array([b[0] for b in bounds])  # lower bounds
    ub = np.array([b[1] for b in bounds])  # upper bounds

    problem = pypesto.Problem(objective=objective, lb=lb, ub=ub)

    opt = dict()
    if maxiter:
        opt['maxiter'] = maxiter
    optimizer = optimize.ScipyOptimizer(options=opt)

    res = optimize.minimize(problem=problem, optimizer=optimizer, result=result, n_starts=3, filename=None)

    return res


ground_truth_func = disturbed_lorenz_system

base_params = (10.0, 8. / 3., 28.0, 1.0, 5.0, 1.0)
bounds = [(5, 15), (0.5, 5), (20, 40.0), (0.5, 5), (3, 10), (0.5, 5)]
init_state = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
lorenz_init_state = [1.0, 1.0, 1.0]

t_span = (0, 90)
t_eval = np.linspace(*t_span, 10000)

base_solution = solve_ivp(ground_truth_func, t_span, init_state, args=base_params, t_eval=t_eval)


t_train_span = (40, 75)
train_indices = (base_solution.t >= t_train_span[0]) & (base_solution.t <= t_train_span[1])
train_indices = random.sample(list(np.where(train_indices)[0]), 1000)
train_indices.sort()
training_data = base_solution.y[:, train_indices]
training_times = base_solution.t[train_indices]



# Normal assimilation

# fit_result = assimilate_data(ground_truth_func, init_state, training_data[0], training_times, bounds)
# fitted_params = fit_result.optimize_result[0].x
# # print(f"Fitted Parameters: {fitted_params}")
# # for optres in fit_result.optimize_result:
# #     print(f"Optimization result: {optres}")

# fitted_solution = solve_system(ground_truth_func, init_state, fitted_params, t_span=t_span)
# print("Fitted times:", fitted_solution.t)
# print("Fitted solution:", fitted_solution.y)

# rms_forward = rms_error(fitted_solution.y[:, fitted_solution.t >= t_train_span[1]], base_solution.y[:, base_solution.t >= t_train_span[1]])
# rms_backward = rms_error(fitted_solution.y[:, fitted_solution.t <= t_train_span[0]], base_solution.y[:, base_solution.t <= t_train_span[0]])
# print(f"RMS Forward ({t_train_span[1]}-200): {rms_forward}")
# print(f"RMS Backward (0-{t_train_span[0]}): {rms_backward}")

# plot_trajectories(base_solution.t, base_solution.y[0], fitted_solution.y[0], f"{system['name']} - x Trajectory")





# Supermodeling

prefit_result = assimilate_data(lorenz_system, lorenz_init_state, training_data[0], training_times, bounds[:3], maxiter=10)
prefit_parameters = [p.x for p in prefit_result.optimize_result]
print(f"Prefit Parameters: {prefit_parameters}")


lorenz_supermodel = lorenz_supermodel_creator(prefit_parameters[0], prefit_parameters[1], prefit_parameters[2])


super_fit_result = assimilate_data(lorenz_supermodel, lorenz_init_state*3, training_data[0], training_times, [[-0.1, 0.1] for _ in range(6)], maxiter=1000)#, solver_params=('LSODA', 1e-10))
supermodel_fitted_params = super_fit_result.optimize_result[0].x
print(f"Fitted Parameters for the supermodel: {supermodel_fitted_params}")
for optres in super_fit_result.optimize_result:
    print(f"Optimization result: {optres}")


# supermodel_fit_solution = solve_system(lorenz_supermodel, lorenz_init_state*3, supermodel_fitted_params, method='LSODA', min_step=1e-15)
supermodel_fit_solution = solve_system(lorenz_supermodel, lorenz_init_state*3, supermodel_fitted_params, t_span=t_span)

rms_forward = rms_error(supermodel_fit_solution.y[0, supermodel_fit_solution.t >= t_train_span[1]], base_solution.y[0, base_solution.t >= t_train_span[1]])
rms_backward = rms_error(supermodel_fit_solution.y[0, supermodel_fit_solution.t <= t_train_span[0]], base_solution.y[0, base_solution.t <= t_train_span[0]])
print(f"RMS Forward ({t_train_span[1]}-200): {rms_forward}")
print(f"RMS Backward (0-{t_train_span[0]}): {rms_backward}")