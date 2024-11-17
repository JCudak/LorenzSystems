
import matplotlib.pyplot as plt

def plot_3d_solution(solution, title='3D Plot of System', colors='blue'):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(solution.y[0], solution.y[1], solution.y[2], lw=0.7, color=colors)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_trajectories(t, true_solution, fitted_solution, title="Comparison of Trajectories"):
    plt.figure(figsize=(10, 6))
    plt.plot(t, true_solution, label="True Solution", lw=2)
    plt.plot(t, fitted_solution, label="Fitted Solution", lw=2, linestyle='dashed')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Dynamic Variable")
    plt.legend()
    plt.grid()
    plt.show()