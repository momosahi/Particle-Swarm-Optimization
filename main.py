from pso import PSO
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# fonction test de Easom
def test_fonction(x1, x2):
    return -math.cos(x1) * math.cos(x2) * math.exp(-((x1 - math.pi) ** 2 + (x2 - math.pi) ** 2))


pso = PSO(
    fonction=test_fonction, dim=2, particle=50, iteration=120, min=[-100, -100], max=[100, 100], w=0.8, phi1=2, phi2=2
)


fig, ax = plt.subplots(1, 1)
line = plt.plot([], [], "b.")


def update_scatter(frame):
    plt.clf()
    plt.xlim(-110, 110)
    plt.ylim(-110, 110)
    i = frame
    plt.title("iter = " + str(i))
    plt.setp(line, "xdata", pso.coord["X"][i][:, 0], "ydata", pso.coord["X"][i][:, 1])
    plt.scatter(pso.coord["X"][i][:, 0], pso.coord["X"][i][:, 1], s=5, marker="o", linewidths=1)
    return line


if __name__ == "__main__":
    pso.run()
    print(f"global best position: {pso.global_best_position}, global best fitness: {pso.global_best_fitness}")
    plt.plot(pso.global_best_fitness_history)
    plt.show()
    animation = FuncAnimation(fig, update_scatter, blit=True, interval=100, frames=pso.iteration)
    animation.save("pso.gif", writer="pillow")
