from pso import PSO
import math


# fonction de Easom
def test_fonction(x1, x2):
    return -math.cos(x1) * math.cos(x2) * math.exp(-((x1 - math.pi) ** 2 + (x2 - math.pi) ** 2))


pso = PSO(
    fonction=test_fonction, dim=2, particule=50, iteration=120, min=[-100, -100], max=[100, 100], w=0.8, phi1=2, phi2=2
)


if __name__ == "__main__":
    pso.run()
    print(pso.best_position)
    print(pso.best_fitness)
    print(pso.best_fitness_iteration)
    print(pso.best_position_iteration)
    print(pso.best_fitness_history)
    print(pso.best_position_history)
    pso.plot()
