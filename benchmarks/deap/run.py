import argparse
import sys
import os
import time
import operator
import math
import random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from deap import base, creator, gp, tools
from benchmarks.datasets import get_dataset
from benchmarks.logging_utils import BenchmarkLogger

def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def run_deap_benchmark(task, n_samples, n_features, pop_size, generations):
    """Sets up and runs the DEAP benchmark for a given configuration."""
    logger = BenchmarkLogger()
    X, y = get_dataset(task, n_samples, n_features)

    pset = gp.PrimitiveSet("MAIN", n_features)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(operator.neg, 1)
    
    if task == 'trigonometric':
        pset.addPrimitive(math.cos, 1)
        pset.addPrimitive(math.sin, 1)

    for i in range(n_features):
        pset.renameArguments(**{f"ARG{i}": f"x{i}"})

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def eval_symb_reg(individual, X, y):
        func = toolbox.compile(expr=individual)
        diff = np.sum([(func(*row) - truth)**2 for row, truth in zip(X, y)])
        return (diff / len(X),)

    toolbox.register("evaluate", eval_symb_reg, X=X, y=y)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    start_time = time.time()
    algorithms.eaSimple(pop, toolbox, 0.5, 0.1, generations, stats=stats, halloffame=hof, verbose=False)
    end_time = time.time()
    
    final_fitness = hof[0].fitness.values[0] if hof else -1.0
    
    params = {
        'n_samples': n_samples, 'n_features': n_features,
        'population_size': pop_size, 'generations': generations
    }
    logger.log('deap', task, params, end_time - start_time, final_fitness)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--n_samples', type=int, required=True)
    parser.add_argument('--n_features', type=int, required=True)
    parser.add_argument('--pop_size', type=int, required=True)
    parser.add_argument('--generations', type=int, required=True)
    args = parser.parse_args()

    from deap import algorithms
    run_deap_benchmark(args.task, args.n_samples, args.n_features, args.pop_size, args.generations)