import argparse
import sys
import os
import time
import random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from benchmarks.datasets import get_dataset
from benchmarks.logging_utils import BenchmarkLogger

# --- Naive GP Implementation ---
OPS = ['ADD', 'SUB', 'MUL']
VAR = []

def eval_program(program, sample):
    stack = []
    for token in reversed(program):
        if token == 'ADD':
            if len(stack) < 2: return 1e6
            stack.append(stack.pop() + stack.pop())
        elif token == 'SUB':
            if len(stack) < 2: return 1e6
            stack.append(stack.pop() - stack.pop())
        elif token == 'MUL':
            if len(stack) < 2: return 1e6
            stack.append(stack.pop() * stack.pop())
        elif token.startswith('VAR'):
            idx = int(token[3:])
            stack.append(sample[idx])
    return stack[0] if stack else 1e6

def fitness(program, X, y):
    error = 0.0
    for i in range(len(X)):
        try:
            pred = eval_program(program, X[i])
            error += (pred - y[i]) ** 2
        except (IndexError, TypeError):
            error += 1e6
    return error / len(X)

def random_program(length=15):
    program = []
    for _ in range(length):
        if random.random() < 0.4 or len(program) < 2:
             program.append(random.choice(VAR))
        else:
            program.append(random.choice(OPS))
    return program

def mutate(program):
    idx = random.randint(0, len(program) - 1)
    if program[idx].startswith('VAR'):
        program[idx] = random.choice(VAR)
    else:
        program[idx] = random.choice(OPS)

def crossover(p1, p2):
    point = random.randint(1, min(len(p1), len(p2)) - 1)
    return p1[:point] + p2[point:]

def main(args):
    global VAR
    logger = BenchmarkLogger()
    X, y = get_dataset(args.task, args.n_samples, args.n_features)
    VAR = [f'VAR{i}' for i in range(args.n_features)]

    start_time = time.time()
    
    population = [random_program() for _ in range(args.pop_size)]
    
    for gen in range(args.generations):
        fitnesses = [fitness(ind, X, y) for ind in population]
        
        sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1])
        population = [ind for ind, fit in sorted_population]

        parents = population[:int(args.pop_size * 0.1)]
        new_pop = parents[:]
        
        while len(new_pop) < args.pop_size:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            if random.random() < 0.3:
                mutate(child)
            new_pop.append(child)
        population = new_pop

    end_time = time.time()
    
    best_program = population[0]
    best_fitness = fitness(best_program, X, y)

    params = {
        'n_samples': args.n_samples, 'n_features': args.n_features,
        'population_size': args.pop_size, 'generations': args.generations
    }
    logger.log('naive', args.task, params, end_time - start_time, best_fitness)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--n_samples', type=int, required=True)
    parser.add_argument('--n_features', type=int, required=True)
    parser.add_argument('--pop_size', type=int, required=True)
    parser.add_argument('--generations', type=int, required=True)
    args = parser.parse_args()
    main(args)