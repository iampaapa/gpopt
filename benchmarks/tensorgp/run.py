import argparse
import sys
import os
import time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tensorgp.engine import Engine, tensor_rmse
from benchmarks.datasets import get_dataset
from benchmarks.logging_utils import BenchmarkLogger


def fitness_function(**kwargs):
    """
    A custom fitness function required by the TensorGP Engine.
    """
    tensors = kwargs.get('tensors')
    target = kwargs.get('target')
    population = kwargs.get('population')
    
    for i in range(len(tensors)):
        fit = tensor_rmse(tensors[i], target).numpy()
        population[i]['fitness'] = fit
        
    best_idx = np.argmin([ind['fitness'] for ind in population])
    
    return population, best_idx


def main(args):
    logger = BenchmarkLogger()
    X, y = get_dataset(args.task, args.n_samples, args.n_features)

    y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    target_dimensions = list(y_tensor.shape)
    
    # use the device argument for the engine, converting 'gpu' to '/gpu:0'
    device_str = '/gpu:0' if args.device == 'gpu' else '/cpu:0'

    engine = Engine(
        fitness_func=fitness_function,
        population_size=args.pop_size,
        tournament_size=3,
        mutation_rate=0.2,
        crossover_rate=0.8,
        max_tree_depth=12,
        target_dims=target_dimensions,
        target=y_tensor,
        elitism=1,
        max_init_depth=8,
        objective='minimizing',
        device=device_str,
        stop_criteria='generation',
        stop_value=args.generations,
        effective_dims=args.n_features,
        operators=['add', 'sub', 'mult', 'div'],
        seed=42,
        save_graphics=False,
        show_graphics=False
    )
    
    start_time = time.time()
    engine.run()
    end_time = time.time()

    if engine.population:
        best_individual = min(engine.population, key=lambda x: x['fitness'])
        best_fitness = best_individual['fitness']
    else:
        best_fitness = -1.0
    
    params = {
        'n_samples': args.n_samples, 'n_features': args.n_features,
        'population_size': args.pop_size, 'generations': args.generations
    }
    
    framework_log_name = f"tensorgp_{args.device}"
    logger.log(framework_log_name, args.task, params, end_time - start_time, best_fitness)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--n_samples', type=int, required=True)
    parser.add_argument('--n_features', type=int, required=True)
    parser.add_argument('--pop_size', type=int, required=True)
    parser.add_argument('--generations', type=int, required=True)
    parser.add_argument('--device', type=str, required=True, choices=['cpu', 'gpu'])
    args = parser.parse_args()
    main(args)