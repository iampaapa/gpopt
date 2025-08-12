import argparse
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import gpopt
from benchmarks.datasets import get_dataset
from benchmarks.logging_utils import BenchmarkLogger

def main(args):
    logger = BenchmarkLogger()
    X, y = get_dataset(args.task, args.n_samples, args.n_features)
    
    config = gpopt.GpoptConfig()
    config.population_size = args.pop_size
    config.generations = args.generations
    config.num_features = args.n_features
    config.num_samples = args.n_samples
    config.function_set = [gpopt.Op.ADD, gpopt.Op.SUB, gpopt.Op.MUL]

    runner = gpopt.GpoptRunner(config)

    start_time = time.time()
    runner.initialize(X.flatten().tolist(), y.tolist())
    runner.run()
    end_time = time.time()
    
    best_ind = runner.get_best_individual()
    
    params = {
        'n_samples': args.n_samples, 'n_features': args.n_features,
        'population_size': args.pop_size, 'generations': args.generations
    }
    logger.log('gpopt', args.task, params, end_time - start_time, best_ind.fitness)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--n_samples', type=int, required=True)
    parser.add_argument('--n_features', type=int, required=True)
    parser.add_argument('--pop_size', type=int, required=True)
    parser.add_argument('--generations', type=int, required=True)
    args = parser.parse_args()
    main(args)