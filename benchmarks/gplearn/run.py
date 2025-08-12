import argparse
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gplearn.genetic import SymbolicRegressor
from benchmarks.datasets import get_dataset
from benchmarks.logging_utils import BenchmarkLogger

def main(args):
    logger = BenchmarkLogger()
    X, y = get_dataset(args.task, args.n_samples, args.n_features)

    est_gp = SymbolicRegressor(population_size=args.pop_size,
                               generations=args.generations,
                               function_set=('add', 'sub', 'mul'),
                               stopping_criteria=0.01,
                               verbose=0,
                               random_state=42)

    def _dummy_validate_data(X, y, **kwargs):
        return X, y
    
    est_gp._validate_data = _dummy_validate_data

    start_time = time.time()
    est_gp.fit(X, y)
    end_time = time.time()
    
    final_fitness = est_gp._program.raw_fitness_ if hasattr(est_gp, '_program') else -1.0

    params = {
        'n_samples': args.n_samples, 'n_features': args.n_features,
        'population_size': args.pop_size, 'generations': args.generations
    }
    logger.log('gplearn', args.task, params, end_time - start_time, final_fitness)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--n_samples', type=int, required=True)
    parser.add_argument('--n_features', type=int, required=True)
    parser.add_argument('--pop_size', type=int, required=True)
    parser.add_argument('--generations', type=int, required=True)
    args = parser.parse_args()
    main(args)