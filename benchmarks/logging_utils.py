import csv
import os
from threading import Lock

class BenchmarkLogger:
    def __init__(self, filename='benchmark_results.csv'):
        self.filename = filename
        self.lock = Lock()
        self._initialize_file()

    def _initialize_file(self):
        if not os.path.exists(self.filename):
            with self.lock:
                with open(self.filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'framework', 'task_name', 'n_samples', 'n_features',
                        'population_size', 'generations', 'run_time', 'final_fitness'
                    ])

    def log(self, framework, task_name, params, run_time, final_fitness):
        with self.lock:
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    framework, task_name, params['n_samples'], params['n_features'],
                    params['population_size'], params['generations'],
                    run_time, final_fitness
                ])