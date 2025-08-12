import subprocess

FRAMEWORKS_TO_RUN = [
    'naive',
    'gpopt',
    'gplearn',
    'deap',
    'tensorgp'
]

TENSORGP_DEVICES = ['cpu', 'gpu']

TASKS = ['simple', 'medium', 'quartic']
N_SAMPLES = [100, 1000, 10000]
N_FEATURES = [2, 5]
POPULATION_SIZES = [100, 500]
GENERATIONS = [50]

def main():
    for framework in FRAMEWORKS_TO_RUN:
        print(f"--- Running Benchmarks for: {framework.upper()} ---")

        if framework == 'tensorgp':
            devices_to_run = TENSORGP_DEVICES
        else:
            devices_to_run = ['default']

        for device in devices_to_run:
            for task in TASKS:
                for n_samples in N_SAMPLES:
                    for n_features in N_FEATURES:
                        for pop_size in POPULATION_SIZES:
                            for generations in GENERATIONS:
                                if framework == 'tensorgp' and n_features != 2:
                                    print(f"  Skipping Task: {task} for TensorGP with {n_features} features (not supported).")
                                    continue

                                if task in ['quartic', 'trigonometric'] and n_features > 2:
                                    continue

                                framework_log_name = f"{framework}_{device}" if framework == 'tensorgp' else framework
                                print(f"  Framework: {framework_log_name}, Task: {task}, Samples: {n_samples}, Features: {n_features}, Pop: {pop_size}, Gens: {generations}")
                                
                                cmd = [
                                    'python', f'benchmarks/{framework}/run.py',
                                    '--task', task,
                                    '--n_samples', str(n_samples),
                                    '--n_features', str(n_features),
                                    '--pop_size', str(pop_size),
                                    '--generations', str(generations)
                                ]
                                
                                if framework == 'tensorgp':
                                    cmd.extend(['--device', device])
                                
                                subprocess.run(cmd)
            print("-" * 40 + "\n")

if __name__ == "__main__":
    main()