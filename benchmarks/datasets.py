import numpy as np

def get_dataset(task_name, n_samples, n_features, seed=42):
    """
    Generates a dataset for a given symbolic regression task.
    """
    np.random.seed(seed)
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    
    if task_name == 'simple':
        # y = x0 + x1
        y = X[:, 0] + X[:, 1]
    elif task_name == 'medium':
        # y = x0 * x1 + 2*x1
        y = X[:, 0] * X[:, 1] + 2 * X[:, 1]
    elif task_name == 'quartic':
        # y = x^4 + x^3 + x^2 + x
        y = X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0]
    elif task_name == 'trigonometric':
        # y = sin(x0) + cos(x1)
        y = np.sin(X[:, 0]) + np.cos(X[:, 1])
    else:
        raise ValueError(f"Unknown task: {task_name}")
        
    return X, y.astype(np.float32)