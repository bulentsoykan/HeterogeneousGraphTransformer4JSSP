"""
Benchmark JSSP Instances for Evaluation.

Includes:
- Classic small instances (FT06, FT10, FT20)
- Taillard instances
- Random instance generators
- Instance loaders

Reference:
- Taillard, E. (1993). Benchmarks for basic scheduling problems.
- Fisher & Thompson (1963). Industrial scheduling problems.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import json
import os


# Classic Fisher-Thompson instances
FT06 = {
    'name': 'FT06',
    'num_jobs': 6,
    'num_machines': 6,
    'processing_times': np.array([
        [1, 3, 6, 7, 3, 6],
        [8, 5, 10, 10, 10, 4],
        [5, 4, 8, 9, 1, 7],
        [5, 5, 5, 3, 8, 9],
        [9, 3, 5, 4, 3, 1],
        [3, 3, 9, 10, 4, 1],
    ]),
    'machine_assignment': np.array([
        [2, 0, 1, 3, 5, 4],
        [1, 2, 4, 5, 0, 3],
        [2, 3, 5, 0, 1, 4],
        [1, 0, 2, 3, 4, 5],
        [2, 1, 4, 5, 0, 3],
        [1, 3, 5, 0, 4, 2],
    ]),
    'optimal_makespan': 55,  # Known optimal
}

FT10 = {
    'name': 'FT10',
    'num_jobs': 10,
    'num_machines': 10,
    'processing_times': np.array([
        [29, 78, 9, 36, 49, 11, 62, 56, 44, 21],
        [43, 90, 75, 11, 69, 28, 46, 46, 72, 30],
        [91, 85, 39, 74, 90, 10, 12, 89, 45, 33],
        [81, 95, 71, 99, 9, 52, 85, 98, 22, 43],
        [14, 6, 22, 61, 26, 69, 21, 49, 72, 53],
        [84, 2, 52, 95, 48, 72, 47, 65, 6, 25],
        [46, 37, 61, 13, 32, 21, 32, 89, 30, 55],
        [31, 86, 46, 74, 32, 88, 19, 48, 36, 79],
        [76, 69, 76, 51, 85, 11, 40, 89, 26, 74],
        [85, 13, 61, 7, 64, 76, 47, 52, 90, 45],
    ]),
    'machine_assignment': np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 2, 4, 9, 3, 1, 6, 5, 7, 8],
        [1, 0, 3, 2, 8, 5, 7, 6, 9, 4],
        [1, 2, 0, 4, 6, 8, 7, 3, 9, 5],
        [2, 0, 1, 5, 3, 4, 8, 7, 9, 6],
        [2, 1, 5, 3, 8, 9, 0, 6, 4, 7],
        [1, 0, 3, 2, 6, 5, 9, 8, 7, 4],
        [2, 0, 1, 5, 4, 6, 8, 9, 7, 3],
        [0, 1, 3, 5, 2, 9, 6, 7, 4, 8],
        [1, 0, 2, 6, 8, 9, 5, 3, 4, 7],
    ]),
    'optimal_makespan': 930,  # Known optimal
}

FT20 = {
    'name': 'FT20',
    'num_jobs': 20,
    'num_machines': 5,
    'processing_times': np.array([
        [29, 9, 49, 62, 44],
        [43, 75, 69, 46, 72],
        [91, 39, 90, 12, 45],
        [81, 71, 9, 85, 22],
        [14, 22, 26, 21, 72],
        [84, 52, 48, 47, 6],
        [46, 61, 32, 32, 30],
        [31, 46, 32, 19, 36],
        [76, 76, 85, 40, 26],
        [85, 61, 64, 47, 90],
        [78, 36, 11, 56, 21],
        [90, 11, 28, 46, 30],
        [85, 74, 10, 89, 33],
        [95, 99, 52, 98, 43],
        [6, 61, 69, 49, 53],
        [2, 95, 72, 65, 25],
        [37, 13, 21, 89, 55],
        [86, 74, 88, 48, 79],
        [69, 51, 11, 89, 74],
        [13, 7, 76, 52, 45],
    ]),
    'machine_assignment': np.array([
        [0, 2, 1, 3, 4],
        [1, 0, 3, 2, 4],
        [0, 1, 4, 3, 2],
        [2, 0, 1, 4, 3],
        [1, 2, 0, 3, 4],
        [0, 2, 3, 1, 4],
        [1, 0, 2, 4, 3],
        [0, 2, 4, 1, 3],
        [2, 1, 0, 3, 4],
        [0, 1, 3, 2, 4],
        [1, 2, 0, 4, 3],
        [0, 3, 2, 1, 4],
        [2, 0, 1, 3, 4],
        [1, 0, 2, 4, 3],
        [0, 2, 1, 3, 4],
        [2, 1, 3, 0, 4],
        [0, 1, 2, 4, 3],
        [1, 0, 3, 2, 4],
        [0, 2, 1, 4, 3],
        [2, 0, 3, 1, 4],
    ]),
    'optimal_makespan': 1165,  # Known optimal
}


# Taillard instances (subset - small sizes for quick evaluation)
def generate_taillard_instance(size: str) -> Dict:
    """
    Generate Taillard-like instances.

    Args:
        size: One of '15x15', '20x15', '20x20', '30x15', '30x20', '50x15', '50x20', '100x20'

    Returns:
        Instance dictionary
    """
    size_map = {
        '15x15': (15, 15),
        '20x15': (20, 15),
        '20x20': (20, 20),
        '30x15': (30, 15),
        '30x20': (30, 20),
        '50x15': (50, 15),
        '50x20': (50, 20),
        '100x20': (100, 20),
    }

    if size not in size_map:
        raise ValueError(f"Unknown size: {size}. Choose from {list(size_map.keys())}")

    num_jobs, num_machines = size_map[size]

    # Generate random processing times (uniform 1-99)
    processing_times = np.random.randint(1, 100, size=(num_jobs, num_machines))

    # Generate random machine orderings
    machine_assignment = np.array([
        np.random.permutation(num_machines) for _ in range(num_jobs)
    ])

    return {
        'name': f'Taillard-{size}',
        'num_jobs': num_jobs,
        'num_machines': num_machines,
        'processing_times': processing_times,
        'machine_assignment': machine_assignment,
        'optimal_makespan': None,  # Unknown for random instances
    }


def generate_random_instance(
    num_jobs: int,
    num_machines: int,
    min_processing_time: int = 1,
    max_processing_time: int = 99,
    seed: Optional[int] = None
) -> Dict:
    """
    Generate random JSSP instance.

    Args:
        num_jobs: Number of jobs
        num_machines: Number of machines
        min_processing_time: Minimum processing time
        max_processing_time: Maximum processing time
        seed: Random seed for reproducibility

    Returns:
        Instance dictionary
    """
    if seed is not None:
        np.random.seed(seed)

    processing_times = np.random.randint(
        min_processing_time,
        max_processing_time + 1,
        size=(num_jobs, num_machines)
    )

    machine_assignment = np.array([
        np.random.permutation(num_machines) for _ in range(num_jobs)
    ])

    return {
        'name': f'Random-{num_jobs}x{num_machines}',
        'num_jobs': num_jobs,
        'num_machines': num_machines,
        'processing_times': processing_times,
        'machine_assignment': machine_assignment,
        'optimal_makespan': None,
    }


def get_benchmark_instance(name: str) -> Dict:
    """
    Get a benchmark instance by name.

    Args:
        name: Instance name (e.g., 'FT06', 'FT10', 'FT20')

    Returns:
        Instance dictionary
    """
    instances = {
        'FT06': FT06,
        'FT10': FT10,
        'FT20': FT20,
    }

    if name not in instances:
        raise ValueError(f"Unknown instance: {name}. Available: {list(instances.keys())}")

    return instances[name]


def save_instance(instance: Dict, filepath: str):
    """Save instance to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    instance_json = {
        'name': instance['name'],
        'num_jobs': int(instance['num_jobs']),
        'num_machines': int(instance['num_machines']),
        'processing_times': instance['processing_times'].tolist(),
        'machine_assignment': instance['machine_assignment'].tolist(),
        'optimal_makespan': instance.get('optimal_makespan'),
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(instance_json, f, indent=2)


def load_instance(filepath: str) -> Dict:
    """Load instance from JSON file."""
    with open(filepath, 'r') as f:
        instance_json = json.load(f)

    return {
        'name': instance_json['name'],
        'num_jobs': instance_json['num_jobs'],
        'num_machines': instance_json['num_machines'],
        'processing_times': np.array(instance_json['processing_times']),
        'machine_assignment': np.array(instance_json['machine_assignment']),
        'optimal_makespan': instance_json.get('optimal_makespan'),
    }


# Create a test suite of instances
def get_test_suite() -> Dict[str, Dict]:
    """
    Get a comprehensive test suite of instances.

    Returns:
        Dictionary mapping instance names to instance data
    """
    suite = {
        'FT06': FT06,
        'FT10': FT10,
        'FT20': FT20,
    }

    # Add small random instances for training
    for size in ['6x6', '10x10', '15x15']:
        num_jobs, num_machines = map(int, size.split('x'))
        suite[f'Random-{size}'] = generate_random_instance(
            num_jobs, num_machines, seed=42
        )

    return suite


if __name__ == '__main__':
    # Example usage
    print("Benchmark Instances:")
    print("=" * 60)

    for name, instance in [('FT06', FT06), ('FT10', FT10), ('FT20', FT20)]:
        print(f"\n{name}:")
        print(f"  Jobs: {instance['num_jobs']}, Machines: {instance['num_machines']}")
        print(f"  Optimal Makespan: {instance['optimal_makespan']}")

    # Generate and save some instances
    os.makedirs('benchmarks/data', exist_ok=True)

    for name in ['FT06', 'FT10', 'FT20']:
        instance = get_benchmark_instance(name)
        save_instance(instance, f'benchmarks/data/{name}.json')

    print("\nBenchmark instances saved to benchmarks/data/")
