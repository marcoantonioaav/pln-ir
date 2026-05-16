import numpy as np
import time
import psutil
import os
from typing import List, Dict

def calculate_recall_at_k(found_indices: List[int], ground_truth_indices: List[int], k_values: List[int] = [1, 10, 25]) -> Dict[int, float]:
    """
    Calculates Recall@K. Since this is for the candidate set initialization,
    we check how many of the top-K ground truth neighbors are present in the top-K found indices.
    """
    recalls = {}
    for k in k_values:
        if len(found_indices) == 0:
            recalls[k] = 0.0
            continue
            
        # We take the minimum to avoid index errors if the approach returns fewer than k points
        k_f = min(k, len(found_indices))
        k_gt = min(k, len(ground_truth_indices))
        
        if k_gt == 0:
            recalls[k] = 0.0
            continue
            
        intersection = len(set(found_indices[:k_f]).intersection(set(ground_truth_indices[:k_gt])))
        recalls[k] = intersection / k
    return recalls

def calculate_mean_distance(found_indices: List[int], query: np.ndarray, dataset: np.ndarray) -> float:
    """Calculates mean distance of found points to the query."""
    if not found_indices:
        return 0.0
    distances = [np.linalg.norm(dataset[idx] - query) for idx in found_indices]
    return float(np.mean(distances))

def calculate_1nn_distance_diff(found_1nn_idx: int, gt_1nn_idx: int, query: np.ndarray, dataset: np.ndarray) -> float:
    """Calculates the difference between the distance of the found 1-NN and the ground truth 1-NN to the query."""
    found_dist = np.linalg.norm(dataset[found_1nn_idx] - query)
    gt_dist = np.linalg.norm(dataset[gt_1nn_idx] - query)
    return float(found_dist - gt_dist)

class PerformanceTracker:
    """Tracks time for build and search operations."""
    def __init__(self):
        self.start_time = 0.0

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Returns elapsed time in seconds."""
        return time.perf_counter() - self.start_time
