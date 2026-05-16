import numpy as np

class DatasetGenerator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(self, num_samples: int, num_dimensions: int, num_queries: int):
        """
        Generates non-clustered random dataset and in-distribution queries.
        Uses a uniform distribution for simplicity.
        
        Args:
            num_samples: Number of points in the dataset
            num_dimensions: Dimensionality of the points
            num_queries: Number of queries
        
        Returns:
            dataset: np.ndarray of shape (num_samples, num_dimensions)
            queries: np.ndarray of shape (num_queries, num_dimensions)
        """
        dataset = self.rng.uniform(size=(num_samples, num_dimensions)).astype(np.float32)
        queries = self.rng.uniform(size=(num_queries, num_dimensions)).astype(np.float32)
        return dataset, queries
