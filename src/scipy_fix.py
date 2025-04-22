# scipy_fix.py
import numpy as np

class KDTree:
    def __init__(self, points):
        self.points = np.asarray(points)
        self.n = len(points)
        
    def query_pairs(self, r):
        """Optimized pairwise distance check using numpy broadcasting"""
        if self.n == 0:
            return set()
        
        # Use efficient numpy operations
        diffs = self.points[:, np.newaxis] - self.points
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        pairs = np.argwhere((dists <= r) & (dists > 0))
        
        # Convert to set of unique pairs
        return set(tuple(sorted(pair)) for pair in pairs if pair[0] < pair[1])