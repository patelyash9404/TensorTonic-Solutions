import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a = np.asarray(a,dtype=float)
    b = np.asarray(b,dtype=float)
    dot_product = np.dot(a, b)
    norm_x = np.linalg.norm(a)
    norm_y = np.linalg.norm(b)
    if norm_x == 0 or norm_y == 0:
        return 0.0      
    return float(dot_product / (norm_x * norm_y))