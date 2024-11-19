import numpy as np

def compute_return_365(r, days):
    R_365 = (1 + r) ** (365 / days) - 1
    return R_365

def compute_return_252(r, days):
    R_252 = (1 + r) ** (252 / days) - 1
    return R_252

# A function to compute the daily return from the annual return
def compute_return_daily(r, days=252):
    R_daily = (1 + r) ** (1 / days) - 1
    return R_daily

def split_vector(vector):
    """
    Splits a numpy vector into a set of numpy vectors where each non-zero
    element is placed in its own vector, and all other elements are zero.

    Parameters:
        vector (numpy.ndarray): The input vector.

    Returns:
        list of numpy.ndarray: A list of numpy vectors.
    """
    vectors = []
    for i, value in enumerate(vector):
        if value != 0:
            new_vector = np.zeros_like(vector)
            new_vector[i] = value
            vectors.append(new_vector)
    return vectors