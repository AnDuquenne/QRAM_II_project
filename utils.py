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


def binary_transform(array):
    """
    Transforms a numpy array into a binary array where:
    - 0 becomes 0
    - Non-zero values become 1

    Parameters:
        array (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: The binary-transformed array.
    """
    return (array != 0).astype(np.float64)


def combine_vectors(vectors):
    return np.sum(vectors, axis=0)


def rowwise_multiply(matrix, vector):
    """
    Multiplies each row of a matrix by the corresponding element of a vector.

    Parameters:
        matrix (numpy.ndarray): A 2D array (matrix).
        vector (numpy.ndarray): A 1D array (vector) with the same number of rows as the matrix.

    Returns:
        numpy.ndarray: A matrix where each row is multiplied by the corresponding vector element.
    """
    if matrix.shape[0] != vector.shape[0]:
        raise ValueError("The number of rows in the matrix must match the length of the vector.")

    return matrix * vector[:, np.newaxis]