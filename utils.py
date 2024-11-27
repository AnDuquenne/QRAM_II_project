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


def extend_cov_matrix(matrix: np.array) -> np.array:
    m_ = np.vstack([matrix, np.zeros(matrix.shape[1])])
    m_ = np.column_stack([m_, np.zeros(m_.shape[0])])
    m_[-1, -1] = 1
    return m_

def extend_vol_vector(vector: np.array) -> np.array:
    return np.append(vector, 0)

def print_yellow(text):
    print(f"\033[93m{text}\033[00m")

def print_green(text):
    print(f"\033[92m{text}\033[00m")

def print_red(text):
    print(f"\033[91m{text}\033[00m")

def print_blue(text):
    print(f"\033[94m{text}\033[00m")

def print_purple(text):
    print(f"\033[95m{text}\033[00m")

def remove_redundent_rows(data_):
    data = data_.copy(deep=True)
    data = data.reset_index(drop=True)
    set_index_removed = []
    for i in range(1, data.shape[0]):
        for j in range(data.shape[1]):
            if data.iloc[i, j] != 0:
                for k in range(i):
                    if data.iloc[k, j] != 0:
                        # rows to remove
                        print_red(f"{k}, {j}")
                        print_red(data.iloc[k, j])
                        set_index_removed.append(k)

    # remove duplicates in the set_index_removed list
    print_red(set_index_removed)
    set_index_removed = list(set(set_index_removed))
    print_red(set_index_removed)
    # remove duplicates
    for i_ in set_index_removed:
        data = data.drop(i_)

    data = data.reset_index(drop=True)

    return data, set_index_removed