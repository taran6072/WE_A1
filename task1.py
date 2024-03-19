import numpy as np
import math

# Replace the "pass" statements with your code for each function.
# Remember to write docstrings!

def square(n):
    """
    Computing the number of marbles in a square of side length n.

    Args:
    n (int): The side length of the square

    """
    s_n = n**2
    return s_n


def triangle(n):
    """
    Computes the number of marbles in a triangular arrangement with n rows.

    Args:
    n : The number of rows in the triangle.

    Returns:
    int: The total number of marbles in the triangular arrangement.
    """

    t_n = int(n*(n+1)/2)
    return t_n


def brute_force_search(m_max):
    """
    Finding all pairs [m, n] such that S(m) = T(n), where m ≤ m_max.
    """
    solutions = []

    # Handling special cases where m = 0 or m = 1, as given in question
    if m_max >= 0:
        solutions.append([0, 0])

    if m_max >= 1:
        solutions.append([1, 1])

    # Iterating through possible values of m and n
    for m in range(2, m_max + 1):
        s_m = square(m)
        for n in range(m, int(math.sqrt(2) * m) + 1):
            t_n = triangle(n)
            if s_m == t_n and m != n:  # We exclude trivial cases where m = n
                solutions.append([m, n])
    return solutions


def floor_root(n):
    """
    Following function computes the floor square root of an integer or array of integers.
    """
    # As given in the question to ensure proper rounding
    return np.floor(np.sqrt(n + 0.5)).astype(int)


def is_square(n):
    """
    This function checks if an integer or an array of integers is a perfect square.
    """
    # Compute the square of the floor square root
    root = floor_root(n)
    # Compares corresponding elements of two arrays (root * root and n) and returns a boolean array
    # where each element indicates whether the corresponding elements are equal or not.
    return np.equal(root * root, n)


def triangle_search(m_max):
    """
    Finds all pairs [m, n] such that 1 + 8m^2 is a perfect square, where m ≤ m_max.

    Args:
    - m_max (int): The maximum value of m to search for.

    Returns:
    - list: A list of lists containing all solution pairs [m, n].
    """
    # Generate candidate values of m
    m_values = np.arange(0, m_max + 1)

    # Compute 1 + 8m^2
    candidate = 1 + 8 * m_values ** 2

    # Check if each candidate is a perfect square
    perfect_square_mask = is_square(candidate)

    # Filter valid m values
    valid_m_values = m_values[perfect_square_mask]

    # Compute n values for valid m values
    valid_n_values = (np.sqrt(1 + 8 * valid_m_values ** 2) - 1) // 2

    # Combine valid m and n values into pairs
    solutions = np.column_stack((valid_m_values, valid_n_values)).astype(int)

    return solutions.tolist()


def matrix_solve(A, B):

    """
    Solves the linear equation XA = Y for X, as given in question

    Args:
    - A, B : NumPy arrays representing matrices of size nxn

    Returns:
    - X - Numpy array of size nxn

    """

    A_inv = np.linalg.inv(A)

    # Solving for X using the equation X = YA^(-1)
    X = np.dot(B, A_inv)

    return X


def triquadrigon(k, a=1, b=1, c=1, d=0):
    """
    Computes the kth pair of integers m_k, n_k such that S(m_k) = T(n_k).

    Args:
    k(int): The index of the pair to compute.
    

    Returns:
    tuple: The kth pair (m_k, n_k).
    """
    # Initialize the first two pairs
    pairs = [(0, 0), (1, 1)]

    # Compute the remaining pairs up to k
    for i in range(2, k + 1):
        m_prev, n_prev = pairs[-1]
        m_prev2, n_prev2 = pairs[-2]

        m_k = a * m_prev + b * n_prev - m_prev2
        n_k = c * m_prev + d * n_prev - n_prev2

        pairs.append((m_k, n_k))

    return pairs[k]
