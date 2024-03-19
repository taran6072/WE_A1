import numpy as np
from scipy.special import bernoulli


def gamma_error_1(n_max):
    
    """
    Compute the first-order gamma error E_n for 1 <= n <= n_max.
    
    Args:
        n_max (int)
        
    Returns:
         Array of length n_max containing the values of E_n.
    """
    # Euler-Mascheroni constant
    gamma = 0.5772156649015329
    # Computing the Harmonic numbers
    H_n = np.cumsum(1 / np.arange(1, n_max + 1))
    # Computing E_n using the given formula
    E_n = H_n - np.log(np.arange(1, n_max + 1)) - gamma
    return E_n


def gamma_error_2(n_max):
     """
    Computing the second-order gamma error E_2_n for 1 <= n <= n_max.
    
    Args:
        n_max (int) 
        
    Returns:
      Array of length n_max containing the values of E_2_n.
    """
    # Euler-Mascheroni constant    
    gamma = 0.5772156649015329
    H_n = np.cumsum(1 / np.arange(1, n_max + 1))
    # Compute 1/n
    inverse_n = 1 / np.arange(1, n_max + 1)
    # Compute E_2_n using  the given formula
    E_2_n = H_n - np.log(np.arange(1, n_max + 1)) - gamma - 0.5 * inverse_n
    return E_2_n


def gamma_error(k, n_max):
    
    """
    Compute the kth-order gamma error E_k_n for 1 <= n <= n_max.

    Parameters:
    k (int): Order of the error.
    n_max(int): Maximum value of n.

    Returns:
      Array of length n_max containing the values of E_k_n.
    """

    # Euler-Mascheroni constant
    gamma = 0.5772156649015329
    
    # Computing the Harmonic numbers
    H_n = np.cumsum(1 / np.arange(1, n_max + 1))
    inverse_n = 1 / np.arange(1, n_max + 1)
    # Compute Bernoulli numbers up to 2*k
   
    bernoulli_coeffs = bernoulli(2*k)
    # Initializing E_k_n with E_1_n
    E_k_n = H_n - np.log(np.arange(1, n_max + 1)) - gamma
    # Adding higher-order corrections
    for j in range(1, k+1):
        E_k_n += bernoulli_coeffs[2*j] / (j * np.power(np.arange(1, n_max + 1), j))
    return E_k_n
