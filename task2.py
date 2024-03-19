import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


def create_hills(x, params):
    """
    We first Generate the altitude of the hill chain for given x values and parameters representing Gaussian hills.

    Parameters:
    - x: numpy array, x values where to evaluate the altitude
    - params:
        params[0] contains height of hills,
        params[1] location of peaks ,
        params[2] contains breadth of hills.

    Returns:
    - r: numpy array, altitude of the hill chain corresponding to each x value.
    """
    
    # Parametres 
    H, mu, sigma = params
    #  Number of hills
    M = len(H)
    
    # initilaizing empty altitude array
    r = np.zeros_like(x)
    
    # Calculating the altitude for each hill
    for i in range(M):
        # Gaussian function for each hill 
        r += H[i] / (sigma[i] * np.sqrt(2 * np.pi)) * np.exp(-((x - mu[i])**2) / (2 * sigma[i]**2))
    return r


def plot_hills(x, r, snow_depth=None):
    '''
    Creating a plot of a given hill range, using the points
    given in x.
    Returns Figure and Axes objects containing the plot.
    '''
    fig, ax = plt.subplots(figsize=(6, 4))
    
    
     # Ploting hill chain altitude
    plt.plot(x, r, label='Hill Chain Altitude', color='green')
    if snow_depth is not None:
        S = r + snow_depth
        plt.fill_between(x, r, S, color='blue', alpha=0.3, label='Snow Surface')
    plt.xlabel('Distance along the hill chain (m)')
    plt.ylabel('Altitude(m)')
    plt.title('Altitude of Hill Chain with Snow Surface')
    plt.legend()
    plt.grid(True)
    
    return fig, ax




def estimate_snowfall(snow_depth, params, L, method):
    """
    Estimate the volume of snowfall V(h) using the specified method.
    
    Parameters:
    - snow_depth: numpy array, measurements of snow depth s(x) equally spaced over [0, L] (including both endpoints)
    - params: numpy array of shape (3, M), where M is the number of hills.  (Same as defined above)
           
    - L: float, total length of the hill range in metres
    - method: string, either 'riemann_left' or 'trapezoid'

    Returns:
    - V:  volume of snowfall V(h) obtained using the given method.
    
    """
    
    #  it generates an array which represents the locations along the hill chain where the snow depth measurements were taken 
    x = np.linspace(0, L, len(snow_depth))
     # Now, we calculate the altitude of the hill chain
    r = create_hills(x, params)
    # Then, calculate the width of each subinterval
    h = x[1] - x[0]
    
    # Calculating the snow surface
    # For Riemann left method, we exclude the last measurement
    if method == 'riemann_left':
        S = r[:-1] + snow_depth[:-1]
        # For trapezoid method, we use all measurements
    elif method == 'trapezoid':
        S = 0.5*(r + snow_depth)[:-1] + 0.5*(r + snow_depth)[1:]
    else:
        raise ValueError("Method must be either 'riemann_left' or 'trapezoid'")
     # Calculating the volume of snowfall
    V_S = h * np.sum(S)
     # Calculating the volume of the hills
    V_r = np.sum([H/2 * (erf((b - mu)/(sig*np.sqrt(2))) - erf((a - mu)/(sig*np.sqrt(2)))) for H, mu, sig in params.T for a, b in zip(x[:-1], x[1:])])
    # Returns the difference
    
    return V_S - V_r




def spaced_estimates(snow_depth, params, L):
    """
    Calculate snowfall estimates using every 2^kth measurement available, for k = 0 to 6.

    Parameters:
    - snow_depth: numpy array, measurements of snow depth equally spaced over [0, L] (including both endpoints)
    - params: numpy array of shape (3, M), containing the values of H_i, mu_i, and sigma_i parameters
    - L: float, total length of the hill range in meters

    Returns:
    - estimates: numpy array with 2 rows and 7 columns, containing snowfall estimates using left Riemann sum and trapezoid rule
    """
    # Calculate the width of the smallest interval between consecutive measurements
    h = L / (len(snow_depth) - 1)

    # Initialize an empty array to store the estimates
    estimates = np.zeros((2, 7))

    # Iterate over k values from 0 to 6
    for k in range(7):
        # Determine the indices of measurements to use (every 2^kth measurement)
        indices = np.arange(0, len(snow_depth), 2**k)
        
        # Calculate snowfall estimates using left Riemann sum and trapezoid rule for the selected indices
        snowfall_riem = estimate_snowfall(snow_depth[indices], params, L, 'riemann_left')
        snowfall_trapz = estimate_snowfall(snow_depth[indices], params, L, 'trapezoid')
        
        # Store the estimates in the array
        estimates[0, k] = snowfall_riem
        estimates[1, k] = snowfall_trapz

    return estimates


def min_points(params, L, eps):
    H, mu, sigma = params

    # Calculating  maximum curvature of the hill chain
    max_curvature = np.max(2 * H / (np.sqrt(2 * np.pi) * sigma ** 3))

    # Calculating lower bound on the total number of equally spaced measurements
    Nmin = np.ceil(np.sqrt((L ** 3 * max_curvature) / (12 * eps)))
    
    return int(Nmin)
