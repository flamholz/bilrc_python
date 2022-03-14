import numpy as np
from scipy.optimize import differential_evolution


"""fitting.py: code for fitting single or multi-exponential models to lifetime spectroscopy data.

Code snippet lightly modified from Richard Schwarzl's Github project:
https://gitlabph.physik.fu-berlin.de/rschwarz/MultiExponentialFitting/
Retrieved March 8th 2022
"""

__author__      = "Avi I. Flamholz"


def fit_multi_exponential(n, x, y):
    """
    Fits a multi-exponential decay by scipy.optimize.differential_evolution

    Parameters
    ----------
    n : integer
        number of exponentials.
    x : (n) array_like
        1-dimensional list of x-values.
    y : (n), array_like
        1-dimensional list of y-values.
        
    Returns
    -------
    a : (N) array
        solution prefactors in order a1, a2, ..., aN.
    b : (N) array
        solution exponents (positive) in order b1, b2, ..., bN.
    red_chi_sq : float
        reduced chi squared as calculated by the sum of squared residuals
    """
    x = np.asarray(x)
    y = np.asarray(y)

    bounds = [[min(x), max(x)]]*n + [[min(y), max(y)]]*n

    def objective(s):
        taui, fi = np.split(s, 2)
        return np.sum((y - np.dot(fi, np.exp(-np.outer(1./taui, x))))**2.)

    result = differential_evolution(objective, bounds)
    s = result['x']
    red_chi_sq = objective(s)/(len(x)-len(s))
    taui, fi = np.split(s, 2)
    return fi, 1./taui, red_chi_sq


def eval_exp_fit(x, a, b):
    """
    Is used to evaluate multi-exponential function for parameter lists a, b

    Parameters
    ----------
    x : (n) array_like
        1-dimensional list of x-values.
    a : (N) array_like
        1-dimensional list of prefactors.
    b : (N) array_like
        1-dimensional list of exponents (positive).
    Returns
    -------
    y : (n) array
        y-values of the multi-exponential function corresponding to the x-values put in.
    """
    return np.dot(a, np.exp(-np.outer(b, x)))
