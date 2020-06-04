# Directional Hadamard derivative of the Wasserstein distance for finite measures. See theorem 4 in [1].

### Author: Thomas Giacomo Nies

#[1] Sommerfeld, Max & Munk, Axel. (2016). Inference for Empirical Wasserstein Distances on Finite Spaces. Journal of the Royal Statistical Society. Series B: Statistical Methodology. 10.1111/rssb.12236.

import numpy as np
import ot
from scipy import optimize
from scipy.spatial.distance import cdist

def Wasserstein_directional_diff(r, s, a, b, x, y, p, metric="euclidean"):
    """
    Hadamard derivative of the funcitonal (a, b) -> W^p_p(a, b) at (a, b) int he direction (r,s).

    """

    n = len(a)
    m = len(b)
    d = cdist(x.reshape((n, 1)), y.reshape((m, 1)), metric=metric)
    c = d**p
    pi = ot.emd(a, b, c)
    pi_is_zero = np.invert(np.isclose(pi, 0, rtol=1e-05, atol=1e-08, equal_nan=False))
    pi_is_zero = np.ndarray.flatten(pi_is_zero)    # matrix if converted into n*m long vector. 
    
    
    # Matrix for complementary slack condition f_i + g_j = p_ij * c_ij. 
    # Here Mx = c with x[i] = f_i and x[n+j] = g_j.
    A_eq = np.zeros((n*m,n+m))
    for ij in range(n*m):
        i, j = np.unravel_index(ij, shape=(n,m))
        A_eq[ij, i] = 1
        A_eq[ij, n + j] = 1
    
    A_eq = A_eq[pi_is_zero]
    b_eq = np.ndarray.flatten(c)
    b_eq = b_eq[pi_is_zero]
    obj = np.ones(n+m)
    
    res = optimize.linprog(obj, A_ub=None, b_ub=None, A_eq=A_eq, b_eq=b_eq, bounds=None, method='interior-point', callback=None, options=None, x0=None)
    x = -1 * res["x"]
    u = x[0:n]
    v = x[n:n+m]
    
    # check:
    # print(np.sum(c*pi), np.sum(u/n)+np.sum(v/m))
    return u, v

if __name__ == "__main__":
    n = 10 
    m = 5 

    # p-Wasserstein 
    p = 2
    
    # Positions
    x = np.random.random(n)
    y = np.random.random(m)
    
    
    # Probability values 
    a = np.ones(n)/n
    b = np.ones(m)/m

    # Some direction with respect to which we differentiate.
    r = np.zeros(n)
    r[0] = 1
    s = np.zeros(m)
    s[0] = 1

    print(Wasserstein_directional_diff(r, s, a, b, x, y, p))
