import numpy as np

def gini(x, w=None, qt_cut=0.):
    x = x.flatten()
    x = x[x>np.nanquantile(x, qt_cut)]
    sorted_x = np.sort(x)
    n = len(x)
    indices = np.arange(1,1+n)
    interior = np.sum( (2.*indices - n - 1.)*abs(sorted_x) ) 
    factor = (np.mean(abs(x)) * n * (n-1.) )**-1
    return interior * factor
    #cumx = np.cumsum(sorted_x, dtype=float)
    # The above formula, with all weights equal to 1 simplifies to:
    #return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    