import numpy as np

def trapz ( y, x):    
    order = np.argsort(x)
    return np.trapz(y[order], x[order])