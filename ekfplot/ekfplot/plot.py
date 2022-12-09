import numpy as np
import matplotlib.pyplot as plt

def errorbar ( x, y, xlow=None, xhigh=None, ylow=None, yhigh=None, ax=None, **kwargs ):
    if ax is None:
        ax = plt.subplot(111)
        
    if xlow is None:
        xlow = np.NaN
    if xhigh is None:
        xhigh = np.NaN
    if ylow is None:
        ylow = np.NaN
    if yhigh is None:
        yhigh = np.NaN
        
    if 'fmt' not in kwargs.keys():
        kwargs['fmt'] = 'o'
    
    xerr =  np.array([[xhigh - x], [x-xlow]]).reshape(2,-1)    
    yerr =  np.array([[yhigh - y], [y-ylow]]).reshape(2,-1)
    
    ax.errorbar ( x, 
                  y, 
                  xerr = xerr,
                  yerr = yerr,
                  **kwargs
                )
    return ax