import numpy as np
import matplotlib.pyplot as plt

def errorbar ( x, y, ylow, yhigh, ax=None ):
    if ax is None:
        ax = plt.subplot(111)
    
    yerr =  np.array([[yhigh - y], [y-ylow]]).reshape(2,-1)
    ax.errorbar ( x, 
                  y, 
                  yerr = yerr 
                )
    return ax