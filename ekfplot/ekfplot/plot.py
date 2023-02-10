import numpy as np
import matplotlib.pyplot as plt

def midpoints ( x ):
    return 0.5*(x[:-1]+x[1:])

def errorbar ( x, y, xlow=None, xhigh=None, ylow=None, yhigh=None, ax=None, c=None, zorder=9,
               scatter_kwargs={}, **kwargs ):
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
    if np.isnan(xerr).all():
        xerr = None
    if np.isnan(yerr).all():
        yerr = None
    
    
    if c is None:
        ax.errorbar ( x, 
                    y, 
                    xerr = xerr,
                    yerr = yerr,
                    **kwargs
                    )
    else:
        kwargs['markersize'] = 0
        ax.errorbar ( x, 
                    y, 
                    xerr = xerr,
                    yerr = yerr,
                    zorder=zorder,
                    **kwargs
                    )
        im = ax.scatter ( x, y, c=c, zorder=zorder+1, **scatter_kwargs )
        return ax, im
    return ax

def c_density ( x, y, return_fn=False, **kwargs ):
    from scipy.stats import gaussian_kde
    # Calculate the point density    
    xy = np.vstack([x,y])
    fn = gaussian_kde(xy, **kwargs)
    if return_fn:
        return fn
    else:
        z = fn(xy)
        return z    

def density_scatter ( x, y, cmap='Greys', ax=None, **kwargs ):
    if ax is None:
        ax = plt.subplot(111)
    z = c_density(x,y)
    im = ax.scatter ( x, y, c=z, cmap=cmap, **kwargs )
    return ax, im

    