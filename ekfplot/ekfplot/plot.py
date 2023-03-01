import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from ekfstats import functions

def midpoints ( x ):
    return 0.5*(x[:-1]+x[1:])

def adjust_font ( ax, fontsize=15 ):
    '''
    Change the fontsize of all lettering on a specific subplot. 
    From https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    '''
    items = [   ax.title, 
                ax.xaxis.label, 
                ax.yaxis.label, ]
    items = items + ax.get_xticklabels() + ax.get_yticklabels() 
    if ax.get_legend() is not None:
        items = items + ax.get_legend().get_texts ()

    for item in items:
        item.set_fontsize(fontsize)
    

def errorbar ( x, y, xlow=None, xhigh=None, ylow=None, yhigh=None, ax=None, c=None, zorder=9,
               scatter_kwargs={}, xsigma=1., ysigma=1., **kwargs ):
    '''
    Draw errorbars where errors are given by distribution quantiles
    '''
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

    xerr =  np.array([[x-xlow],[xhigh - x]]).reshape(2,-1) * xsigma
    yerr =  np.array([[y-ylow],[yhigh - y]]).reshape(2,-1) * ysigma
    if np.isnan(xerr).all():
        xerr = None
    if np.isnan(yerr).all():
        yerr = None

    if c is None:
        ax.errorbar (
                    x,
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
    '''
    Compute gKDE density based on sample
    '''
    # Calculate the point density
    if x.size < 10:
        return np.ones_like(x)
    xy = np.vstack([x,y])
    fn = gaussian_kde(xy, **kwargs)

    if return_fn:
        return fn
    else:
        z = fn(xy)
        return z
    
def density_contour (data_x,data_y, ax=None, npts=100, **kwargs):
    '''
    Draw a contour based on density
    '''
    if ax is None:        
        ax = plt.subplot(111)
    fmask = functions.finite_masker ( data_x, data_y )
    data_x = data_x[fmask]
    data_y = data_y[fmask]
    gkde = c_density ( data_x,data_y, return_fn=True )
    grid_x = np.linspace(data_x.min(),data_x.max(),npts)
    grid_y = np.linspace(data_y.min(),data_y.max(),npts)    
    vecx,vecy = np.meshgrid(grid_x, grid_y )
    vecz = gkde((vecx.ravel(),vecy.ravel())).reshape(vecx.shape)
    
    ax.contour ( vecx, vecy, vecz, **kwargs )
    return ax
    

def density_scatter ( x, y, cmap='Greys', ax=None, **kwargs ):
    '''
    Draw a scatterplot colored by density
    '''
    if ax is None:
        ax = plt.subplot(111)
    fmask = functions.finite_masker ( x, y )
    x = x[fmask]
    y = y[fmask]
    z = c_density(x,y)
    im = ax.scatter ( x, y, c=z, cmap=cmap, **kwargs )
    return ax, im

