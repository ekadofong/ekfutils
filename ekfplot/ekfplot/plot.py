import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.stats import gaussian_kde
from ekfstats import functions, sampling

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
    
def imshow ( im, ax=None, q=0.025, **kwargs ):
    if ax is None:
        ax = plt.subplot(111)
    vmin,vmax = np.nanquantile(im, [q,1.-q])
    ax.imshow ( im, vmin=vmin, vmax=vmax, **kwargs )
    return ax

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
    if x.size < 30:        
        return np.ones_like(x)
    xy = np.vstack([x,y])
    fn = gaussian_kde(xy, **kwargs)

    if return_fn:
        return fn
    else:
        z = fn(xy)
        return z
    
def density_contour (data_x,data_y, ax=None, npts=100, label=None, **kwargs):
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
    
    im = ax.contour ( vecx, vecy, vecz, **kwargs )
    if label is not None:
        if 'cmap' not in kwargs.keys():
            kwargs['color'] = plt.cm.viridis(0.5)
        else:
            if isinstance(kwargs['cmap'], str):
                kwargs['color'] = getattr(plt.cm,kwargs['cmap'])(0.5)
            else:
                kwargs['color'] = kwargs['cmap'](0.5)
            del kwargs['cmap']
        ax.plot ( 0, 0, label=label, **kwargs)
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
    im = ax.scatter ( x, y, c=z, cmap=cmap, vmin=0., vmax=z.max(), **kwargs )
    return ax, im

def running_quantile ( x, y, bins, alpha=0.16, ax=None, erronqt=False, label=None, **kwargs ):
    if ax is None:
        ax = plt.subplot(111)    
    qt = [alpha, 0.5, 1.-alpha]
    xmid, ystat = sampling.binned_quantile ( x, y, bins=bins, qt=qt, erronqt=erronqt)
    
    if erronqt:
        errorbar ( xmid, ystat[:,1,2],
                xlow = bins[:-1],
                xhigh = bins[1:], 
                ylow=ystat[:,1,1],
                yhigh=ystat[:,1,3],
                ax=ax,
                label=label,
                **kwargs
                ) 
        ax.fill_between ( xmid, ystat[:,0,2], ystat[:,2,2], alpha=0.15,**kwargs )       
    else:        
        errorbar ( xmid, ystat[:,1],
                xlow = bins[:-1],
                xhigh = bins[1:], 
                ylow=ystat[:,0],
                yhigh=ystat[:,2],
                ax=ax,
                label=label,
                **kwargs
                )