import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib import patches
from matplotlib import patheffects
from scipy.integrate import quad
from scipy import ndimage
from ekfstats import functions, sampling
from . import colors as ec


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
    
def imshow ( im, ax=None, q=0.025, origin='lower', **kwargs ):
    if ax is None:
        ax = plt.subplot(111)
    vmin,vmax = np.nanquantile(im, [q,1.-q])
    imshow_out = ax.imshow ( im, vmin=vmin, vmax=vmax, origin=origin, **kwargs )
    return imshow_out, ax

def imshow_segmentationmap ( segmap, ax=None, color='w', origin='lower', lw=1, **kwargs ):
    if ax is None:
        ax = plt.subplot(111)    
    segmap_inflated = ndimage.binary_dilation( segmap, iterations=(lw+1)//2).astype(int)
    segmap_shrunk = ndimage.binary_erosion(segmap, iterations=(lw+1)//2 ).astype(int)
    outline = segmap_inflated - segmap_shrunk
    
    colorbase = ec.ColorBase ( color )
    cmap = colorbase.sequential_cmap ( fade=0. )
    im = ax.imshow ( outline>0, cmap=cmap, origin=origin, **kwargs )
    return im, ax

def outlined_plot ( x, y,  *args, color='k', lw=4, ax=None, bkgcolor='w', label=None, **kwargs ):
    if ax is None:
        ax = plt.subplot(111)
      
    ax.plot ( x,y, *args, lw=lw, color=bkgcolor, **kwargs )  
    ax.plot ( x,y, *args, lw=lw*0.5, color=color, label=label, **kwargs )
    return ax

def text ( rx, ry, text, ax=None, ha=None, va=None, bordercolor=None, borderwidth=1., coord_type='relative', **kwargs ):
    if ax is None:
        ax = plt.subplot(111)
    
    if ha is None:
        ha = 'right' if rx > 0.5 else 'left'
    if va is None:
        va = 'top' if ry > .5 else 'bottom'
    
    if coord_type == 'relative':
        txt = ax.text ( rx, ry, text, transform=ax.transAxes, ha=ha, va=va, **kwargs )
    elif coord_type == 'absolute':
        txt = ax.text ( rx, ry, text, **kwargs )
        
    if bordercolor is not None:
        txt.set_path_effects ( [patheffects.withStroke(linewidth=borderwidth, foreground=bordercolor)])
    return ax
    

def errorbar ( x, y, xlow=None, xhigh=None, ylow=None, yhigh=None, ax=None, c=None, zorder=9,
               xerr=None, yerr=None,
               scatter_kwargs={}, xsigma=1., ysigma=1., **kwargs ):
    '''
    Draw errorbars where errors are given by distribution quantiles
    '''
    if ax is None:
        ax = plt.subplot(111)
        
    if xerr is not None:
        if (xlow is not None) or (xhigh is not None):
            raise ValueError ("Both xerr and xlow/xhigh are defined!")
    if yerr is not None:
        if (ylow is not None) or (yhigh is not None):
            raise ValueError ("Both yerr and ylow/yhigh are defined!")    
    
    if xerr is None:
        if xlow is None:
            xlow = np.NaN
        if xhigh is None:
            xhigh = np.NaN
        xerr =  np.array([[x-xlow],[xhigh - x]]).reshape(2,-1) * xsigma
    if yerr is None:
        if ylow is None:
            ylow = np.NaN
        if yhigh is None:
            yhigh = np.NaN            
        yerr =  np.array([[y-ylow],[yhigh - y]]).reshape(2,-1) * ysigma
    
    if np.isnan(xerr).all():
        xerr = None
    if np.isnan(yerr).all():
        yerr = None

    if 'marker' in kwargs.keys():
        kwargs['fmt'] = kwargs['marker']
        del kwargs['marker']
    elif 'fmt' not in kwargs.keys():
        kwargs['fmt'] = 'o'
        
    if 'cmap' in kwargs.keys():
        scatter_kwargs['cmap'] = kwargs['cmap']
        del kwargs['cmap']
        
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

def density_contour (data_x,data_y, ax=None, npts=100, label=None, quantiles=None,filled=False, **kwargs):
    '''
    Draw a contour based on density
    '''
    if ax is None:        
        ax = plt.subplot(111)
    fmask = functions.finite_masker ( [data_x, data_y] )
    data_x = data_x[fmask]
    data_y = data_y[fmask]

    gkde = sampling.c_density ( data_x,data_y, return_fn=True, nmin=0 )    
    grid_x = np.linspace(data_x.min(),data_x.max(),npts)
    grid_y = np.linspace(data_y.min(),data_y.max(),npts)    
    vecx,vecy = np.meshgrid(grid_x, grid_y )
    vecz = gkde((vecx.ravel(),vecy.ravel())).reshape(vecx.shape)
    # \\ np.trapz(np.trapz(vecz, vecx), grid_y) == 1 for a normalized GKDE
    # \\ dx = np.mean(np.diff(grid_x))
    # \\ dy = np.mean(np.diff(grid_y))
    # \\ np.sum(vecz*dx*dy) == 1 for a normalized gKDE
    if ('levels' in kwargs.keys()) and (quantiles is not None):
        raise ValueError ("Cannot define both levels and quantiles")
    elif quantiles is not None:
        dx = np.mean(np.diff(grid_x))
        dy = np.mean(np.diff(grid_y))
        vy = np.sort(vecz.flatten())[::-1]
        cumulative = np.cumsum(vy*dx*dy)
        #quantiles = np.array([ quad(lambda x: functions.gaussian(x, 'normalize',0.,1.), -sigma,sigma)[0] for sigma in quantiles])**2
        quantiles = np.power(quantiles, 2) # \\ need to take the square to line up with 1D 
        levels = np.sort([ vy[np.argmin(abs(cumulative-qx))] for qx in quantiles ])
        arr = np.diff(levels) > np.finfo(float).resolution
        skip = np.sum(~arr)
        levels = levels[skip:]
        kwargs['levels'] = levels
    
    if filled:
        fn = ax.contourf
    else:
        fn = ax.contour
    im = fn ( vecx, vecy, vecz, **kwargs )    
    if label is not None:
        if 'cmap' not in kwargs.keys():
            kwargs['color'] = plt.cm.viridis(0.5)
        else:
            if isinstance(kwargs['cmap'], str):
                kwargs['color'] = getattr(plt.cm,kwargs['cmap'])(0.5)
            else:
                kwargs['color'] = kwargs['cmap'](0.5)
            del kwargs['cmap']
        ax.plot ( 0, 0, label=label, lw=2, **kwargs)
    return ax
    

def density_scatter ( x, y, cmap='Greys', ax=None, rasterize=True, **kwargs ):
    '''
    Draw a scatterplot colored by density
    '''
    if ax is None:
        ax = plt.subplot(111)
    fmask = functions.finite_masker ( [x, y] )
    x = x[fmask]
    y = y[fmask]
    z = sampling.c_density(x,y)
    im = ax.scatter ( x, y, c=z, cmap=cmap, vmin=0., vmax=z.max(), **kwargs )
    ax.set_rasterization_zorder ( 10 )
    return ax, im

def running_quantile ( x, 
                       y, 
                       bins, 
                       alpha=0.16, 
                       ax=None, 
                       erronqt=False, 
                       label=None, 
                       yerr=None, 
                       erralpha=0.15, 
                       show_counts=False, 
                       ytext = 0.3,
                       **kwargs ):
    if ax is None:
        ax = plt.subplot(111)    
    qt = [alpha, 0.5, 1.-alpha]
    out = sampling.binned_quantile ( x, y, bins=bins, qt=qt, erronqt=erronqt, yerr=yerr, return_counts=show_counts)
    if show_counts:
        xmid, ystat, counts = out
    else:
        xmid, ystat = out
    
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
        
        ypad = np.zeros( [1+ystat.shape[0],2] )
        ypad[1:,0] = ystat[:,0,2]
        ypad[1:,1] = ystat[:,2,2]
        ypad[0,0] = ypad[1,0]
        ypad[0,1] = ypad[1,1]
                        
        ax.fill_between ( bins, ypad[:,0], ypad[:,1], alpha=erralpha,**kwargs )       
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
    
    if show_counts:
        yspan = np.subtract(*ax.get_ylim()[::-1])
        ymin = ax.get_ylim()[0]
        for idx,xm in enumerate(xmid):
            ax.text ( xm, ymin + ytext * yspan, counts[idx], ha='center', fontsize=plt.rcParams['font.size']*.5, **kwargs )
    return xmid, ystat

def get_subplot_aspectratio ( ax ):
    figwidth, figheight = ax.get_figure().get_size_inches ()
    ax_bbox = ax.get_position ()
    sp_fracw = ax_bbox.width
    sp_frach = ax_bbox.height
    sp_width = figwidth * sp_fracw
    sp_height = figheight * sp_frach
    display_aspect = sp_height/sp_width
    
    if ax.get_xscale() == 'linear':
        data_width = np.subtract(*ax.get_xlim())
    elif ax.get_xscale() == 'log':
        data_width =  np.subtract(*np.log10(ax.get_xlim()))
        
    if ax.get_yscale() == 'linear':
        data_height = np.subtract(*ax.get_ylim())
    elif ax.get_yscale() == 'log':
        data_height = np.subtract(*np.log10(ax.get_ylim()))
                
    data_aspect = data_height/data_width

    subplot_aspect = display_aspect/data_aspect
    return subplot_aspect
    
def alpha_pcolor ( x_edges, y_edges, Z, alpha=1., ax=None, cmap='Greys', **kwargs ):
    '''
    Create a pseudocolor plot with varying alpha transparency based on provided data.

    Parameters:
        x_edges (array-like): The bin edges for the x-axis.
        y_edges (array-like): The bin edges for the y-axis.
        Z (array-like): The data values for each (x, y) bin.
        alpha (float or array-like, optional): Transparency value or an array of alpha values for each (x, y) bin. Default is 1.0.
        ax (AxesSubplot, optional): The Axes on which to create the plot. If not provided, a new subplot will be created.
        cmap (str or colormap, optional): Colormap for coloring the bins. Default is 'Greys'.
        **kwargs: Additional keyword arguments to pass to the ax.pcolor function.

    Returns:
        AxesSubplot: The subplot containing the pseudocolor plot.

    This function creates a pseudocolor plot with varying alpha transparency based on the provided data. It can be used to
    visualize data with two-dimensional binning. The 'x_edges' and 'y_edges' are arrays that define the bin edges along the
    x and y axes respectively. The 'Z' array contains the data values associated with each (x, y) bin. The 'alpha' parameter
    specifies the transparency level for the plot. It can be a single value applied to the entire plot or an array-like
    specifying individual alpha values for each (x, y) bin. The 'ax' parameter allows plotting on a specific subplot. If
    'ax' is not provided, a new subplot will be created. The 'cmap' parameter sets the colormap for coloring the bins.

    Note: This function utilizes the Matplotlib library for plotting and modifies the provided 'ax' subplot if provided.

    '''    
    if ax is None:
        ax = plt.subplot(111)
    if np.isscalar(alpha):
        ax.pcolor ( 
            x_edges,
            y_edges,
            Z,
            alpha=alpha,
            **kwargs
        )
    else:
        if isinstance(cmap, str):
            cmap = getattr(plt.cm, cmap)
        norm = plt.Normalize ( np.nanmin(Z), np.nanmax(Z) )
        
        for i in range(len(x_edges) - 1):
            for j in range(len(y_edges) - 1):
                rect = patches.Rectangle((x_edges[i], y_edges[j]), x_edges[i + 1] - x_edges[i], y_edges[j + 1] - y_edges[j],
                                facecolor=cmap(norm(Z[j, i])), alpha=alpha[j, i], edgecolor=cmap(norm(Z[j,i])))
                ax.add_patch(rect)
        ax.set_xlim ( x_edges[0], x_edges[-1] )
        ax.set_ylim ( y_edges[0], y_edges[-1] )
    return ax

def imshow_astro ( img, wcs, ax=None, q=0.025, plot_type='imshow', **kwargs ):
    if ax is None:
        ax = plt.subplot(111)
        
    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    Y,X = np.mgrid[:img.shape[0],:img.shape[1]]   
    
    ra, dec = wcs.wcs_pix2world ( X, Y, 0 )
    
    vmin,vmax = np.nanquantile(img, [q,1.-q])
    if plot_type == 'imshow':
        imout = ax.pcolormesh ( ra, dec, img, vmin=vmin, vmax=vmax, **kwargs )
    elif plot_type == 'contour':
        imout = ax.contour ( ra, dec, img, **kwargs )
    
    if np.subtract(*ax.get_xlim()) < 0:
        ax.set_xlim ( ax.get_xlim ()[::-1] )
        
    return imout, ax

def rectangle ( xy, width, height, *args, rotation=0, ax=None, **kwargs ):
    if ax is None:
        ax = plt.subplot(111)
    
    center = (xy[0] + width/2, xy[1]+height/2)
    
    #Rotate rectangle patch object
    ts = ax.transData    
    tr = transforms.Affine2D().rotate_deg_around(center[0],center[1], rotation)
    # t = ts + tr  < XX the order matters a lot here, it's transform to display then rotate or vice versa
    t = tr + ts
         
      
    rect1 = patches.Rectangle(xy,width,height,*args,**kwargs,transform=t)
    ax.add_patch(rect1)
    return ax