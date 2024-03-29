import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib import patches
from matplotlib import patheffects
from scipy.integrate import quad
from scipy import ndimage
from astropy import units as u
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
    
def imshow ( im, ax=None, q=0.025, origin='lower', center=False, cval=0., **kwargs ):
    if ax is None:
        ax = plt.subplot(111)   
    if hasattr(im, 'unit'):
        im = im.value
                     
    vmin,vmax = np.nanquantile(im, [q,1.-q])
    if center:
        vextremum = np.max(np.abs([cval-vmin,vmax-cval]))
        vmin = cval - vextremum
        vmax = cval + vextremum

    imshow_out = ax.imshow ( im, vmin=vmin, vmax=vmax, origin=origin, **kwargs )
    
    
    return imshow_out, ax

def hist2d ( x, y, bins=None, alpha=0.01, ax=None, xscale='linear', yscale='linear', **kwargs ):
    if ax is None:
        ax = plt.subplot(111)
        
    if bins is None:
        nbins = 10
    elif isinstance(bins, int):
        nbins = bins
        

    if (bins is None) or isinstance(bins, int):   
        scale_fns = {'linear': lambda input: np.linspace ( *np.quantile(input[np.isfinite(input)], 
                                                                        [alpha, 1.-alpha]), nbins),
                    'log':lambda input: np.logspace ( *np.quantile(np.log10(input)[np.isfinite(np.log10(input))], 
                                                                   [alpha, 1.-alpha]), nbins)}
        bins = [scale_fns[xscale](x), scale_fns[yscale](y)]
    im = ax.hist2d ( x,y, bins=bins, **kwargs )
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    return im, ax

def contour ( im, ax=None, **kwargs ):
    if ax is None:
        ax = plt.subplot(111)
    X,Y = np.mgrid[:im.shape[0],:im.shape[1]]
    out = ax.contour ( Y, X, im, **kwargs)
    return out, ax

def imviz ( im, ax=None,  vtype='imshow', **kwargs):
    if ax is None:
        ax = plt.subplot(111)
        
    if vtype == 'imshow':
        out,ax = imshow ( im, ax, **kwargs)  
    elif vtype == 'contour':
        out,ax = contour (im, ax, **kwargs )
    return out,ax

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
        txt = ax.text ( rx, ry, text, ha=ha, va=va, **kwargs )
        
    if bordercolor is not None:
        txt.set_path_effects ( [patheffects.withStroke(linewidth=borderwidth, foreground=bordercolor)])
    return ax
    

def errorbar ( x, y, xlow=None, xhigh=None, ylow=None, yhigh=None, ax=None, c=None, 
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
        eim=ax.errorbar ( x,
                    y,
                    xerr = xerr,
                    yerr = yerr,                    
                    **kwargs,
                    )
        im = ax.scatter ( x, y, c=c,zorder=eim.lines[0].zorder+0.1, 
                         **scatter_kwargs )
        return ax, im
    return ax

def density_contour_scatter (data_x, data_y, ax=None, cmap='Greys', scatter_s=1, **kwargs):
    if ax is None:
        ax = plt.subplot(111)
    scatter_color = cmap(1.)
    s_im = ax.scatter ( data_x, data_y, zorder=0, color=scatter_color, s=scatter_s )  
    c_im, ax = density_contour ( data_x, data_y, ax, cmap=cmap, **kwargs )  
    return (s_im, c_im), ax

def density_contour (data_x,data_y, ax=None, npts=100, label=None, quantiles=None,filled=False, **kwargs):
    """
    Draw contour lines or filled contours representing the density of data points.

    Parameters:
    -----------
    data_x : array_like
        The x-coordinate values of the data points.
    data_y : array_like
        The y-coordinate values of the data points.
    ax : matplotlib.axes.Axes, optional
        The axis on which to draw the contour plot. If not provided, a new subplot will be created.
    npts : int, optional
        The number of points along each axis for creating the contour plot grid. Default is 100.
    label : str, optional
        A label for the contour plot. Default is None.
    quantiles : array_like, optional
        An array of quantiles used to compute contour levels. If provided, levels are derived from these quantiles.
        Default is None.
    filled : bool, optional
        If True, filled contours will be drawn; otherwise, only contour lines will be drawn. Default is False.
    **kwargs : dict, optional
        Additional keyword arguments passed to `ax.contour` or `ax.contourf`.

    Returns:
    --------
    matplotlib.axes.Axes
        The axis object on which the contour plot is drawn.
    """
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
    return im, ax
    

def density_scatter ( x, y, cmap='Greys', ax=None, rasterize=True, **kwargs ):
    """
    Draw a scatter plot colored by density.

    Parameters:
    -----------
    x : array_like
        The x-coordinate values of the data points.
    y : array_like
        The y-coordinate values of the data points.
    cmap : str or Colormap, optional
        The colormap used for coloring the points by density. Default is 'Greys'.
    ax : matplotlib.axes.Axes, optional
        The axis on which to draw the scatter plot. If not provided, a new subplot will be created.
    rasterize : bool, optional
        If True, rasterize the scatter plot for better performance. Default is True.
    **kwargs : dict, optional
        Additional keyword arguments passed to `ax.scatter`.

    Returns:
    --------
    matplotlib.axes.Axes
        The axis object on which the scatter plot is drawn.
    matplotlib.collections.PathCollection
        The PathCollection object representing the scatter plot.
    """
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
                       err_format='errorbar', 
                       std_alpha=0.15, 
                       err_alpha=0.15,
                       show_counts=False, 
                       ytext = 0.3,
                       text_kwargs = None,
                       show_std=True,
                       **kwargs ):
    '''
        Compute running quantiles for a set of data points within specified bins.

        This function calculates running quantiles of a dataset `y` as a function of a variable `x`, using user-defined bins. 
        The computed quantiles are specified by the `alpha` parameter. The results can be plotted on an existing matplotlib 
        `ax` or a new one is created if `ax` is not provided.

        Parameters:
        x (array-like): The independent variable.
        y (array-like): The dependent variable.
        bins (array-like): Binning specification for `x`.
        alpha (float, optional): The quantile value to compute (default is 0.16, corresponding to the 16th percentile).
        ax (matplotlib.axes.Axes, optional): The Axes object to plot the results (default is None, which creates a new subplot).
        erronqt (bool, optional): If True, error bars on quantiles are plotted (default is False).
        label (str, optional): Label for the data series (default is None).
        yerr (array-like, optional): Error values for `y` data (default is None).
        std_alpha (float, optional): Alpha value for the error shading (default is 0.15).
        show_counts (bool, optional): If True, display counts within each bin (default is False).
        ytext (float, optional): Vertical position for count labels (default is 0.3).
        **kwargs: Additional keyword arguments to customize the plot.

        Returns:
        xmid (array): The midpoints of the bins.
        ystat (array): Computed quantiles or quantiles with error bars, depending on `erronqt`.

        Example:
        >>> import matplotlib.pyplot as plt
        >>> x = [1, 2, 3, 4, 5]
        >>> y = [5, 6, 4, 8, 3]
        >>> bins = [1, 2, 3, 4, 5]
        >>> running_quantile(x, y, bins, alpha=0.2, ax=plt.gca(), erronqt=True, label='Data')
        >>> plt.show()
        '''    
    if ax is None:
        ax = plt.subplot(111) 
    if text_kwargs is None:
        text_kwargs = {'fontsize':plt.rcParams['font.size']*.5}
           
    qt = [alpha, 0.5, 1.-alpha]
    out = sampling.binned_quantile ( x, y, bins=bins, qt=qt, erronqt=erronqt, yerr=yerr, return_counts=show_counts)
    if show_counts:
        xmid, ystat, counts = out
    else:
        xmid, ystat = out
    
    if erronqt:
        if err_format == 'errorbar':
            errorbar ( xmid, ystat[:,1,2],
                    xlow = bins[:-1],
                    xhigh = bins[1:], 
                    ylow=ystat[:,1,1],
                    yhigh=ystat[:,1,3],
                    ax=ax,
                    label=label,
                    **kwargs
                    ) 
            
       
        elif err_format == 'fill_between':            
            ax.plot (
                xmid,
                ystat[:,1,2],   
                label=label,
                **kwargs             
            )
            ax.fill_between (
                xmid,
                ystat[:,1,1],
                ystat[:,1,3],
                alpha=err_alpha,
                **kwargs
            )
        elif err_format is None:
            ax.plot (
                xmid,
                ystat[:,1,2],   
                label=label,
                **kwargs             
            )            
        else:
            raise ValueError (f"err_format:{err_format} not recognized!")
        if show_std:
            ypad = np.zeros( [1+ystat.shape[0],2] )
            ypad[1:,0] = ystat[:,0,2]
            ypad[1:,1] = ystat[:,2,2]
            ypad[0,0] = ypad[1,0]
            ypad[0,1] = ypad[1,1] 
                        
            if err_format == 'fill_between':
                for eidx in range(2):
                    ax.plot ( bins, ypad[:,eidx], **kwargs)
            else:
                ax.fill_between ( bins, ypad[:,0], ypad[:,1], alpha=std_alpha,**kwargs )           
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
        
        ymin = ax.get_ylim()[0]
        for idx,xm in enumerate(xmid):
            
            if len(ystat.shape) == 2:
                ym = ystat[idx,1]
            else:
                ym = ystat[idx,1,2]
            if ax.get_yscale() == 'log':
                # \\ log10(yn) = log10(ym) + log10(dm)
                # \\ log10(yn) = log10(ym * dm)
                # \\ yn = ym * dm
                # \\ yn = ym * ymax/ymin * ytext
                # \\ yn = ym * (ymax/ymin) * 10**ytext
                yspan = np.divide(*ax.get_ylim()[::-1])             
                yplot = ym * yspan ** ytext                     
            else:     
                yspan = np.subtract(*ax.get_ylim()[::-1])           
                yplot = ym + yspan * ytext
                
            ymin,ymax = ax.get_ylim()
            if yplot > ymax:
                yplot = ymax      
            if yplot < ymin:
                yplot = ymin
            text (  xm, 
                    yplot,
                    counts[idx], 
                    ha='center',                     
                    coord_type='absolute',
                    ax=ax,
                    **text_kwargs,                
                    **kwargs )
    return xmid, ystat

def get_subplot_aspectratio ( ax ):
    """
    Calculates the aspect ratio of a subplot.

    Parameters:
    - ax (matplotlib.axes.Axes): The subplot to calculate the aspect ratio for.

    Returns:
    - float: The aspect ratio of the subplot.

    Note:
    This function computes the aspect ratio of a subplot by considering the size of the subplot within the figure
    and the aspect ratio of the data contained within it.
    """    
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
 
def set_aspect_ratio ( ax, desired_aspect, learning_rate=0.5, tolerance=0.01 ):
    """
    Adjusts the aspect ratio of a matplotlib subplot to match the desired aspect ratio.

    Parameters:
    - ax (matplotlib.axes.Axes): The subplot to adjust.
    - desired_aspect (float): The desired aspect ratio (width / height).
    - learning_rate (float, optional): The rate at which the aspect ratio adjusts towards the desired value. Defaults to 0.5.
    - tolerance (float, optional): The acceptable difference between the realized aspect ratio and the desired aspect ratio. Defaults to 0.01.

    Returns:
    - matplotlib.axes.Axes: The adjusted subplot.

    Raises:
    - ValueError: If the aspect ratio adjustment enters a runaway state (aspect_ratio < 0.05 or aspect_ratio > 10).

    Note:
    This function iteratively adjusts the aspect ratio of a subplot until it matches the desired aspect ratio within the specified tolerance.
    Look. I don't really get how matplotlib's subplot aspect ratios work. It is
    arcane to me. Hence, this monstrosity:
    """
    aspect_ratio=1.
    while True:
        ax.set_aspect ( aspect_ratio, adjustable='box' )
        realized_ratio = get_subplot_aspectratio(ax)
        if abs(realized_ratio - desired_aspect) < tolerance:
            break
        elif (aspect_ratio < 0.05) or (aspect_ratio>10.):
            raise ValueError ("Runaway aspect ratio!")
         
        aspect_ratio += learning_rate*(desired_aspect - realized_ratio)

    return ax 

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

def imshow_astro ( img, wcs, ax=None, q=0.025, vtype='imshow', use_projection=False, pixcache=None, **kwargs ):
    if ax is None:
        ax = plt.subplot(111)
    
    if pixcache is None:    
        x = np.arange(img.shape[1])
        y = np.arange(img.shape[0])
        Y,X = np.mgrid[:img.shape[0],:img.shape[1]]   
        
        ra, dec = wcs.wcs_pix2world ( X, Y, 0 )
        pixcache = (ra,dec)
    else:
        print('[imshow_astro: Warning] Using cached RA, DEC!')
        ra, dec = pixcache
        
    
    vmin,vmax = np.nanquantile(img, [q,1.-q])
    if 'vmin' not in kwargs.keys():
        kwargs['vmin'] = vmin
    if 'vmax' not in kwargs.keys():
        kwargs['vmax'] = vmax
        
    if vtype == 'imshow':
        imout = ax.pcolormesh ( ra, dec, img, **kwargs )
    elif vtype == 'contour':
        imout = ax.contour ( ra, dec, img, **kwargs )
    
    if np.subtract(*ax.get_xlim()) < 0:
        ax.set_xlim ( ax.get_xlim ()[::-1] )
        
    return imout, ax, pixcache

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

def add_physbar ( xcenter, y, pixscale, distance, ax=None, bar_physical_length = 10. * u.kpc, **kwargs ):
    if ax is None:
        ax = plt.subplot(111)
    bar_lengthsky = ((bar_physical_length/distance).decompose()*u.rad).to(u.arcsec)
    bar_lengthpix = bar_lengthsky / pixscale # arcsec / pix
    
    xbegin = xcenter - bar_lengthpix/2.
    xend = xcenter + bar_lengthpix/2.
    ypad = abs(np.subtract(*ax.get_ylim())) * 0.05
    print(ypad)
    ax.hlines ( y, xbegin, xend, lw=3, color='k' )
    ax.text ( 
        xcenter, 
        y + ypad, 
        f'{bar_physical_length.value:i} {bar_physical_length.unit}', 
        ha='center', 
        va='bottom', 
        **kwargs 
    )    
    
def histstack ( x, y, ax=None, xbins=10, ybins=10, show_quantile=True, quantile=0.5, quantile_kwargs={}, edgecolor='k', facecolor='lightgrey', linewidth=1 ):
    if ax is None:
        ax = plt.subplot(111)
    if show_quantile:
        if 'color' not in quantile_kwargs.keys():
            quantile_kwargs['color'] = 'tab:red'
        if 's' not in quantile_kwargs.keys():
            quantile_kwargs['s'] = 8.
    
        
    if isinstance(xbins, int):
        xbins = np.linspace(*np.quantile(x, [0.05,.95]), num=xbins)
    if isinstance(ybins, int):
        ybins = np.linspace(*np.quantile(y, [0.05,.95]), num=ybins)        
    
    assns = np.digitize ( x, xbins )
    zidx=0
    for bin_index in np.arange(1, xbins.size)[::-1]:
        histout = np.histogram(y[assns==bin_index], bins=ybins)    
        base = np.median(x[assns==bin_index])
        ys = .75*histout[0]/histout[0].max() + base        
        shift = (ys.max()-ys.min())*.4
        
        out = ax.step( ys - shift, ybins[1:], color=edgecolor, alpha=0.7, where='pre', zorder=zidx, linewidth=linewidth)
        ax.fill_betweenx( 
            ybins[1:], 
            base - shift,
            ys - shift, 
            zorder = zidx -0.1,            
            step='post',
            color = facecolor,
        )
        if show_quantile:
            ax.scatter ( 
                base, 
                np.median(y[assns==bin_index]), 
                zorder=len(xbins) + 30, 
                **quantile_kwargs
            )
        zidx += 1

def violinplot ( distributions, ax=None, clist=None, labels=None, **kwargs ):
    if ax is None:
        ax = plt.subplot(111)
    
    if clist is None:
        clist = [ ec.ColorBase('C0') for _ in range(len(distributions)) ]
    elif isinstance ( clist, str ):
        basecolor = clist
        clist = [ ec.ColorBase(basecolor) for _ in range(len(distributions)) ]
    elif not isinstance(clist[0], ec.ColorBase):
        clist = [ ec.ColorBase(x) for x in clist ]
        
    parts = ax.violinplot([x for x in distributions], bw_method=0.35, **kwargs)

        
    # \\ update violinplot shaded region colors (alpha < 1 by default)
    for pidx,body in enumerate(parts['bodies']):
        body.set_facecolor ( clist[pidx].base)
        #parts['cbars'].set
        body.set_edgecolor ( clist[pidx].base )

    # \\ update violinplot line colors
    #parts['cbars'].set_ec ( np.array([cc.base for cc in clist]) )
    #parts['cmins'].set_ec ( np.array([cc.base for cc in clist]) )
    #parts['cmaxes'].set_ec ( np.array([cc.base for cc in clist]) )
    
    # \\ add 16th->84th quantile 
    quantiles = np.array([np.quantile(dist,[.16,.84]) for dist in distributions])        
    ax.vlines ( np.arange(1,len(distributions)+1), quantiles[:,0], quantiles[:,1], colors=[x.modulate(-0.1).base for x in clist], lw=4)

    # \\ set qualitative labels
    ax.set_xticks (range(1,len(distributions)+1))
    if labels is not None:
        ax.set_xticklabels(labels)
    
    return parts, ax
