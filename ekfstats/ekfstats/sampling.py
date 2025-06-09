import numpy as np
from scipy import optimize, stats
from scipy import interpolate,integrate
from scipy.stats import gaussian_kde, sigmaclip
from statsmodels.stats.proportion import proportion_confint
from . import functions

def finite_masker ( arr_l, inplace=False, ul=np.inf, ll=-np.inf ):
    '''
    Returns a mask that is True where both input arrays 
    are finite-valued
    '''
    if not isinstance(arr_l, list):
        arr_l = list(arr_l)
    mask = np.isfinite(arr_l[0])&(arr_l[0]<ul)&(arr_l[0]>ll)
    for arr in arr_l[1:]:
        mask &= np.isfinite(arr)&(arr<ul)&(arr>ll)
    if inplace:
        arr_out = []
        for arr in arr_l:
            arr_out.append(arr[mask])
        if len(arr_out) == 1:
            return arr_out[0]        
        return arr_out
    else:
        return mask
    
def fmasker ( *args ):
    '''
    Convenience wrapper for finite_masker that takes
    arguments and returns in-place.
    '''
    return finite_masker(args, inplace=True)

def resample ( x_l, npull=500 ):
    if not isinstance(x_l, list):
        x_l = [x_l]
        
    arr = np.zeros([npull, len(x_l), len(x_l[0])])
    for idx in range(npull):
        indices = np.random.randint(0, len(x_l[0]), size=len(x_l[0]) )
        for jdx in range(len(x_l)):
            arr[idx, jdx] = x_l[jdx][indices]
    return arr

def estimate_y ( xs,fn, args, u_args=None, npull=1000):
    ys = np.zeros([npull, xs.size])
    if u_args is None:
        assert len(args.shape) == 2
        
    for _ in range(npull):
        if u_args is not None:
            apull = np.random.normal(args, u_args)
        else:
            apull = args[np.random.randint(0, args.shape[0])]
        ys[_] = fn(xs, *apull)
    return ys

def c_density ( x, y, return_fn=False, clean=True, nmax=None, nmin=30, **kwargs ):
    '''
    Compute gKDE density based on sample
    '''
    # Calculate the point density
    if x.size < nmin:        
        return np.ones_like(x)
    if clean:
        fmask = finite_masker ( [x, y] )
        if fmask.sum() == 0:
            raise ValueError ("All inf/nan array encountered")
        x = x[fmask]
        y = y[fmask]
        
    if nmax is not None:
        if x.size > nmax:
            indices = np.random.randint(0, x.size, nmax)
            x = x[indices]
            y = y[indices]
    xy = np.vstack([x,y])
    fn = gaussian_kde(xy, **kwargs)

    if return_fn:
        return fn
    else:
        z = fn(xy)
        return z
    
def rejection_sample_fromarray ( x, y, nsamp=10000 ):
    pdf_x = interpolate.interp1d(x,y,bounds_error=False, fill_value=0.)
    sample = rejection_sample ( x, pdf_x, nsamp )
    return sample

def rejection_sample ( x, pdf_x, nsamp=10000, maxiter=100, oversample=5):    
    '''
    Do rejection sampling for a probability density function
    that is known over a set of points x
    '''
    neg_fn = lambda x: -pdf_x(x)
    pout = optimize.minimize(neg_fn, x.mean(), method='nelder-mead', bounds=[(x.min(),x.max())] )
    if pout.nit == 0:
        raise ValueError ("Minimization did not complete correctly.")
    max_pdf = pdf_x(pout.x)
    
    sample = np.zeros(nsamp)
    nadded = 0   
    niter = 0 
    while nadded < nsamp:
        #idx_draw = np.random.randint(0, x.size, size=5*nsamp)
        x_draw = np.random.uniform(x.min(),x.max(),oversample*nsamp)
        pdf_at_draw = pdf_x(x_draw)
        uni_at_draw = np.random.uniform(0.,max_pdf,x_draw.size)
        keep = pdf_at_draw >= uni_at_draw        
        to_add = x_draw[keep][:(nsamp-nadded)]
        sample[nadded:(nadded+to_add.size)] = to_add
        nadded += keep.sum()  
        niter += 1
        if niter > maxiter:
            print('Warning! Max iterations reached')
            sample = sample[:nadded+to_add.size]
            break   
    return sample  

def sample_from_pdf ( var, prob, nsamp=100, is_bounds=False, spacing='linear', ngrid=10000, return_indices=False, verify=True):    
    """
    Process the input variable and generate samples based on probability distribution.

    Args:
        var : Union[list, np.ndarray] - Input variable which can be a list or numpy array.
        is_bounds : bool - Flag indicating whether bounds are provided.
        spacing : str - Type of spacing ('linear' or 'log') for generating variables if bounds are provided.
        ngrid : int - Number of grid points for spacing.
        prob : Callable - Probability distribution function.
        nsamp : int - Number of samples to generate.
        return_indices : bool - Flag indicating whether to return the indices along with the coordinates.

    Return:
        samples : Union[np.ndarray, Tuple[np.ndarray, np.ndarray]] - Generated samples. If return_indices is True, returns a tuple of samples and indices.
    
    Raise:
        AssertionError: If the standard deviation of the differences divided by their mean is not less than 1e-5 when bounds are not provided.
    """       
    if not isinstance(var, list):        
        if not is_bounds:
            if verify:
                assert np.std(np.diff(var))/np.mean(np.diff(var)) <  1e-5
        else:            
            if spacing == 'linear':                          
                var = np.linspace(*var,ngrid)                
                prob = prob(var)                 
            elif spacing == 'log':
                var = np.logspace(*var,ngrid)
                prob = prob(var) * var * np.log(10.)                       
        if return_indices:
            indices = np.random.choice( np.arange(len(var)), p=prob/prob.sum(), size=nsamp)
            return var[indices], indices
        else:
            return np.random.choice( var, p=prob/prob.sum(), size=nsamp)
    else:
        indices = np.arange(var[0].size)#.reshape(var[0].shape)
        flattened_prob  = prob.flatten()
        #flattened_prob /= flattened_prob.sum()
        pulled_indices = np.random.choice ( indices, p=flattened_prob, size=nsamp )
        coords = np.zeros ( [len(var), nsamp] )
        for idx in range(len(var)):
            coords[idx] = var[idx].flatten()[pulled_indices]
        if return_indices:
            return coords, pulled_indices    
        else:
            return coords

def iqr ( ys, alpha=0.16 ):
    return np.subtract(*np.nanquantile(ys, [1.-alpha, alpha]))

def get_quantile ( xs, ys, alpha ):
    midpts = 0.5*(xs[1:]+xs[:-1])
    ctrapz = integrate.cumulative_trapezoid(ys,xs)
    return np.interp( alpha, ctrapz, midpts )

def weighted_quantile ( x, w, qts ):
    psort = np.argsort(x)
    cog = np.cumsum(w[psort]) / w.sum()
    output = np.interp ( qts, cog, x[psort] )
    return output

def get_quantile_of_value ( x, val ):
    return np.interp(val, np.sort(x), np.linspace(0.,1.,x.size))

def pdf_product ( xA, pdfA, xB, pdfB, npts=100, normalize=True, return_midpts=False, alpha=1e-3 ):
    '''
    return an approximation of the probability density function that describes
    the product of two PDFs A & B
    '''    
    product = cross_then_flat(xA,xB)
    probdensity    = cross_then_flat(pdfA,pdfB)
    xmin,xmax = weighted_quantile ( product, probdensity, [alpha, 1.-alpha] ) 
    
    domain = np.linspace(xmin,xmax, npts)
    midpts = 0.5*(domain[:-1]+domain[1:])
    assns       = np.digitize ( product, domain )            
    pdf         = np.array([np.sum(probdensity[assns==x]) for x in np.arange(1, domain.shape[0])])
    if normalize:
        nrml        = np.trapz(pdf, midpts)
        pdf        /= nrml
    interpfn = build_interpfn( midpts, pdf )
    if return_midpts:
        return midpts, interpfn
    else:
        return interpfn

def cross_then_flat ( a, b):
    '''
    [ b0 ] x [ a0 a1 a2 a3 ] 
    | b1 |
    | b2 |
    [ b3 ]                
    '''
    return np.asarray(np.matrix(a).T*np.matrix(b)).flatten()
    

def dynamic_upsample ( x, p, N=100 ):
    # \\ sample wherein high log10(p) points are 
    # \\ upsampled more
    def dp_pred ( dx, x ):
        dp = p(x+dx) - p(x)
        return dp

    def dp_find ( dx,x ):
        dp = dp_pred(dx, x)
        ds_sq = dp**2 + dx**2
        resid = (ds_opt**2 - ds_sq)**2
        return resid                
    # \\ establish uniform spacing along the
    # \\ line integral;
    # \\ ds^2 = dx^2 + dp^2
    # \\ ds^2 = dx^2(1 + (df/dx @ x)^2)
    dX = x.max() - x.min()
    dp_vec = np.diff(p(x)) 
    dx_vec = np.diff(x)
    ds_vec = np.sqrt(dp_vec**2 + dx_vec**2)
    dS = np.sum(ds_vec)    
    ds_opt = dS / N # \\ steps to get along 

    cx = np.zeros(N)
    cx[0] = x.min()
    dx_init = dX/N
    dx_min = dx_init * 0.01
    for idx in range(1,N):
        x_i = cx[idx-1]    
        dpdx_i = dp_pred(dx_init, x_i)/dx_init
        start = np.sqrt(ds_opt**2 / (1. + dpdx_i**2)) # starting guess fo dx
        pout = optimize.minimize(dp_find, start, bounds=((dx_min,ds_opt),), args=(x_i,))
        assert pout.status == 0
        dx_opt = float(pout.x)
        cx[idx] = x_i + dx_opt
    return cx, p(cx)

def bootstrap_metric ( x, metric_fn, u_x=None, npull=1000, err_type='1684', quantiles=None, vartype='linear', verbose=True):
    if not isinstance(err_type, str):
        efunc = err_type
    elif err_type == '1684_combined':
        efunc = lambda foo: np.subtract(*np.quantile(foo, [0.84, 0.16]))
    elif err_type == '1684':
        efunc = lambda foo: np.quantile(foo, [0.16, 0.84])
    elif err_type == '0595':
        efunc = lambda foo: np.quantile(foo, [0.05,0.95])        
    elif err_type == 'std':
        efunc = np.std
    elif err_type == 'quantiles':
        efunc = lambda foo: np.quantile(foo, quantiles)
        
    if u_x is not None:
        _resamp = np.random.normal ( x, u_x )  
        if vartype == 'log10':
            _resamp = np.log10(_resamp)      
        resamp = np.random.choice( _resamp, size=[npull,x.size] )        
    else:
        resamp = np.random.choice ( x, size=[npull, x.size] )
        if vartype == 'log10':
            resamp = np.log10(resamp)
    
    try:
        resampled_metric = metric_fn ( resamp, axis=1 )
    except TypeError:
        if verbose:
            print('Cannot do array math with input metric function; looping:')
        resampled_metric = np.array([ metric_fn(ix) for ix in resamp ])
    
    emetric = efunc(resampled_metric)
    return emetric

def poissonian_histsample ( r, bins=10, size=2000, **kwargs ):
    y_r,bin_edges = np.histogram(
        r,
        bins=bins,         
        **kwargs
    )
    
    sample = np.zeros([size, len(y_r)])
    for idx in range(len(y_r)):
        sample[:,idx] = stats.poisson.rvs (mu=y_r[idx], size=size)
    return sample,  midpts(bin_edges)
    
def gamma_histcounts (r, bins=10, ci=0.68, weights=None, u_weights=None, return_generator=False, weight_fn=None, **kwargs):
    """
    Compute weighted histogram counts with confidence intervals using a Gamma approximation to Poisson statistics.

    Parameters
    ----------
    r : array_like
        Input data to be histogrammed.
    bins : int or sequence of scalars or str, optional
        Number of bins or bin edges. Passed to `np.histogram`.
    ci : float, optional
        Confidence level for the intervals (e.g., 0.68 for 68% confidence). Default is 0.68.
    weights : array_like, optional
        Weights for each data point in `r`. If None, unweighted counts are used.
    return_generator : bool, optional
        If True, returns a function that generates random realizations of the weighted histogram from the 
        Gamma-distributed posterior. Default is False.
    weight_fn : callable, optional
        A function to estimate the weight to assign to bins with zero weighted counts. If None, uses the median 
        of the provided weights.
    **kwargs
        Additional keyword arguments passed to `np.histogram`.

    Returns
    -------
    tuple
        If `return_generator` is False:
            (lower_limit_weighted, upper_limit_weighted)
        Confidence interval bounds for each bin of the weighted histogram.

        If `return_generator` is True:
            (generator, limits)
        - `generator`: A function `f(n)` that returns `n` random samples from the posterior distribution 
          of the weighted histogram.
        - `limits`: Tuple of confidence bounds as above.

    Notes
    -----
    This function assumes Poisson-distributed counts and uses a Gamma distribution to approximate 
    the posterior distribution of bin counts. The resulting confidence intervals are scaled to reflect 
    weighted data. Bins with zero weighted counts are handled using either the median weight or a user-defined 
    function evaluated at bin centers.
    """   
    if weights is None:
        weights = np.ones(r.shape, dtype=float) 
    if u_weights is None:
        u_weights = (np.zeros_like(weights),np.zeros_like(weights))
    elif not isinstance(u_weights, tuple):
        print('[sampling.gamma_histcounts] Interpreting u_weights as symmetric uncertainties')
        u_weights = (u_weights, u_weights)
        
    alpha = (1. - ci)/2.       
    y_r,_ = np.histogram(
        r,
        bins=bins,         
        **kwargs
    ) 
    raw_counts = y_r.copy()
    

    upper_limit_counts = stats.gamma.ppf(1.-alpha, y_r + 0.5, scale=1) 
    lower_limit_counts = stats.gamma.ppf(alpha, y_r+0.5, scale=1)   

    weighted_hist,bin_edges = np.histogram(
        r,
        bins=bins,  
        weights=weights,       
        **kwargs
    )
    binned_weighted_var_l,_ = np.histogram(
        r,        
        bins=bins,
        weights=u_weights[0]**2,
    )
    binned_weighted_var_h,_ = np.histogram(
        r,        
        bins=bins,
        weights=u_weights[1]**2,
    )
    #from ekfparse import strings
    #tag = strings.random_string(3)
    #np.save(f'/tmp/{tag}_r.npy', r)
    #np.save(f'/tmp/{tag}_weights.npy', weights)
    #np.save(f'/tmp/{tag}_u_weights.npy', u_weights[0])
    #print(f'saved with tag {tag}')
    
    # \\ replace 0 counts with 1 count * average weight
    if weight_fn is None:
        weighted_hist[weighted_hist==0] = np.nanmedian(weights)
    else:       
        weighted_hist[weighted_hist==0] = weight_fn(midpts(bin_edges)[weighted_hist==0])
    y_r[y_r == 0] = 1

    count_err_low = weighted_hist * (1. - lower_limit_counts/y_r)
    count_err_high = weighted_hist * (upper_limit_counts/y_r - 1.)
    # Combine both sources of uncertainty
    lower_uncert = np.sqrt(binned_weighted_var_l + count_err_low**2)
    upper_uncert = np.sqrt(binned_weighted_var_h + count_err_high**2)    

    #print('ABC')
    #print(weighted_hist)
    #print(upper_uncert)
    lower_limit_weighted = weighted_hist - lower_uncert
    upper_limit_weighted = weighted_hist + upper_uncert
    
    # weighted_hist * upper_limit_counts / y_r - weighted_hist
    # weighted_hist * (upper_limit_counts/y_r - 1.)
    #upper_limit_weighted = weighted_hist * upper_limit_counts / y_r
    #lower_limit_weighted = weighted_hist * lower_limit_counts / y_r

    if return_generator:
        limits = (lower_limit_weighted, upper_limit_weighted) 
        generator = lambda n: stats.gamma.rvs (raw_counts+0.5, scale=1, size=(n,len(raw_counts))) * weighted_hist/y_r
        return generator, limits
    else:
        return (lower_limit_weighted, upper_limit_weighted)

        
        

def poissonian_histcounts (r, bins=10, ci=0.68, weights=None, **kwargs):
    alpha = (1. - ci)/2.
    
    y_r,_ = np.histogram(
        r,
        bins=bins,         
        **kwargs
    ) 
    upper_limit_counts = stats.poisson.ppf(1.-alpha, y_r)
    # \\ P(0|mu) = exp(-mu) = 0.05 simplified for N=0
    # \\ mu ~ 2.3  upper limit of one-sided 95% confidence
    ul = -np.log(1.- ci)    
    upper_limit_counts = np.where(np.isclose(upper_limit_counts,0), ul, upper_limit_counts)    
    lower_limit_counts = stats.poisson.ppf(alpha, y_r)
    lower_limit_counts = np.where(lower_limit_counts>0, lower_limit_counts, 1)

    weighted_hist,_ = np.histogram(
        r,
        bins=bins,  
        weights=weights,       
        **kwargs
    )
    # \\ replace 0 counts with 1 count * average weight
    weighted_hist[weighted_hist==0] = np.nanmedian(weights)
    y_r[y_r == 0] = 1

    upper_limit_weighted = weighted_hist * upper_limit_counts / y_r
    lower_limit_weighted = weighted_hist * lower_limit_counts / y_r
    #print("ULC")
    #print(weighted_hist)    
    return (lower_limit_weighted, upper_limit_weighted)    
    
def bootstrap_histcounts(r,bins=10,npull=1000,weights=None,err=0., u_weights=0., **kwargs):    
    y_arr = np.zeros([npull, len(bins)-1])
    for idx in range(npull):
        resample = np.random.choice(np.arange(len(r)),len(r), replace=True)
        if weights is None:
            _weights = np.ones_like(r)
        else:
            _weights = weights
        y_r,_ = np.histogram(
            np.random.normal(r,err)[resample],
            bins=bins, 
            weights=np.random.normal(_weights,u_weights)[resample],
            **kwargs
        ) 
        y_arr[idx] = y_r
    return y_arr
    
def upsample(x,y, npts=3000):
    '''
    simple upsampling
    '''
    domain = np.linspace(x.min(),x.max(),npts)
    return domain, np.interp(domain, x, y)

def build_interpfn ( x, y ):
    fn = interpolate.interp1d ( x, y, bounds_error=False, fill_value=0. )
    return fn

# \\ backwards compatibility
wide_kdeBW = functions.wide_kdeBW 
gaussian = functions.gaussian

def midpts ( bins ):
    return 0.5*(bins[1:]+bins[:-1])

def running_metric (x, y,metric_fn, midpts, dx=None, xerr=None, yerr=None, erronmetric=False, nresamp=100, return_counts=False ):    
    if dx is None:
        dx = np.ones(midpts.size)*np.median(np.diff(midpts))*0.75
    elif isinstance(dx, float) or isinstance(dx, int):
        dx = np.ones(midpts.size)*dx
        
        
    if isinstance(midpts, int):
        midpts = np.linspace(x.min(),x.max(),midpts)
    if erronmetric:
        ystats = np.zeros([midpts.size, 1, 5])
    else:
        ystats = np.zeros([midpts.size, 1])    
    if return_counts:
        counts = np.zeros(midpts.shape[0], dtype=int)
        

    if not erronmetric:
        nresamp = 1 

    carr = np.zeros([nresamp, len(midpts), 1])       
    for rindex in range(nresamp):
        choice = np.random.choice(x.size, x.size, replace=True)
        
        # \\ resample from uncertainties assuming Gaussian, if applicable
        if not erronmetric:
            xpull = x # \\ if no err estimate just use the sample
        elif xerr is not None:
            xpull = np.random.normal(x[choice], xerr)
        else:
            xpull = x[choice]
        
        if not erronmetric:
            ypull = y
        elif yerr is not None:
            ypull = np.random.normal(y[choice], yerr)
        else:
            ypull = y[choice]
            
        for idx in range(midpts.size):
            xmin = midpts[idx] - dx[idx]
            xmax = midpts[idx] + dx[idx]
            mask = (xpull>=xmin)&(xpull<xmax)

            outcome = metric_fn ( ypull[mask] )
            if erronmetric:
                carr[rindex,idx] = outcome
            else:
                ystats[idx] = outcome
    
    # \\ if we did error estimation, summary stats for that estimation
    if erronmetric:
        #ystats = np.nanquantile(carr, [0.025,.16,.5,.84,.95], axis=0).T.reshape(len(midpts),len(qt),5)
        ystats = np.zeros([len(midpts), 1, 5])
        for ix,eqt in enumerate([0.025,.16,.5,.84,.95]):
            ystats[:,:,ix] = np.nanquantile(carr, eqt, axis=0)
    return midpts, ystats, dx

def running_quantile (x, y, midpts, dx=None, xerr=None, yerr=None, qt=0.5, erronqt=False, nresamp=100, return_counts=False ):
    if isinstance(qt, float):
        qt = [qt]    
    if dx is None:
        dx = np.ones(midpts.size)*np.median(np.diff(midpts))*0.75
    elif isinstance(dx, float) or isinstance(dx, int):
        dx = np.ones(midpts.size)*dx
        
        
    if isinstance(midpts, int):
        midpts = np.linspace(x.min(),x.max(),midpts)
    if erronqt:
        ystats = np.zeros([midpts.size, len(qt), 5])
    else:
        ystats = np.zeros([midpts.size, len(qt)])    
    if return_counts:
        counts = np.zeros(midpts.shape[0], dtype=int)
        

    if not erronqt:
        nresamp = 1 

    carr = np.zeros([nresamp, len(midpts), len(qt)])       
    for rindex in range(nresamp):
        choice = np.random.choice(x.size, x.size, replace=True)
        
        # \\ resample from uncertainties assuming Gaussian, if applicable
        if not erronqt:
            xpull = x # \\ if no err estimate just use the sample
        elif xerr is not None:
            xpull = np.random.normal(x[choice], xerr)
        else:
            xpull = x[choice]
        
        if not erronqt:
            ypull = y
        elif yerr is not None:
            ypull = np.random.normal(y[choice], yerr)
        else:
            ypull = y[choice]
            
        for idx in range(midpts.size):
            xmin = midpts[idx] - dx[idx]
            xmax = midpts[idx] + dx[idx]
            mask = (xpull>=xmin)&(xpull<xmax)

            outcome = np.nanquantile ( ypull[mask], qt )
            if erronqt:
                carr[rindex,idx] = outcome
            else:
                ystats[idx] = outcome
    
    # \\ if we did error estimation, summary stats for that estimation
    if erronqt:
        #ystats = np.nanquantile(carr, [0.025,.16,.5,.84,.95], axis=0).T.reshape(len(midpts),len(qt),5)
        ystats = np.zeros([len(midpts), len(qt), 5])
        for ix,eqt in enumerate([0.025,.16,.5,.84,.95]):
            ystats[:,:,ix] = np.nanquantile(carr, eqt, axis=0)
    return midpts, ystats, dx

def binned_metric ( x, y, metric_fn, bins, xerr=None, yerr=None, erronmetric=False, nresamp=100, return_counts=False ):
    if isinstance(bins, int):
        bins = np.linspace( np.nanmin(x), np.nanmax(x), bins )

    assns = np.digitize ( x, bins )    
    
    xmid = midpts ( bins )
    if erronmetric:
        ystats = np.zeros([xmid.size, 1, 5])
    else:
        ystats = np.zeros([xmid.size, 1])
        
    if return_counts:
        counts = np.zeros(xmid.shape[0], dtype=int)
    
    for idx in range(1, bins.size):
        if return_counts:
            counts[idx-1] = (assns==idx).sum()        
        if erronmetric:
            carr = np.zeros([nresamp, 1])
            indices = np.arange(y.size)[assns==idx] 
            for jdx in range(nresamp):                                                               
                if yerr is not None:                    
                    pull_indices = np.random.choice ( indices, indices.size, replace=True  )                    
                    u_pull = yerr[pull_indices]
                    m_pull = y[pull_indices]
                    pull = np.random.normal(m_pull, u_pull)
                else:
                    pull = np.random.choice ( y[assns==idx], size=(assns==idx).sum(), replace=True )
                pulled_ys = metric_fn ( pull )
                carr[jdx] = pulled_ys
            ystats[idx-1] = np.nanquantile(carr,[0.025,.16,.5,.84,.95], axis=0).T
        else:
            ystats[idx-1] = metric_fn ( y[assns==idx] )
            
    if return_counts:
        return xmid, ystats, counts
    return xmid, ystats

def binned_quantile ( x, y, bins, xerr=None, yerr=None, qt=0.5, erronqt=False, nresamp=100, return_counts=False ):
    if isinstance(bins, int):
        bins = np.linspace( np.nanmin(x), np.nanmax(x), bins )
    if isinstance(qt, float):
        qt = [qt]
    assns = np.digitize ( x, bins )    
    
    xmid = midpts ( bins )
    if erronqt:
        ystats = np.zeros([xmid.size, len(qt), 5])
    else:
        ystats = np.zeros([xmid.size, len(qt)])
        
    if return_counts:
        counts = np.zeros(xmid.shape[0], dtype=int)
    
    for idx in range(1, bins.size):
        if return_counts:
            counts[idx-1] = (assns==idx).sum()        
        if erronqt:
            carr = np.zeros([nresamp, len(qt)])
            indices = np.arange(y.size)[assns==idx] 
            for jdx in range(nresamp):                                                               
                if yerr is not None:                    
                    pull_indices = np.random.choice ( indices, indices.size, replace=True  )                    
                    u_pull = yerr[pull_indices]
                    m_pull = y[pull_indices]
                    pull = np.random.normal(m_pull, u_pull)
                else:
                    pull = np.random.choice ( y[assns==idx], size=(assns==idx).sum(), replace=True )
                pulled_ys = np.nanquantile ( pull, qt )
                carr[jdx] = pulled_ys
            ystats[idx-1] = np.nanquantile(carr,[0.025,.16,.5,.84,.95], axis=0).T
        else:
            ystats[idx-1] = np.nanquantile ( y[assns==idx], qt)
            
    if return_counts:
        return xmid, ystats, counts
    return xmid, ystats


def metric_against_orthogonalproj ( x, y, metric_fn, bins, return_proj=False, aggregation_mode = 'running', **kwargs ):
    '''
    Bin against the 
    '''
    ell = y-x
    xc = x + ell/2.
    d = ell/np.sqrt(2.)    
    
    if aggregation_mode == 'running':
        xcmid, qt, dxc = running_metric(
            xc,            
            d,
            metric_fn,
            midpts(bins),
            **kwargs,        
        )   
    elif aggregation_mode == 'binned':
        xcmid, qt = binned_metric(
            xc,
            d,
            metric_fn,
            bins,
            **kwargs,        
        )           
    
    if return_proj:
        if aggregation_mode=='running':
            return xcmid, qt, dxc
        return xcmid, qt
    else:
        xp = xcmid
        yp = (xcmid + qt.T/np.sqrt(2.)).T        
        return xp, yp, dxc
    
def quantiles_against_orthogonalproj ( x, y, bins, return_proj=False, aggregation_mode = 'running', **kwargs ):
    '''
    Bin against the 
    '''
    ell = y-x
    xc = x + ell/2.
    d = ell/np.sqrt(2.)    
    
    if aggregation_mode == 'running':
        xcmid, qt, dxc = running_quantile(
            xc,
            d,
            midpts(bins),
            **kwargs,        
        )   
    elif aggregation_mode == 'binned':
        xcmid, qt = binned_quantile(
            xc,
            d,
            bins,
            **kwargs,        
        )           
    
    if return_proj:
        if aggregation_mode=='running':
            return xcmid, qt, dxc
        return xcmid, qt
    else:
        xp = xcmid
        yp = (xcmid + qt.T/np.sqrt(2.)).T        
        return xp, yp, dxc

def classfraction ( x_classa, x_classb, bins=10, add=False, alpha=0.05, method='jeffreys' ):
    """
    Compute the class fraction and its confidence interval between two datasets.

    Parameters:
        x_classa (array-like): The values of the first dataset, typically representing a class A.
        x_classb (array-like): The values of the second dataset, typically representing a class B.
        bins (int, array-like, optional): The number of bins or bin edges to use for histogram binning.
            If int, it represents the number of equal-width bins. If array-like, it provides custom bin edges.
            Default is 10.
        add (bool, optional): If True, the fraction of class A and class B will be summed to compute the class fraction.
            If False, the fraction of class B will be used as the denominator.
            Default is False.
        alpha (float, optional): The significance level for the confidence interval. Must be between 0 and 1.
            Default is 0.05.
        method (str, optional): The method to compute the confidence interval. Possible values are 'jeffreys' and 'beta'.
            Default is 'jeffreys'.

    Returns:
        tuple: A tuple containing three arrays:
            - bin_edges (array): The bin edges used for histogram binning.
            - class_fraction (array): The computed class fraction for each bin.
            - confidence_interval (array): The confidence interval of the class fraction for each bin.
    """    
    if isinstance(bins, (float,int)):
        bins = np.linspace ( min(np.nanmin(x_classa),np.nanmin(x_classb)),
                             max(np.nanmax(x_classa),np.nanmax(x_classb)),
                             bins )
    hista, bin_edges = np.histogram(x_classa, bins=bins)
    histb, _         = np.histogram(x_classb, bins=bins)
    if add:
        denom = hista + histb
    else:
        denom = histb
    confidence_interval = proportion_confint ( hista, denom, alpha=alpha, method=method)
    return bin_edges, hista/denom, confidence_interval



def running_sigmaclip ( x, y, low=4., high=4., xbins=None, Nbins=30, apply=True ):
    if xbins is None:
        xbins = np.linspace(np.nanmin(x), np.nanmax(x), Nbins )
    dx = np.mean(np.diff(xbins))*1.5
    xmid = midpts(xbins)
    
    cuts = np.zeros([len(xbins)-1, 3])
    cuts[:,0] = xmid
    for idx,cx in enumerate(xmid):
        mask = (x > (cx-dx))&(x < (cx+dx))
        clip = sigmaclip(y[mask], low=low, high=high)
        cuts[idx,1] = clip.lower
        cuts[idx,2] = clip.upper

    survivor_mask = (y>np.interp(x, cuts[:,0], cuts[:,1]))&(y<np.interp(x, cuts[:,0], cuts[:,2]))    
    if apply:    
        return x[survivor_mask], y[survivor_mask]
    else:
        return survivor_mask, cuts
    
    

def sigmaclipped_std ( x, low=4., high=4. ):
    return sigmaclip(x,low,high).clipped.std()

def gelmanrubin ( chains ):
    '''
    Gelman-Rubin statistic to quantify chain convergence.
    '''
    # assume chains are emcee-like, i.e. dimensions are STEP/WALKER/PARAMETER
    mean_val_of_chains = np.mean(chains,axis=0)    
    mean_of_chainmeans = np.mean(mean_val_of_chains, axis=0)

    nsteps = chains.shape[0]
    nwalkers = chains.shape[1]
    nparam = chains.shape[2]
    gr_B = nsteps/(nwalkers-1)*np.sum((mean_val_of_chains-mean_of_chainmeans)**2,axis=(0,1))
    mvals = mean_val_of_chains.reshape(1,*mean_val_of_chains.shape)
    gr_W = nwalkers**-1*((nsteps-1)**-1*np.sum((chains-mvals)**2,axis=(0,1)))
    
    gr_R = (nsteps-1)/nsteps * gr_W + nsteps**-1 * gr_B
    gr_R /= gr_W
    
    assert gr_R.size == nparam
    return gr_R

def bin_by_count ( x, min_count, dx_min=0. ):
    x = fmasker(x)
    sortmask = np.argsort(x)
    return_to_orig = np.argsort(sortmask)    
    sorted_x = x[sortmask]
    
    bin_edges = [np.min(x) - abs(np.min(x))*1e-4]
    while bin_edges[-1] < np.max(x):
        bin_start = bin_edges[-1]
        above_cut = sorted_x[sorted_x>=bin_start]
        #print(min(min_count-1, len(above_cut)-1))
        if(len(above_cut)-2)<min_count: # - 2 so that we don't end up wit a last bin of one item
            # \\ hit the end of the sample
            bin_edges.pop()
            bin_edges.append(max(x) + abs(max(x))*1e-4)            
        else:
            dx_count = above_cut[min_count] - bin_start
            
            if dx_count > dx_min:
                bin_edges.append(bin_start + dx_count)
            else:
                bin_edges.append(bin_start + dx_min)
        
    
    assns = np.digitize(x, bin_edges) 
    return assns, np.array(bin_edges)
        
    
def strict_bin_by_count ( x, min_count ):
    sortmask = np.argsort(x)
    return_to_orig = np.argsort(sortmask)

    sorted_x = x[sortmask]
    #sorted_y = y[sortmask]
    bin_assignments = np.repeat(np.arange(0,len(sorted_x)//min_count), min_count)
    bin_assignments = np.concatenate([bin_assignments, np.ones(len(sorted_x)-len(bin_assignments))*bin_assignments[-1]])
    binned = [ sorted_x[bin_assignments==idx] for idx in np.arange(0,bin_assignments[-1]+1) ] 
    bin_edges = [ np.min(sorted_x[bin_assignments==idx]) for idx in np.arange(0,bin_assignments[-1]+1) ] 
    bin_edges = np.array(bin_edges + [np.max(sorted_x[bin_assignments==max(bin_assignments)])]) 
    
    return bin_assignments[return_to_orig].astype(int), bin_edges