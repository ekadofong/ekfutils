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

def mask ( mask, *args ):
    masked_args = []
    for arg in args:
        masked_args.append(arg[mask])
    return fmasker(*masked_args)

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

def compare_parameter_posteriors(posterior_sample_a, posterior_sample_b, step=0.01, alpha_init=0.5):
    """
    Compare two posterior samples to quantify the level of agreement.

    This function computes the smallest symmetric credible interval (centered on the median) 
    of the difference between two posterior samples that contains zero. The returned value 
    indicates the minimum credible level (alpha) at which the two posteriors are 
    statistically consistent (i.e., their difference includes zero).

    Parameters
    ----------
    posterior_sample_a : array-like
        Samples from the first posterior distribution.
    posterior_sample_b : array-like
        Samples from the second posterior distribution. Must be the same shape as `posterior_sample_a`.

    step : float, optional
        Increment step size for increasing the credible interval width (default is 0.01).

    Returns
    -------
    alpha : float
        The smallest credible level such that zero lies within the corresponding symmetric 
        credible interval of the posterior difference. A lower value indicates higher 
        consistency between the two posteriors.

    Notes
    -----
    This method assumes the input samples are 1D arrays and come from distributions 
    where the difference is approximately unimodal. For multimodal distributions, results 
    may be less interpretable.
    """

    delta = posterior_sample_a - posterior_sample_b
    
    alpha = alpha_init
    while True:
        bounds = np.quantile(delta, [(1.-alpha)/2., 0.5 + alpha/2.])
        if (0. > bounds[0])&(0. < bounds[1]):
            break
        alpha += step
    return alpha
            

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
    
def gamma_histcounts(r, bins=10, ci=0.68, weights=None, u_weights=None,
                     return_generator=False, weight_fn=None, **kwargs):
    """
    Compute confidence intervals for a weighted histogram using a Gamma approximation
    to the Poisson distribution.

    Parameters
    ----------
    r : array_like
        Input data to be histogrammed.
    bins : int, sequence of scalars, or str, optional
        Number of bins or bin edges; passed to `np.histogram`.
    ci : float, optional
        Confidence level (e.g., 0.68 for ~1sigma). Default is 0.68.
    weights : array_like, optional
        Weights associated with each entry in `r`. Defaults to 1 for each point.
    u_weights : array_like or tuple of arrays, optional
        Uncertainties on weights. Can be a single array (assumed symmetric), or a tuple
        (lower uncertainties, upper uncertainties). Default is zero uncertainty.
    return_generator : bool, optional
        If True, also return a sampler function that generates realizations from the
        posterior distribution of histogram counts.
    weight_fn : callable, optional
        Function to assign weights to bins with zero total weight. Defaults to using
        the median of provided weights.
    **kwargs
        Additional arguments passed to `np.histogram`.

    Returns
    -------
    If return_generator is False:
        (lower, upper) : tuple of arrays
            Lower and upper confidence limits on the weighted histogram counts.
    If return_generator is True:
        (generator, (lower, upper)) : tuple
            - generator : callable, returns samples from posterior of each bin.
            - (lower, upper) : confidence interval bounds.

    Notes
    -----
    - Assumes Poisson counting with Gamma posterior for each bin.
    - Handles zero-weight bins using user-defined or fallback substitution.
    - Final error bars include both Poisson (counting) and weight uncertainty.
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
    
    # \\ replace 0 counts with 1 count * average weight
    if weight_fn is None:
        weighted_hist[weighted_hist==0] = np.nanmedian(weights)
    else:       
        weighted_hist[weighted_hist==0] = weight_fn(midpts(bin_edges)[weighted_hist==0])
    y_r[y_r == 0] = 1

    # \\ for each bin i
    # \\ sum(weights) * ( 1 - N_low / N )
    # \\ W/N ( N - N_low )
    # \\ (average weight) * (counts - lower_limit_on_counts )
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

def running_ridgeline (xc,yc,midpts, dx=None, **kwargs):
    # Compute the ridgeline (mode of d) in each bin
    xc,yc = fmasker(xc,yc)
    #bin_edges = np.asarray(bins)
    #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if dx is None:
        dx = np.median(np.diff(midpts))/2.
    ridgeline_vals = []

    for idx in range(len(midpts)):
        lo = midpts[idx] - dx
        hi = midpts[idx] + dx
        in_bin = (xc >= lo) & (xc < hi)
        if np.sum(in_bin) < 5:
            ridgeline_vals.append(np.nan)
            continue

        d_in_bin = yc[in_bin]
        try:
            kde = gaussian_kde(d_in_bin, bw_method=kwargs.get("bw_method", "scott"))
            d_grid = np.linspace(np.min(d_in_bin), np.max(d_in_bin), 500)
            mode_val = d_grid[np.argmax(kde(d_grid))]            
        except KeyboardInterrupt:
            mode_val = np.nan

        ridgeline_vals.append(mode_val)

    qt = np.array(ridgeline_vals)
    return midpts, qt

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
    elif aggregation_mode == 'ridgeline':
        # Compute the ridgeline (mode of d) in each bin
        bin_edges = np.asarray(bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ridgeline_vals = []

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            in_bin = (xc >= lo) & (xc < hi)
            if np.sum(in_bin) < 5:
                ridgeline_vals.append(np.nan)
                continue

            d_in_bin = d[in_bin]
            try:
                kde = gaussian_kde(d_in_bin, bw_method=kwargs.get("bw_method", "scott"))
                d_grid = np.linspace(np.min(d_in_bin), np.max(d_in_bin), 500)
                mode_val = d_grid[np.argmax(kde(d_grid))]
            except Exception:
                mode_val = np.nan

            ridgeline_vals.append(mode_val)

        xcmid = bin_centers
        qt = np.array(ridgeline_vals)
        dxc = np.diff(bins)                 
    else:
        raise ValueError(f"Unknown aggregation_mode: {aggregation_mode}")
        
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
    return sigmaclip(fmasker(x),low,high).clipped.std()

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


def vec_to_cov_matrix(cov, D):
    """
    Convert a flattened upper triangular vector of a DxD covariance matrix
    into the full symmetric covariance matrix.
    
    Parameters:
    - cov: list or array of length D*(D+1)//2, containing the upper triangle
           of the covariance matrix, row-wise.
    - D: dimension of the desired square covariance matrix
    
    Returns:
    - A (D x D) numpy array representing the full symmetric covariance matrix.
    """
    if len(cov) != D * (D + 1) // 2:
        raise ValueError("Length of cov does not match D*(D+1)//2")

    full_cov = np.zeros((D, D))
    idx = 0
    for i in range(D):
        for j in range(i, D):
            full_cov[i, j] = cov[idx]
            full_cov[j, i] = cov[idx]  # symmetry
            idx += 1
    return full_cov

import numpy as np

def vecs_to_cov_matrices(flat_covs, D):
    """
    Convert a flattened list of N upper-triangular covariance matrices into
    full symmetric (N x D x D) array.

    Parameters:
    - flat_covs: list or array of length N * (D*(D+1)//2), containing upper triangles
                 of N covariance matrices, row-wise.
    - D: dimension of each square covariance matrix

    Returns:
    - A numpy array of shape (N, D, D) representing all covariance matrices.
    """
    num_elements_per_matrix = D * (D + 1) // 2
    if len(flat_covs) % num_elements_per_matrix != 0:
        raise ValueError("Length of flat_covs is not divisible by D*(D+1)//2")

    N = len(flat_covs) // num_elements_per_matrix
    flat_covs = np.asarray(flat_covs)
    cov_matrices = np.zeros((N, D, D))

    for n in range(N):
        idx_start = n * num_elements_per_matrix
        idx_end = idx_start + num_elements_per_matrix
        cov = flat_covs[idx_start:idx_end]
        cov_matrices[n] = vec_to_cov_matrix(cov, D)

    return cov_matrices

def cov_matrices_to_vecs(cov_matrices):
    """
    Convert a (N x D x D) array of symmetric covariance matrices into
    a flattened list of their upper-triangular parts (row-wise).

    Parameters:
    - cov_matrices: numpy array of shape (N, D, D)

    Returns:
    - flat_covs: list of length N * (D*(D+1)//2), containing upper triangle
                 of each matrix in row-wise order.
    """
    N, D, D2 = cov_matrices.shape
    if D != D2:
        raise ValueError("Input matrices must be square (D x D)")

    flat_covs = []
    for n in range(N):
        for i in range(D):
            for j in range(i, D):
                flat_covs.append(cov_matrices[n, i, j])
    return flat_covs

def generate_psd_from_correlation(diag_vals):
    """
    Generate a random symmetric positive semi-definite (PSD) matrix 
    with a specified diagonal.

    The resulting matrix is constructed by:
      1. Generating a random correlation matrix from a random lower 
         triangular matrix via Cholesky-like decomposition.
      2. Scaling the correlation matrix so that its diagonal matches 
         the provided `diag_vals`.

    Parameters
    ----------
    diag_vals : array_like
        A 1D array or list of positive values specifying the desired 
        diagonal entries of the resulting PSD matrix.

    Returns
    -------
    A : ndarray of shape (n, n)
        A symmetric positive semi-definite matrix with diagonal 
        approximately equal to `diag_vals`.

    Notes
    -----
    - The matrix is generated using a random correlation matrix, so 
      the off-diagonal structure is arbitrary and varies each call.
    - The resulting matrix satisfies A ≈ D^{1/2} C D^{1/2}, where 
      D = diag(diag_vals) and C is a random correlation matrix.
    """    
    n = len(diag_vals)
    L = np.tril(np.random.randn(n, n))
    C = L @ L.T
    C /= np.outer(np.sqrt(np.diag(C)), np.sqrt(np.diag(C)))  # convert to correlation
    d_sqrt = np.sqrt(diag_vals)
    A = C * np.outer(d_sqrt, d_sqrt)
    return A

def generate_nonnegative_psd_from_correlation(diag_vals):
    """
    Generate a symmetric positive semi-definite (PSD) matrix with a given diagonal
    and only nonnegative entries (including covariances).

    This is achieved by:
      1. Generating a random correlation matrix with only nonnegative entries.
      2. Scaling it such that the final matrix has the specified diagonal.

    Parameters
    ----------
    diag_vals : array_like
        A 1D array or list of positive values specifying the desired diagonal
        of the resulting matrix.

    Returns
    -------
    A : ndarray of shape (n, n)
        A symmetric PSD matrix with diagonal approximately equal to `diag_vals`,
        and all elements A[i, j] >= 0.

    Notes
    -----
    - The resulting matrix A is of the form A = D^{1/2} C D^{1/2}, where
      D = diag(diag_vals) and C is a correlation matrix with only nonnegative entries.
    - The correlation matrix is generated using a nonnegative random matrix,
      which may bias the off-diagonal entries toward larger values.
    """
    diag_vals = np.asarray(diag_vals)
    n = len(diag_vals)

    # Generate a nonnegative random matrix and symmetrize it
    L = np.tril(np.abs(np.random.randn(n, n)))
    C = L @ L.T

    # Normalize to correlation matrix with unit diagonal
    C /= np.outer(np.sqrt(np.diag(C)), np.sqrt(np.diag(C)))

    # Ensure all elements are nonnegative due to construction
    assert np.all(C >= 0)

    # Scale to match desired diagonal
    d_sqrt = np.sqrt(diag_vals)
    A = C * np.outer(d_sqrt, d_sqrt)

    return A


from scipy.stats import truncnorm

def truncated_normal_sample(mean, s, lower, upper):
    """
    Sample from a truncated normal distribution with arbitrary-shaped mean and std arrays.

    Parameters
    ----------
    mean : array_like
        Mean(s) of the normal distribution(s).
    s : array_like
        Standard deviation(s) of the normal distribution(s).
    lower : array_like or float
        Lower bound(s) for truncation. Can be scalar or array of same shape as mean.
    upper : array_like or float
        Upper bound(s) for truncation. Can be scalar or array of same shape as mean.

    Returns
    -------
    samples : ndarray
        Samples from the specified truncated normal distribution(s), with same shape as `mean`.
    """
    mean = np.asarray(mean)
    s = np.asarray(s)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    # Ensure all arrays can broadcast together
    shape = np.broadcast_shapes(mean.shape, s.shape, np.shape(lower), np.shape(upper))

    # Broadcast all inputs
    mean = np.broadcast_to(mean, shape)
    s = np.broadcast_to(s, shape)
    lower = np.broadcast_to(lower, shape)
    upper = np.broadcast_to(upper, shape)

    # Compute standardized bounds
    a = (lower - mean) / s
    b = (upper - mean) / s

    return truncnorm.rvs(a, b, loc=mean, scale=s)




def asymmetric_truncnorm_sample(mean, sigma_lower, sigma_upper, lower_limit=-np.inf, upper_limit=np.inf, size=1):
    """
    Generate samples from an asymmetric truncated normal distribution.
    
    The distribution is constructed by stitching together two normal distributions
    about the mean, with different standard deviations on either side.
    
    Parameters:
    -----------
    mean : float or array_like
        Mean of the distribution(s)
    sigma_upper : float or array_like
        Standard deviation for x > mean
    sigma_lower : float or array_like
        Standard deviation for x <= mean
    lower_limit : float or array_like
        Lower truncation bound
    upper_limit : float or array_like
        Upper truncation bound
    size : int or tuple of ints, optional
        Number of samples to generate. If parameters are arrays of length N,
        this generates size samples for each of the N distributions.
        
    Returns:
    --------
    samples : ndarray
        Array of samples with shape (size, N) if parameters are arrays,
        or (size,) if parameters are scalars
    """
    # Convert inputs to numpy arrays
    mean = np.asarray(mean)
    sigma_upper = np.asarray(sigma_upper)
    sigma_lower = np.asarray(sigma_lower)
    lower_limit = np.asarray(lower_limit)
    upper_limit = np.asarray(upper_limit)
    
    # Ensure all parameters have the same shape
    params = np.broadcast_arrays(mean, sigma_upper, sigma_lower, lower_limit, upper_limit)
    mean, sigma_upper, sigma_lower, lower_limit, upper_limit = params
    
    # Flatten for easier processing
    orig_shape = mean.shape
    mean_flat = mean.flatten()
    sigma_upper_flat = sigma_upper.flatten()
    sigma_lower_flat = sigma_lower.flatten()
    lower_limit_flat = lower_limit.flatten()
    upper_limit_flat = upper_limit.flatten()
    
    n_dists = len(mean_flat)
    
    # Initialize output array
    if orig_shape == ():
        output_shape = (size,)
    else:
        output_shape = (size, n_dists)
    
    samples = np.zeros(output_shape)
    
    for i in range(n_dists):
        mu = mean_flat[i]
        s_upper = sigma_upper_flat[i]
        s_lower = sigma_lower_flat[i]
        a = lower_limit_flat[i]
        b = upper_limit_flat[i]
        
        # Calculate the probability mass on each side of the mean
        # For the lower side (x <= mean)
        if a <= mu:
            # Standardized bounds for lower distribution
            a_lower_std = (a - mu) / s_lower
            b_lower_std = 0  # At the mean
            
            # CDF values for the lower truncated normal
            lower_cdf_a = truncnorm.cdf(a_lower_std, a_lower_std, np.inf)
            lower_cdf_b = truncnorm.cdf(b_lower_std, a_lower_std, np.inf)
            p_lower = lower_cdf_b - lower_cdf_a
        else:
            p_lower = 0
            
        # For the upper side (x > mean)
        if b > mu:
            # Standardized bounds for upper distribution
            a_upper_std = 0  # At the mean (exclusive)
            b_upper_std = (b - mu) / s_upper
            
            # CDF values for the upper truncated normal
            upper_cdf_a = truncnorm.cdf(a_upper_std, -np.inf, b_upper_std)
            upper_cdf_b = truncnorm.cdf(b_upper_std, -np.inf, b_upper_std)
            p_upper = upper_cdf_b - upper_cdf_a
        else:
            p_upper = 0
            
        # Normalize probabilities
        p_total = p_lower + p_upper
        if p_total == 0:
            raise ValueError(f"No valid range for distribution {i}: mean={mu}, bounds=[{a}, {b}]")
            
        p_lower /= p_total
        p_upper /= p_total
        
        # Generate samples
        for j in range(size):
            # Decide which side to sample from
            if np.random.random() < p_lower and a <= mu:
                # Sample from lower side
                a_std = max((a - mu) / s_lower, -10)  # Clip for numerical stability
                b_std = 0
                sample_std = truncnorm.rvs(a_std, b_std, loc=0, scale=1)
                sample = mu + sample_std * s_lower
            else:
                # Sample from upper side
                a_std = 0
                b_std = min((b - mu) / s_upper, 10)  # Clip for numerical stability
                sample_std = truncnorm.rvs(a_std, b_std, loc=0, scale=1)               
                sample = mu + sample_std * s_upper
            
            if orig_shape == ():
                samples[j] = sample
            else:
                samples[j, i] = sample
    
    # Reshape output if needed
    if orig_shape != () and len(orig_shape) > 0:
        if size == 1:
            samples = samples.reshape(orig_shape)
        else:
            samples = samples.reshape((size,) + orig_shape)
    
    return samples


