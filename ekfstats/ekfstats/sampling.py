import numpy as np
from scipy import optimize
from scipy import interpolate,integrate
from scipy.stats import gaussian_kde
from statsmodels.stats.proportion import proportion_confint
from . import functions

def c_density ( x, y, return_fn=False, clean=True, nmax=None, nmin=30, **kwargs ):
    '''
    Compute gKDE density based on sample
    '''
    # Calculate the point density
    if x.size < nmin:        
        return np.ones_like(x)
    if clean:
        fmask = functions.finite_masker ( [x, y] )
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

def rejection_sample ( x, pdf_x, nsamp=10000, maxiter=100, oversample=5 ):    
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

def sample_from_pdf ( var, prob, nsamp=100, is_bounds=False, spacing='linear', ngrid=10000,):        
    if not isinstance(var, list):        
        if not is_bounds:
            assert np.std(np.diff(var))/np.mean(np.diff(var)) <  1e-5
        else:
            if spacing == 'linear':                
                var = np.linspace(*var,ngrid)
                prob = prob(var) 
            elif spacing == 'log':
                var = np.logspace(*var,ngrid)
                prob = prob(var) * var * np.log(10.)                   
        return np.random.choice( var, p=prob/prob.sum(), size=nsamp)
    else:
        indices = np.arange(var[0].size)#.reshape(var[0].shape)
        flattened_prob = prob.flatten()
        flattened_prob /= flattened_prob.sum()
        pulled_indices = np.random.choice ( indices, p=flattened_prob, size=nsamp )
        coords = np.zeros ( [len(var), nsamp])
        for idx in range(len(var)):
            coords[idx] = var[idx].flatten()[pulled_indices]
        return coords


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
        if erronqt:
            carr = np.zeros([nresamp, len(qt)])
            indices = np.arange(y.size)[assns==idx] 
            if return_counts:
                counts[idx-1] = (assns==idx).sum()
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