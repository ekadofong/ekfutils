import numpy as np
from scipy import optimize
from scipy import interpolate,integrate
from . import functions

def rejection_sample_fromarray ( x, y, nsamp=10000 ):
    pdf_x = interpolate.interp1d(x,y,bounds_error=False, fill_value=0.)
    sample = rejection_sample ( x, pdf_x, nsamp )
    return sample

def rejection_sample ( x, pdf_x, nsamp=10000 ):    
    '''
    Do rejection sampling for a probability density function
    that is known over a set of points x
    '''
    neg_fn = lambda x: -pdf_x(x)
    pout = optimize.minimize(neg_fn, x.mean() )
    max_pdf = pdf_x(pout.x)
    
    sample = np.zeros(nsamp)
    nadded = 0    
    while nadded < nsamp:
        #idx_draw = np.random.randint(0, x.size, size=5*nsamp)
        x_draw = np.random.uniform(x.min(),x.max(),5*nsamp)
        pdf_at_draw = pdf_x(x_draw)
        uni_at_draw = np.random.uniform(0.,max_pdf,x_draw.size)
        keep = pdf_at_draw >= uni_at_draw        
        to_add = x_draw[keep][:(nsamp-nadded)]
        sample[nadded:(nadded+to_add.size)] = to_add
        nadded += keep.sum()       
    return sample  

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
