import numpy as np
from scipy import optimize
from scipy import interpolate,integrate

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

def pdf_product ( xA, pdfA, xB, pdfB, npts=100, normalize=True ):
    '''
    return an approximation of the probability density function that describes
    the product of two PDFs A & B
    '''    
    product = cross_then_flat(xA,xB)
    probdensity    = cross_then_flat(pdfA,pdfB)
    xmin,xmax = np.quantile(product, [0.,1.])
    domain = np.linspace(xmin,xmax, npts)

    assns       = np.digitize ( product, domain )            
    pdf         = np.array([np.sum(probdensity[assns==x]) for x in np.arange(1, domain.shape[0]+1)])
    if normalize:
        nrml        = np.trapz(pdf, domain)
        pdf        /= nrml
    interpfn = build_interpfn( domain, pdf )
    return interpfn

def cross_then_flat ( a, b):
    '''
    [ b0 ] x [ a0 a1 a2 a3 ] 
    | b1 |
    | b2 |
    [ b3 ]                
    '''
    return np.asarray(np.matrix(a).T*np.matrix(b)).flatten()

def upsample(x,y, npts=1000):
    '''
    simple upsampling
    '''
    domain = np.linspace(x.min(),x.max(),npts)
    return domain, np.interp(domain, x, y)

def build_interpfn ( x, y ):
    fn = interpolate.interp1d ( x, y, bounds_error=False, fill_value=0. )
    return fn

def wide_kdeBW ( size, alpha=3. ):
    bw = alpha*size**(-1./5.)    