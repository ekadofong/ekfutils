import numpy as np
from numba import njit, prange
from scipy import linalg
from scipy.special import erf

def wide_kdeBW ( size, alpha=3. ):
    bw = alpha*size**(-1./5.)  
    return bw  

def gaussian ( x, A, m, s):
    '''
    1D Gaussian. 
    '''
    if A == 'normalize':
        A = np.sqrt(2.*np.pi * s**2)**-1    
    return A * np.exp ( -(x-m)**2 / (2.*s**2) )

def sigmoid ( sigmoid_x, bound, k ):
    '''
    Logistic function.
    '''
    sigmoid_y = ( 1. + np.exp(-k*(sigmoid_x - bound)))**-1
    return sigmoid_y

def powerlaw ( x, index, normalization ):
    return normalization * x**index

def schechter ( m, phi_ast, M_ast, alpha ):
    '''                                                                         
    phi(M) = dN/dM                                                              
    '''
    nd = phi_ast/np.log10(M_ast) * (m/M_ast)**(alpha) * np.e**(-m/M_ast)
    return nd

def logschechter ( m, phi_ast, M_ast, alpha ):
    '''                                                                         
    phi(M) = dN/d(log_10{M})                                                    
    '''
    nd = np.log(10.)*phi_ast *(m/M_ast)**(alpha+1.)*np.e**(-m/M_ast)
    return nd

def logschechter_mag ( m, phi_ast, M_ast, alpha ):
    # \\ magnitude version from Loveday+11
    nd = 0.4*np.log(10.)*phi_ast*(10.**(0.4*(M_ast - m)))**(1.+alpha)*np.exp(-10.**(0.4*(M_ast-m)))    
    return nd

def logschechter_alog ( logm, phi_ast, logM_ast, alpha ):
    '''                                                                         
    phi(log_10{M}) = dN/d(log_10{M})                                            
    '''
    t0 = np.log(10.)*phi_ast
    t1 = 10.**((logm - logM_ast)*(alpha+1.))
    t2 = np.e**(-10.**(logm-logM_ast))
    nd =t0*t1*t2
    return nd

def log_uncertainty ( n, u_n ):
    # X = log10(n)
    # v_X = (dX/dn)^2 v_n
    #     =  (ln10 n)^-2 v_n
    return abs(u_n/(np.log(10.)*n))

def logratio_uncertainty ( n, u_n, d, u_d ):
    # X = log10 ( n / d)
    # v_X = (1/(n/d ln10) 1/d)^2 v_n + ( 1/(n/d ln10) n/d^2 )^2 v_d
    # v_X = (d/[n ln10])^2 * ( v_n / d^2 + n^2/d^4 v_d )
    #     = (n ln10)^-2 * ( v_n + n_2/d^2 v_d )
    return np.sqrt ( (n * np.log(10.))**-2. * ( u_n**2  + n**2/d**2 * u_d**2 ) )

def log_of_uniform ( xpos, xmax ):
    '''
    NOT loguniform; this distribution goes as
    X ~ 1/(xmax) 10^X ln10
    and is the distribution in logspace of a uniform distribution, i.e.
    where X := log10(x)
    '''
    return xmax**-1 * 10.**xpos * np.log(10.)

def skewnormal ( t, xi, w, a ):
    '''
    skewed normal. 
    xi = location 
    w = scale
    a = skew
    '''
    prefactor = np.sqrt(2.*np.pi*w**2)**-1
    t0 = np.exp(-(t-xi)**2 / (2.*w**2))
    t1 = 1. + erf( a*(t-xi)/np.sqrt(2.*w**2) )
    return prefactor * t0 * t1

def laplace_pdf(x, mu, b):
    """
    Calculate the probability density function of the Laplace distribution.

    Parameters:
    x : float or numpy array
        Point(s) at which to evaluate the PDF.
    mu : float
        Location parameter of the Laplace distribution.
    b : float
        Scale parameter of the Laplace distribution.

    Returns:
    float or numpy array
        The probability density at each point x.
    """
    return (1 / (2 * b)) * np.exp(-np.abs(x - mu) / b)

def cauchy_pdf(x, x0, gamma):
    """
    Calculate the probability density function of the Cauchy distribution.

    Parameters:
    x : float or numpy array
        Point(s) at which to evaluate the PDF.
    x0 : float
        Location parameter of the Cauchy distribution.
    gamma : float
        Scale parameter of the Cauchy distribution.

    Returns:
    float or numpy array
        The probability density at each point x.
    """
    return 1 / (np.pi * gamma * (1 + ((x - x0) / gamma)**2))


def log_multivariate_gaussian(x, mu, V, Vinv=None, method=1):
    """
    EXTRACTION OF astroML.utils.log_multivariate_gaussian
    by https://github.com/astroML/astroML
    
    Evaluate a multivariate gaussian N(x|mu, V)

    This allows for multiple evaluations at once, using array broadcasting

    Parameters
    ----------
    x: array_like
        points, shape[-1] = n_features

    mu: array_like
        centers, shape[-1] = n_features

    V: array_like
        covariances, shape[-2:] = (n_features, n_features)

    Vinv: array_like or None
        pre-computed inverses of V: should have the same shape as V

    method: integer, optional
        method = 0: use cholesky decompositions of V
        method = 1: use explicit inverse of V

    Returns
    -------
    values: ndarray
        shape = broadcast(x.shape[:-1], mu.shape[:-1], V.shape[:-2])

    Examples
    --------

    >>> x = [1, 2]
    >>> mu = [0, 0]
    >>> V = [[2, 1], [1, 2]]
    >>> log_multivariate_gaussian(x, mu, V)
    -3.3871832107434003
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    V = np.asarray(V, dtype=float)

    ndim = x.shape[-1]
    x_mu = x - mu

    if V.shape[-2:] != (ndim, ndim):
        raise ValueError("Shape of (x-mu) and V do not match")

    Vshape = V.shape
    V = V.reshape([-1, ndim, ndim])

    if Vinv is not None:
        assert Vinv.shape == Vshape
        method = 1

    if method == 0:
        Vchol = np.array([linalg.cholesky(V[i], lower=True)
                          for i in range(V.shape[0])])

        # we may be more efficient by using scipy.linalg.solve_triangular
        # with each cholesky decomposition
        VcholI = np.array([linalg.inv(Vchol[i])
                          for i in range(V.shape[0])])
        logdet = np.array([2 * np.sum(np.log(np.diagonal(Vchol[i])))
                           for i in range(V.shape[0])])

        VcholI = VcholI.reshape(Vshape)
        logdet = logdet.reshape(Vshape[:-2])

        VcIx = np.sum(VcholI * x_mu.reshape(x_mu.shape[:-1]
                                            + (1,) + x_mu.shape[-1:]), -1)
        xVIx = np.sum(VcIx ** 2, -1)

    elif method == 1:
        if Vinv is None:
            Vinv = np.array([linalg.inv(V[i])
                             for i in range(V.shape[0])]).reshape(Vshape)
        else:
            assert Vinv.shape == Vshape

        logdet = np.log(np.array([linalg.det(V[i])
                                  for i in range(V.shape[0])]))
        logdet = logdet.reshape(Vshape[:-2])

        xVI = np.sum(x_mu.reshape(x_mu.shape + (1,)) * Vinv, -2)
        xVIx = np.sum(xVI * x_mu, -1)

    else:
        raise ValueError("unrecognized method %s" % method)

    return -0.5 * ndim * np.log(2 * np.pi) - 0.5 * (logdet + xVIx)

@njit(parallel=True)
def fast_log_multivariate_gaussian(x, mu, V, Vinv=None, method=1):
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    ndim = x.shape[-1]
    x_mu = x - mu

    if V.shape[-2:] != (ndim, ndim):
        raise ValueError("Shape of (x-mu) and V do not match")

    Vshape = V.shape
    V_reshaped = V.reshape(-1, ndim, ndim)
    n_matrices = V_reshaped.shape[0]

    logdet = np.empty(n_matrices, dtype=np.float64)
    xVIx = np.empty(n_matrices, dtype=np.float64)

    if method == 1:
        if Vinv is None:
            Vinv = np.empty_like(V_reshaped)
            for i in prange(n_matrices):
                Vinv[i] = np.linalg.inv(V_reshaped[i])
        else:
            Vinv = Vinv.reshape(-1, ndim, ndim)

        for i in prange(n_matrices):
            logdet[i] = np.log(np.linalg.det(V_reshaped[i]))
            x_mu_i = x_mu[i]
            xVI = np.dot(Vinv[i], x_mu_i)
            xVIx[i] = np.dot(x_mu_i, xVI)

    else:
        raise ValueError("Only method=1 is supported in fast version.")

    const = -0.5 * ndim * np.log(2 * np.pi)
    return const - 0.5 * (logdet + xVIx)
