import numpy as np
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
