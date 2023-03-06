import numpy as np

def finite_masker ( arr0, arr1, inplace=False ):
    '''
    Returns a mask that is True where both input arrays 
    are finite-valued
    '''
    mask = np.isfinite(arr0)&np.isfinite(arr1)
    if inplace:
        masked_arr0 = arr0[mask]
        masked_arr1 = arr1[mask]
        return masked_arr0, masked_arr1
    else:
        return mask

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
    Sigmoid function.
    '''
    sigmoid_y = ( 1. + np.exp(-k*(sigmoid_x - bound)))**-1
    return sigmoid_y

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