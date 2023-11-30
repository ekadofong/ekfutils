import numpy as np

def finite_masker ( arr_l, inplace=False ):
    '''
    Returns a mask that is True where both input arrays 
    are finite-valued
    '''
    if not isinstance(arr_l, list):
        arr_l = [arr_l]
    mask = np.isfinite(arr_l[0])
    for arr in arr_l[1:]:
        mask &= np.isfinite(arr)
    if inplace:
        arr_out = []
        for arr in arr_l:
            arr_out.append(arr[mask])
        return arr_out
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
