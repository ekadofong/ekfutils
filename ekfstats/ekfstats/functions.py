import numpy as np


def wide_kdeBW ( size, alpha=3. ):
    bw = alpha*size**(-1./5.)  
    return bw  

def gaussian ( x, A, m, s):
    if A == 'normalize':
        A = np.sqrt(2.*np.pi * s**2)**-1    
    return A * np.exp ( -(x-m)**2 / (2.*s**2) )

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