import numpy as np
from scipy import integrate 

def _kroupa_imf ( m ):
    mmin = 0.03
    mmax = 120.
    low_m = 0.08
    high_m = 0.5
    
    low_index = -0.3
    mid_index = -1.3
    high_index = -2.3
    
    if isinstance( m, (float,int)):
        m = np.array([m])
        
    indices = np.ones_like(m)
    indices[m<low_m] = low_index
    indices[(m>=low_m)&(m<high_m)] = mid_index
    indices[m>=high_m] = high_index
    
    # \\ scaling so that the broken p-laws match up:
    # \\ low_m**-0.3 = A * low_m**-1.3
    # \\ low_m ** (low_index - mid_index) = A 
    # \\ A*high_m**mid_index = B*high_m**high_index
    # \\ A * high_m ** (mid_index - high_index) = B    
    mid_scaling = low_m**(low_index-mid_index)
    high_scaling = mid_scaling * high_m ** ( mid_index - high_index )
    scaling = np.ones_like(m)
    scaling[(m>=low_m)&(m<high_m)] = mid_scaling
    scaling[m>=high_m] = high_scaling
    return scaling * m**indices

norm = integrate.quad( _kroupa_imf, 0.08, 120. )[0]
def kroupa_imf (m):    
    return _kroupa_imf(m)/norm

def rough_stellar_lifetime ( m, mfid=1., tfid=1e10 ):
    '''
    Assuming tMS(1 Msun) = 10^10 yr,
    and
    tMS(M) propto M^-2.5
    '''
    return tfid * (m/mfid)**-2.5

def rough_mdeath ( lifetime, mfid=1., tfid=1e10 ):
    '''
    tMS(M)/tMS(Mfid) = (M/Mfid)^-2.5
    Mfid * (tMS(M)/tMS(Mfid))^-0.4 = M
    '''
    return mfid * (lifetime/tfid)**-0.4