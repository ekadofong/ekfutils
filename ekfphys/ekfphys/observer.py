import numpy as np
import extinction

def gecorrection(wave, Av, Rv=3.1, unit='AA', return_magcorr=False):
    """Calculate the Galactic extinction correction for a given wavelength and Av.

    Parameters:
    -----------
    wave : array_like
        Wavelength in units of `unit`. Must be a 1D array.
    Av : float
        V-band extinction in magnitudes.
    Rv : float, optional (default=3.1)
        Total-to-selective extinction ratio.
    unit : {'aa', 'micron', 'nm', 'um'}, optional (default='AA')
        Unit of the wavelength.
    return_magcorr : bool, optional (default=False)
        If True, return the extinction correction in magnitudes instead of flux ratio.

    Returns:
    --------
    corr : array_like
        Extinction correction factor for the given wavelength(s). If `return_magcorr` is True,
        then the result is the extinction correction in magnitudes.
    """
    Alambda = extinction.ccm89(wave, Av, Rv, unit=unit.lower())
    if return_magcorr:
        return Alambda
    else:
        corr = 10.**(0.4*Alambda)
        return corr
    
def balmerdecrement_to_av ( balmerdecrement, intrinsicratio=2.86, RV=4.05 ):
    balmerintrinsic=2.86
    phi = 2.5*np.log10(intrinsicratio/balmerdecrement)
    dk = np.subtract(*extinction.calzetti00 ( np.array([6563.,4862.]), 1., 1. ) ) 
    AV = RV*phi/dk
    return AV    

def extinction_correction ( wavelength, av, u_av=None, RV=4.05, curve=None ):
    if curve is None:
        curve = extinction.calzetti00
    k0 = curve ( np.array([wavelength]), av, RV )
    alambda = av * (k0/RV + 1.) 
    if u_av is not None:
        u_alambda = u_av * ( k0/RV + 1. )
        u_corr = abs(0.4*np.log(10.)*corr) * u_alambda
    else:
        #u_alambda = None
        u_corr = None
    corr = float(10.**(alambda/2.5))
    
    return corr, u_corr
    

def photometric_kcorrection ( gr, redshift ):
    '''
    Calculate the K-correction for SDSS r-band based on a given g-r color and redshift.

    This function computes the K-correction using the Chilingarian et al. (2012) Equation 1
    and Table A3. The K-correction is a correction factor applied to photometric data
    to account for the shift in observed wavelengths due to redshift.

    Parameters:
    gr (float): The g-r color of the object.
    redshift (float): The redshift of the object.

    Returns:
    float: The calculated K-correction for the SDSS r-band.

    References:
    Chilingarian, I. V., Melchior, A.-L., & Zolotukhin, I. 2012, AJ, 144, 47
    '''
    arr = np.array([[0., 0., 0., 0.,],
                    [-1.61294, 3.81378, -3.56114, 2.47133],
                    [9.13285,9.85141,-5.1432,-7.02213],
                    [-81.8341,-30.3631,38.5052,0.,],
                    [250.732,-25.0159,0.,0.,],
                    [-215.377,0.,0.,0.]
                    ])
    kcorrection = 0.
    for y_index in np.arange(4):
        for z_index in np.arange(5):
            a_xy = arr[z_index, y_index]
            val = a_xy * redshift**z_index * gr**y_index
            kcorrection += val
    return kcorrection
