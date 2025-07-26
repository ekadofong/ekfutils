import numpy as np
import extinction
from astropy import units as u
from astropy.io import fits
from astropy import constants as co
from astropy.modeling.physical_models import BlackBody
import healpy as hp
from .calc_kcor import calc_kcor

def gecorrection(wave, AvorEBV, Rv=3.1, unit='AA', etype='AV', return_magcorr=False):
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
    if etype == 'AV':
        Av = AvorEBV
    elif etype == 'EBV':
        Av = Rv * AvorEBV
        
    if isinstance(wave, float) or isinstance(wave, int):
        wave = np.array([wave], dtype=float)        
    
    Alambda = extinction.ccm89(wave, Av, Rv, unit=unit.lower())
    if return_magcorr:
        return Alambda
    else:
        corr = 10.**(0.4*Alambda)
        return corr
    
def balmerdecrement_to_av ( balmerdecrement, intrinsicratio=2.86, RV=4.05 ):
    phi = 2.5*np.log10(intrinsicratio/balmerdecrement)
    dk = np.subtract(*extinction.calzetti00 ( np.array([6563.,4862.]), 1., 1. ) ) 
    AV = RV*phi/dk
    return AV    

def alambda_to_av ( alambda, wv_eff, rv=4.05 ):
    kl = extinction.calzetti00 ( np.array([wv_eff]), 1., 1. ) - 1. 
    av = alambda*(1. + kl/rv)**-1
    return av

def calzetti_flux_correction_with_uncertainty(wavelength, Av, u_Av, R_V=4.05, return_magcorr=False):
    """
    Compute the flux correction factor and its uncertainty using
    the Calzetti et al. (1994) attenuation law.

    Parameters
    ----------
    wavelength : array_like
        Wavelengths in Angstroms.
    A_V : float
        Attenuation in magnitudes at V-band (5500 Å).
    u_A_V : float
        Uncertainty in A(V).
    R_V : float, optional
        Total-to-selective extinction ratio (default is 4.05).

    Returns
    -------
    flux_correction : ndarray
        Multiplicative factor to convert observed to intrinsic flux:
        F_intrinsic = F_observed * flux_correction
    uncertainty : ndarray
        1σ uncertainty in the flux correction factor.
    """
    wavelength = np.array(wavelength, dtype=float)
    lambda_um = wavelength / 1e4  # Convert Å to microns

    k_lambda = np.zeros_like(lambda_um)
    valid = (lambda_um >= 0.12) & (lambda_um <= 2.2)
    x = lambda_um[valid]

    # Calzetti law
    k_valid = np.where(
        x <= 0.63,
        2.659 * (-2.156 + 1.509 / x - 0.198 / x**2 + 0.011 / x**3) + R_V,
        2.659 * (-1.857 + 1.040 / x) + R_V
    )
    k_lambda[valid] = k_valid
    k_lambda[~valid] = np.nan

    A_lambda = k_lambda * (Av / R_V)
    u_A_lambda = k_lambda * (u_Av / R_V)
    if return_magcorr:
        return A_lambda, u_A_lambda
    
    flux_corr = 10**(0.4 * A_lambda)

    # Error propagation
    ln10 = np.log(10)
    u_flux_corr = flux_corr * ln10 * 0.4 * (k_lambda / R_V) * u_Av

    return flux_corr, u_flux_corr

def calzetti_curve(wavelength, R_V=4.05):
    """
    Compute the flux correction factor and its uncertainty using
    the Calzetti et al. (1994) attenuation law.

    Parameters
    ----------
    wavelength : array_like
        Wavelengths in Angstroms.
    A_V : float
        Attenuation in magnitudes at V-band (5500 Å).
    u_A_V : float
        Uncertainty in A(V).
    R_V : float, optional
        Total-to-selective extinction ratio (default is 4.05).

    Returns
    -------
    flux_correction : ndarray
        Multiplicative factor to convert observed to intrinsic flux:
        F_intrinsic = F_observed * flux_correction
    uncertainty : ndarray
        1σ uncertainty in the flux correction factor.
    """
    wavelength = np.array(wavelength, dtype=float)
    lambda_um = wavelength / 1e4  # Convert Å to microns

    k_lambda = np.zeros_like(lambda_um)
    valid = (lambda_um >= 0.12) & (lambda_um <= 2.2)
    x = lambda_um[valid]

    # Calzetti law
    k_valid = np.where(
        x <= 0.63,
        2.659 * (-2.156 + 1.509 / x - 0.198 / x**2 + 0.011 / x**3) + R_V,
        2.659 * (-1.857 + 1.040 / x) + R_V
    )
    k_lambda[valid] = k_valid
    k_lambda[~valid] = np.nan

    return k_lambda / R_V



def extinction_correction ( wavelength, av, u_av=None, RV=4.05, curve=None, return_magcorr=False ):
    if curve is None:
        curve = extinction.calzetti00
    if not hasattr(wavelength, '__len__'):
        wavelength = np.array([wavelength])

    alambda = curve(wavelength, av, RV)
    corr = 10.**(alambda/2.5)
    if u_av is not None:
        u_alambda = curve(wavelength, u_av, RV)
        u_corr = abs(0.4*np.log(10.)*corr) * u_alambda
    else:
        #u_alambda = None
        u_corr = None
    if return_magcorr:
        if u_av is None:
            return alambda
        else:
            return alambda, u_alambda
    
    return corr, u_corr
    

def photometric_kcorrection ( color, redshift, correction_band='sdss-r' ):
    '''
    Calculate the K-correction for SDSS r-band based on a given g-r color and redshift.

    This function computes the K-correction using the Chilingarian et al. (2012) Equation 1
    and Table A3. The K-correction is a correction factor applied to photometric data
    to account for the shift in observed wavelengths due to redshift.

    Parameters:
    color (float): The g-r color of the object.
    redshift (float): The redshift of the object.

    Returns:
    float: The calculated K-correction for the SDSS r-band.

    References:
    Chilingarian, I. V., Melchior, A.-L., & Zolotukhin, I. 2012, AJ, 144, 47
    '''
    if correction_band=='sdss-r':
        arr = np.array([[0., 0., 0., 0.,],
                        [-1.61294, 3.81378, -3.56114, 2.47133],
                        [9.13285,9.85141,-5.1432,-7.02213],
                        [-81.8341,-30.3631,38.5052,0.,],
                        [250.732,-25.0159,0.,0.,],
                        [-215.377,0.,0.,0.]
                        ])
    else:
        raise ValueError(f"{correction_band} not recognized!")
    kcorrection = 0.
    for y_index in np.arange(4):
        for z_index in np.arange(5):
            a_xy = arr[z_index, y_index]
            val = a_xy * redshift**z_index * color**y_index
            kcorrection += val
    return kcorrection

def stellar_photometry ( temperature, filter_file  ):
    if not hasattr(temperature, 'unit'):
        temperature = temperature * u.K
            
    transmission = np.genfromtxt ( filter_file )
    wl = transmission[:,0]*u.AA
    freq = (co.c/wl).to(u.Hz)
    # nrml = np.trapz ( ytrans / nu_trans, nu_trans )
    normalization = np.trapz ( transmission[:,1][::-1] / freq[::-1], freq[::-1] )
    
    if temperature.ndim > 0:
        filter_flux = np.zeros(len(temperature), dtype=float) * u.erg/u.s/u.cm**2/u.Hz
        for idx in range(len(temperature)):
            bb = BlackBody ( temperature[idx] )
            in_filter_flux = np.trapz ( (bb(wl)*transmission[:,1])[::-1]/freq[::-1], freq[::-1] ) * np.pi * u.sr
            filter_flux[idx] = in_filter_flux/normalization
    else:
        bb = BlackBody ( temperature )
        filter_flux = np.trapz ( (bb(wl)*transmission[:,1])[::-1]/freq[::-1], freq[::-1] ) * np.pi * u.sr
        filter_flux /= normalization
    return filter_flux

def flambda_to_fnu ( wv, flux ):
    '''
    nuF_nu = lF_l
    F_nu = l/nu F_l
         = l^2/c F_l
    '''
    if not hasattr ( wv, 'unit' ) or not hasattr ( flux, 'unit' ):
        raise TypeError ("Must supply astropy Quantities!")
    eunit = flux.unit * wv.unit
    return (wv**2/co.c * flux).to(eunit/u.Hz)

def fnu_to_flambda ( wvorfreq, flux ):
    '''
    lam F_lam = nu F_nu 
    F_lam = nu / lam F_nu 
          = nu^2 / c F_nu
    '''    
    if not hasattr ( wvorfreq, 'unit' ) or not hasattr ( flux, 'unit' ):
        raise TypeError ("Must supply astropy Quantities!") 
    if wvorfreq.unit.is_equivalent(u.m):
        freq = co.c/wvorfreq
    else:
        freq = wvorfreq
            
    eunit = flux.unit * freq.unit
    return (freq**2 / co.c * flux).to(eunit/u.AA)

def photometry_from_spectrum ( wv, flux, wv_filter=None, trans_filter=None, filter_file=None, trans_type='photon'  ):
    if filter_file is not None:
        transmission = np.genfromtxt ( filter_file )
        wv_filter = transmission[:,0]*u.AA
        trans_filter = transmission[:,1]  
            
    if flux.unit.is_equivalent(u.erg/u.s/u.cm**2/u.AA):
        flux = flambda_to_fnu(wv, flux)
        #pass
    elif flux.unit.is_equivalent(u.erg/u.s/u.cm**2/u.Hz):
        #flux = fnu_to_flambda(wv, flux)
        pass
    else:
        raise ValueError ('Flux units not dimensionally correct!')
    
    # \\ filter transmission needs to be monotonically increasing in wavelength
    wsort = np.argsort(wv_filter)
    wv_filter = wv_filter[wsort]
    trans_filter = trans_filter[wsort]
        
    wmask = (wv>=wv_filter.min())&(wv<=wv_filter.max())
    wv = wv[wmask]
    flux = flux[wmask]
    freq = (co.c/wv).to(u.Hz)
    
    filter_interpolated = np.interp(wv.value, wv_filter.to(wv.unit).value, trans_filter)

    #filter_interpolated = filter_interpolated/freq**2 # account for nu T(nu) = lambda T(lambda)
    
    if trans_type == 'photon':
        
        
        band_flux = np.trapz ( flux*filter_interpolated/freq, freq )
        normalization = np.trapz ( filter_interpolated/freq, freq )
        #band_flux = np.trapz(flux*filter_interpolated*wv, wv)
        #normalization = np.trapz(filter_interpolated*wv, wv)
    elif trans_type == 'energy':
        band_flux = np.trapz ( flux*filter_interpolated, freq )
        normalization = np.trapz ( filter_interpolated, freq )
        #band_flux = np.trapz(flux*filter_interpolated, wv)
        #normalization = np.trapz(filter_interpolated, wv)        
    return band_flux/normalization  


class SolarReference ():
    def __init__ ( self, spectrum = None):
        if spectrum is None:
            self.spectrum = fits.open('/Users/kadofong/work/projects/literature_ref/solar/sun_composite.fits')[1]
            
    def band_luminosity ( self, filter_file ):
        fnu = flambda_to_fnu(self.spectrum.data['WAVE']*u.AA, self.spectrum.data['FLUX']*u.erg/u.s/u.cm**2/u.AA)        
        truesun = photometry_from_spectrum(self.spectrum.data['WAVE']*u.AA, fnu, filter_file=filter_file )
        truesun *= 4.*np.pi*(u.AU).to(u.cm)**2
        return truesun


def estimate_sky_footprint(ra_deg, dec_deg, nside=128, plot=True):
    """
    NOT RELIABLE

    Parameters
    ----------
    ra_deg : array-like
        Right Ascension in degrees.
    dec_deg : array-like
        Declination in degrees.
    nside : int
        HEALPix resolution parameter. Higher values give better spatial resolution.
    plot : bool
        Whether to plot the resulting footprint.

    Returns
    -------
    footprint_mask : np.ndarray
        Boolean array of shape (12 * nside^2,) with True for observed pixels.
    """
    # Convert RA, Dec to theta, phi (in radians)
    theta = np.radians(90.0 - dec_deg)  # colatitude
    phi = np.radians(ra_deg)

    # Get HEALPix pixel indices
    pix = hp.ang2pix(nside, theta, phi, nest=False)

    # Create mask
    npix = hp.nside2npix(nside)
    footprint_mask = np.zeros(npix, dtype=bool)
    footprint_mask[np.unique(pix)] = True

    if plot:
        hp.mollview(footprint_mask.astype(float), title="Estimated Survey Footprint", cmap='cividis')

    return footprint_mask
