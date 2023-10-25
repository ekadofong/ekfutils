import numpy as np
from scipy import integrate 
from astropy import units as u
from astropy import constants as co
from astropy.modeling.physical_models import BlackBody

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

def flux21_to_mhi ( flux, z=0., dlum=None, pixel_area=None):
    '''
    21cm flux to HI mass; equation 8.20 & 8.21 from Bruce's book
    '''
    if dlum is None:
        dlum = cosmo.luminosity_distance(z)
    if pixel_area is None:
        pixel_area = 1.*u.pix
        
    prefactor = 2.343e5 * (1.+z) * u.M_sun
    dfactor = (dlum/u.Mpc)**2
    fluxfactor = flux / (u.Jy * u.km/u.s)
    
    himass = prefactor * dfactor * fluxfactor
    hisd = himass / pixel_area
    return hisd

# from Eker + 2018
solar_sb = co.sigma_sb.to(u.L_sun/u.K**4/u.R_sun**2).value

coeffs = [[2.028, -0.976],
          [4.572,-0.102],
          [5.743,-0.007],
          [4.329,0.010],
          [3.967,0.093],
          [2.865,1.105]]
breakpts = [0.,0.45,0.72,1.05,2.4,7.,np.inf]
def mass_luminosity ( mass ):
    '''
    Mass-Luminosity from https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.5491E/abstract
    '''
    if isinstance(mass, float):
        mass = np.array(mass)

    luminosity = np.zeros_like(mass)
    for idx in range(len(coeffs)):
        mask = (mass>breakpts[idx])&(mass<=breakpts[idx+1])
        cc = coeffs[idx]
        luminosity[mask] = cc[0]*np.log10(mass[mask]) + cc[1]
    return 10.**luminosity

def mass_temperature_radius ( mass ):
    '''
    MTR relation from https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.5491E/abstract
    '''
    if isinstance(mass, float):
        mass = np.array(mass)

    luminosity = mass_luminosity(mass)
    radius = np.zeros_like(mass)
    temperature = np.zeros_like(mass)

    low_mass = mass < 1.5
    radius[low_mass] = 0.438 * mass[low_mass]**2 + 0.479*mass[low_mass] + 0.075
    temperature[low_mass] = (luminosity[low_mass] / (4.*np.pi * radius[low_mass]**2 * solar_sb))**0.25

    temperature[~low_mass] = 10.**(-0.17 * np.log10(mass[~low_mass])**2 + 0.888*np.log10(mass[~low_mass]) + 3.671)
    radius[~low_mass] = np.sqrt(luminosity[~low_mass]/(4.*np.pi*solar_sb*temperature[~low_mass]**4))
    
    return radius*u.R_sun, temperature*u.K, luminosity*u.L_sun

def ionizing_flux ( temperature ):
    '''
    Assuming a blackbody spectrum, the rate of ionizing photons emitted by a 
    star of a given temperature.
    '''
    if not hasattr(temperature, 'unit'):
        temperature = temperature * u.K
    
    ionizing_flux = np.zeros(len(temperature), dtype=float)
    for idx in range(len(temperature)):
        bb = BlackBody ( temperature[idx] )
        ionizing_energy = 13.6*u.eV
        # E = hnu
        # E = h c / lambda
        # lambda = hc / E
        ionizing_wavelength = (co.h * co.c / ionizing_energy ).to(u.AA)
        ionizing_frequency = (ionizing_energy / co.h).to(u.Hz)
        max_frequency = (co.c / (10.*u.AA)).to(u.Hz)
        
        wv = np.arange(10., ionizing_wavelength.value, 1.) * u.AA
        nu = np.linspace(ionizing_frequency, max_frequency, 1000) 
        
        energy_per_photon = (nu * co.h).to(u.erg) / u.photon
        ionizing_intensity = np.trapz( bb(nu) / energy_per_photon, nu ) 
        iflux = ionizing_intensity * np.pi * u.sr # why is it x pi sr instead of 4pi?
        ionizing_flux[idx] = iflux.to(u.photon/u.s/u.cm**2).value
    return ionizing_flux * u.photon/u.s/u.cm**2