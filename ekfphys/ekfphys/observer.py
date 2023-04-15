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
