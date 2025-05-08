
import numpy as np
from scipy import integrate

from ekfstats import sampling

def integrate_up_massfunction ( mf, logm_grid, integration_type='quad', mf_scale='log' ):
    """
    Compute the cumulative number density above a given mass threshold by integrating a mass function.

    Parameters
    ----------
    mf : callable
        The mass function, which takes log10(M) as input. It should return either the log10 of the 
        differential number density (if mf_scale='log') or the linear value (if mf_scale='linear').
        The units of the output should be consistent with dN/dlog10M/dV.
    
    logm_grid : array-like
        Array of log10 mass values over which to evaluate and integrate the mass function.
    
    integration_type : {'trapz', 'quad'}, optional
        Integration method. 'trapz' uses NumPy's trapezoidal rule, while 'quad' uses SciPy's adaptive
        quadrature (`scipy.integrate.quad`).
    
    mf_scale : {'log', 'linear'}, optional
        Scale of the mass function output. Use 'log' if mf returns log10(dN/dlog10M/dV),
        and 'linear' if it returns dN/dlog10M/dV.

    Returns
    -------
    logm_values : ndarray
        The log10 mass values at which the cumulative number densities are computed.

    cumulative_density : ndarray
        Cumulative number density (i.e., number density above each mass in logm_values),
        integrated up from the high-mass end.
    """    
    # \\ mf_scale refers to whether the reported mass function is dN/dlog10M/dV or log10(dN/dlog10M/dV), 
    # \\ _NOT_ whether it is dn/dM or dn/dlog10M
    if integration_type=='trapz':
        if mf_scale == 'log':
            densities = 10.**mf(logm_grid)
        else:
            densities = mf(logm_grid)
        total_density = np.trapz(densities, logm_grid)
        result = total_density - integrate.cumulative_trapezoid(densities, logm_grid)
        return sampling.midpts(logm_grid), result
    elif integration_type=='quad':
        result = np.zeros_like(logm_grid)
        if mf_scale == 'log':
            fn = lambda m: 10.**mf(m)
        else:
            fn = mf
        for midx in range(len(logm_grid)):        
            result[midx] = integrate.quad(fn, logm_grid[midx], 17)[0]
        return logm_grid, result


def abundance_match ( smf, hmf, logmstar_min=5., logmstar_max=10., dm=0.1, smf_scale='log', hmf_scale='linear' ):
    """
    Perform abundance matching between a stellar mass function and a halo mass function.

    Parameters
    ----------
    smf : callable
        The stellar mass function, as a function of log10(stellar mass). Should return either log10(dN/dlog10M/dV)
        or dN/dlog10M/dV depending on `smf_scale`.

    hmf : callable
        The halo mass function, as a function of log10(halo mass). Should return either log10(dN/dlog10M/dV)
        or dN/dlog10M/dV depending on `hmf_scale`.

    logmstar_min : float, optional
        Minimum log10 stellar mass for the output grid.

    logmstar_max : float, optional
        Maximum log10 stellar mass for the output grid.

    dm : float, optional
        Step size in log10 mass for the integration and interpolation grids.

    smf_scale : {'log', 'linear'}, optional
        Scale of the SMF output. Use 'log' if it returns log10(dN/dlog10M/dV), or 'linear' otherwise.

    hmf_scale : {'log', 'linear'}, optional
        Scale of the HMF output. Use 'log' if it returns log10(dN/dlog10M/dV), or 'linear' otherwise.

    Returns
    -------
    logmstar : ndarray
        Array of log10 stellar mass values for which the halo masses are inferred.

    logmhalo : ndarray
        Array of log10 halo mass values assigned to each stellar mass, based on abundance matching.
    """    
    integration_grid = np.arange(5., 16.+dm/2., dm)
    logm_grid = np.arange(logmstar_min, logmstar_max+dm/2., dm)
    _,iup_smf = integrate_up_massfunction( smf, integration_grid, mf_scale=smf_scale )
    _,iup_hmf = integrate_up_massfunction( hmf, integration_grid, mf_scale=hmf_scale)

    halo_masses_at_stellarmass = np.interp(iup_smf[::-1], iup_hmf[::-1], integration_grid[::-1])[::-1]
    mask = (integration_grid>logmstar_min)&(integration_grid<logmstar_max)
    return integration_grid[mask], halo_masses_at_stellarmass[mask]