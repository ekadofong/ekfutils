
import numpy as np
from scipy import integrate

from ekfstats import sampling

def integrate_up_massfunction ( mf, logm_grid, integration_type='quad', mf_scale='log' ):
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
    integration_grid = np.arange(5., 16.+dm/2., dm)
    logm_grid = np.arange(logmstar_min, logmstar_max+dm/2., dm)
    _,iup_smf = integrate_up_massfunction( smf, integration_grid, mf_scale=smf_scale )
    _,iup_hmf = integrate_up_massfunction( hmf, integration_grid, mf_scale=hmf_scale)

    halo_masses_at_stellarmass = np.interp(iup_smf[::-1], iup_hmf[::-1], integration_grid[::-1])[::-1]
    mask = (integration_grid>logmstar_min)&(integration_grid<logmstar_max)
    return integration_grid[mask], halo_masses_at_stellarmass[mask]