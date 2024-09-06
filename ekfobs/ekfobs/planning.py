import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from astropy import units as u
from astropy import coordinates
import progressbar
from astropy.io import fits
from astropy import wcs
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from astroquery.utils.tap.core import TapPlus

from ekfplot import plot as ek


# Set the logging level for astroquery.utils.tap.core
logger = logging.getLogger('astroquery.utils.tap.core')
logger.setLevel(logging.CRITICAL)
logger.propagate = False

def _find_bo_star_panstarrs (skycoord, maglim=16., return_all=False):
    ps1_match = Catalogs.query_object(skycoord.to_string('hmsdms'), catalog="Panstarrs", radius=5.*u.arcmin, table='mean').to_pandas()

    is_star = (ps1_match['iMeanPSFMag'] - ps1_match['iMeanKronMag'])<=0.01 
    # \\ star-gal sep based off of https://outerspace.stsci.edu/display/PANSTARRS/How+to+separate+stars+and+galaxies
    is_bright = ps1_match['rMeanPSFMag'] < maglim
    is_use = ps1_match['qualityFlag'] & 4 != 0

    offset_stars = ps1_match.loc[is_star&is_bright&is_use,['raMean','decMean','distance','rMeanPSFMag','objInfoFlag','qualityFlag']].sort_values('distance')    
    if return_all:
        return offset_stars
    else:
        if len(offset_stars) == 0:
            raise ValueError ("No suitable offset star found within 5 arcmin!")
        
        ostar = offset_stars.iloc[0]
        return ostar  
    
def _find_bo_star_gaia (skycoord, maglim=16., nreturn=1, pmlim=None, radius=None ):
    if pmlim is None:
        pmlim = 20.*u.mas/u.yr
    if radius is None:
        radius=5.*u.arcmin
    gaia_query = Gaia.cone_search_async(coordinate=skycoord, radius=radius, verbose=False)
    gaia_objs = gaia_query.get_results()

    low_pm = (gaia_objs['pmra']<pmlim)&(gaia_objs['pmdec']<pmlim)
    is_bright = gaia_objs['phot_g_mean_mag']<maglim
    is_not_same_object = gaia_objs['dist'] > (2.*u.arcsec).to(u.deg).value
    offset_stars = gaia_objs[low_pm&is_bright&is_not_same_object].to_pandas ().sort_values('dist')
    offset_stars = offset_stars.rename({'dist':'distance'}, axis=1)    

    if len(offset_stars) == 0:
        raise ValueError (f"No suitable offset star found within {radius}!")
    
    if nreturn > 1:
        ostar = offset_stars.iloc[:nreturn]
    else:
        ostar = offset_stars.iloc[0]
    return ostar      

def _find_bo_star (*args, catalog='gaia', **kwargs):
    if catalog == 'gaia':
        return _find_bo_star_gaia (*args, **kwargs)
    elif catalog == 'panstarrs':
        return _find_bo_star_panstarrs (*args, **kwargs)

def find_blindoffset_stars (coords, maglim=16., catalog='gaia',):
    offset_stars = []
    
    # \\ set up progress bar
    widgets = [
        'Progress: ', progressbar.Percentage(),
        ' ', progressbar.Bar(marker='=', left='[', right=']'),
        ' ', progressbar.ETA(),       # Estimated Time of Arrival (time remaining)
        ' ', progressbar.Timer(),     # Time elapsed
    ]    
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(coords))
    pbar.start()
    for idx,obj in enumerate(coords):
        if hasattr(obj, 'ra'):
            skycoord = obj
        else:
            skycoord = coordinates.SkyCoord(obj[0], obj[1], units=('deg','deg'))
        
        ostar = _find_bo_star ( skycoord, maglim, catalog=catalog )

        offset_stars.append(ostar)
        pbar.update(idx)
    ovl_ostars = pd.concat(offset_stars, axis=1).T
    ovl_ostars['distance_arcsec'] = ovl_ostars['distance'] * u.deg.to(u.arcsec)
    return ovl_ostars
        
        
def make_findingchart (source_info, cutout_filename, offset_star, name=None):
    if name is None:
        name = source_info.name
    cutout = fits.open(f'./cutouts/{name}.fits')[0]
    cutout_wcs = wcs.WCS(cutout.header)
    
    #row = sample.iloc[ridx]
    name = row.name
    
    fig = plt.figure(figsize=(13, 4))
    ax = fig.add_subplot(1,3,1)
    
    alt_l_index = np.arange(len(alt_l), dtype=int)[is_observable][ridx]
    ax.plot(obsframe.obstime.datetime, alt_l[alt_l_index].alt, 'o')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    #ax.set_xtic(rotation=45)
    tw = ax.twinx()
    tw.plot(obsframe.obstime.datetime, alt_l[alt_l_index].secz, 'o', color='C1')
    tw.set_ylim(1.,3.)
    tw.set_ylabel('airmass', color='C1', rotation=270, labelpad=20)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_xlabel('Time [UTC]')
    ax.set_ylabel('altitude', color='C0')
    
    
    ek.text(
        0.025,
        0.975,
        r'%s' % row.name,
        ax=ax,
        fontsize=10,
        bordercolor='w',
        borderwidth=2
    )
    
    ax.spines.left.set_color("C0")

    
    ax=fig.add_axes(132, projection=cutout_wcs.celestial)
    ek.imshow(cutout.data[0],origin='lower', cmap='Greys', q=0.0025, ax=ax)
    
    msize = 20
    ek.celestial_plot(source_info['RA'], source_info['DEC'], ax, marker='o', markeredgecolor='lime', markerfacecolor='None', markersize=msize)
    ek.celestial_plot(offset_star['raMean'], offset_star['decMean'], ax, marker='o', markeredgecolor='cyan', markerfacecolor='None', markersize=msize)
    
    ek.text(0.025, 0.975, 'Offset Star', color='cyan', ax=ax, bordercolor='w', borderwidth=3)
    ek.text(0.025, 0.905, 'Science Target', color='lime', ax=ax, bordercolor='w', borderwidth=3)


    ras = [source_info['RA'], offset_star['raMean']]
    decs = [source_info['DEC'],offset_star['decMean']]
    ek.celestial_plot( ras, decs, ax, color='r', lw=0.2, ls=':' )
    ek.text(
        np.mean(ras),
        np.mean(decs),
        r"%i''" % offset_star.distance_arcsec,
        color='tab:red',
        coord_type='absolute',
        bordercolor='w',
        borderwidth=2,
        transform=ax.get_transform('fk5'),
        ax=ax
    )
        
    
    ax.grid(color='white', ls=':')
    ax.set_xlabel('RA [J2000]')
    ax.set_ylabel('DEC [J2000]')
    

    imax = fig.add_axes(133)
    thumbnail = mpimg.imread(f'./thumbnails/{name}.jpg')
    imax.imshow(thumbnail)
    imax.axis('off')
    fha = fha_from_sfr(row)
    fha_oom = np.floor(np.log10(fha))

    ek.text(
        0.025,
        0.975,
        r'''$m_r = %.1f$
$m_{N708}=%.1f$
$F_{\rm H\alpha}^{\rm est} = %.2f\times 10^{%i} \frac{\rm erg}{\rm s\ cm^{2}}$'''%(
            utils.flux2mag(row['r_cModelFlux_Merian']),
            utils.flux2mag(row['N708_cModelFlux_Merian']),
            fha / 10.**fha_oom,
            fha_oom
        ),
        ax=imax,
        color='w',
        fontsize=12
    )
    plt.tight_layout ()
    
    pos = imax.get_position ()
    new_pos = [pos.x0 - 0.1, pos.y0, pos.width, pos.height]
    imax.set_position (new_pos)    