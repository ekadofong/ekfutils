import time
import os
import glob
import logging
import urllib
import requests
from xml.etree import ElementTree as ET
import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy import coordinates
from astropy import units as u
from astropy import table
from astropy.nddata import Cutout2D
import astroquery
from astroquery.mast import Observations

def get_SandFAV ( ra, dec, region_size = 2., Rv = 3.1, verbose=False):
    '''
    Query the IRSA dust database to get Av from SandF (2011)
    
    args:
        ra (float): RA (J2000) in degrees
        dec (float): DEC (J2000) in degrees
        region_size (float): region size to average over in degrees
        Rv (float, default=3.1): AV/E(B-V) = Rv
    '''
    
    
    #\\ fetch XML output
    url=f'https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?regSize={region_size}&locstr={ra}+{dec}'
    if verbose:
        print(f"HTTP GET: {url}")
    output = urllib.request.urlopen(url).read()
        
    #\\ parse XML
    root = ET.fromstring(output)
    ebv_element = root.find('.//meanValueSandF')
    ebv = ebv_element.text.strip(' \n') # \\ get rid of formatting
    if verbose:
        print(f"E(B-V) = {ebv}")
    ebv = float(ebv.split()[0]) # \\ formatted as e.g. '0.03  (mag)'
    Av = ebv * Rv
    return Av



class DustEngine ( object ):
    def __init__ (self):
        self.coord_cache = []
        self.av_cache = []

    def _concat ( self, xl):
        '''
        Convenience function since coordinates.concatenate does not handle
        the len(X) == 1 case gracefully.
        '''
        if len(xl) == 0:
            return np.array([])
        elif len(xl) == 1:
            if isinstance(xl[0], coordinates.SkyCoord):
                return xl[0]
            else:
                return np.array(xl)
        else:
            if isinstance(xl[0], coordinates.SkyCoord):
                return coordinates.concatenate(xl)
            else:
                return np.concatenate([xl])

    def get_SandFAV ( self, ra, dec, unit='deg', match_radius=None, verbose=0, **kwargs ):
        """
        Retrieve the A_V from Schlafly and Finkbeiner (2011) value for given right ascension and declination.

        Args:
            ra : float - Right ascension of the target location.
            dec : float - Declination of the target location.
            unit : str - Unit of the coordinates (default is 'deg').
            match_radius : Optional[float] - Radius within which to match coordinates from the cache (default is None).
            verbose : int - Verbosity level for logging information (default is 0).
            **kwargs : dict - Additional arguments that may include 'region_size' for match radius.

        Return:
            av : float - Averaged A_V from Schlafly and Finkbeiner (2011) value for the given coordinates.

        Raise:
            None
        """
            
        if match_radius is None:
            if 'region_size' in kwargs.keys():
                match_radius = kwargs['region_size'] * u.deg
            else:
                match_radius = 2. * u.deg
        coord = coordinates.SkyCoord ( ra, dec, unit=unit )
        
        if len(self.coord_cache) > 0:
            cache_separation = coord.separation ( self._concat(self.coord_cache) )
            close_matches = cache_separation < match_radius
            no_matches = close_matches.sum() == 0
        else:
            no_matches = True
        if no_matches:
            if verbose > 0:
                print('[DustEngine.get_SandFAV] No cached matches, querying IPAC...')
            av = get_SandFAV ( ra, dec, **kwargs )
            self.coord_cache.append(coord)
            self.av_cache.append(av)            
        else:
            if verbose > 0:
                nobs = int(close_matches.sum())
                print(f'[DustEngine.getSandFAV] Averaging {nobs} GE matches...')
            av = np.mean(self._concat(self.av_cache)[close_matches])

        return av
        


def identify_galexcoadds ( ra, dec, radius=None, verbose=True ):
    """
    Get nearby observations from MAST for a given position.

    Parameters:
        ra (float): Right ascension of the position.
        dec (float): Declination of the position.
        radius (astropy.units.quantity.Quantity, optional): Radius around the position (default is 10 arcseconds).

    Returns:
        tuple: A tuple containing the table of nearby observations and the names of the FUV and NUV tiles.

    """    
    if radius is None:
        radius = 10. * u.arcsec
    start = time.time ()
    # \\ get all nearby MAST regions  
    obs_table = Observations.query_region ( f'{ra} {dec}', radius=radius )    
    if verbose:
        print(f'Queried GALEX observations in {time.time() - start:.2f} seconds.')
        start = time.time ()        
    
    galex_table = obs_table[obs_table['obs_collection'] == "GALEX"]
    
    # \\ only include COADDS that are within 1 GALEX FOV of queried coordinates,
    # \\ since the coadds are 0 padded outside the circular field of view.
    galex_fov = 0.6 * u.deg # \\ FOV radius, in degrees. Actual is about 0.625? But coadd seems a little smaller
    coadd_centers = coordinates.SkyCoord ( galex_table['s_ra'], galex_table['s_dec'], unit='deg' )
    target = coordinates.SkyCoord ( ra, dec, unit='deg')
    center2target_separation = target.separation(coadd_centers)
    is_within_coadd = center2target_separation < galex_fov
    galex_table = galex_table[is_within_coadd]
    
    fuv = galex_table[galex_table['filters']=='FUV']
    nuv = galex_table[galex_table['filters']=='NUV']
    
    # \\ choose highest exptime NUV & FUV tiles
    if len(fuv) > 0:
        fc = fuv[[np.argmax(fuv['t_exptime'])]] 
        fuv_name = fc['target_name'][0]
    else:
        fc = None
        fuv_name = None
    if len(nuv) > 0:
        nc = nuv[[np.argmax(nuv['t_exptime'])]]
        nuv_name = nc['target_name'][0] 
    else:
        nc = None
        nuv_name = None
        
    if fc is None and nc is None:
        return (None, None), None
    elif fc is None:
        choice = nc        
    elif nc is None:
        choice = fc  
    else: 
        choice = table.vstack ([fc,nc])
    
    return (fuv_name, nuv_name), choice

def get_galexobs ( ra, dec, radius=None ):
    """
    Get nearby observations from MAST for a given position.

    Parameters:
        ra (float): Right ascension of the position.
        dec (float): Declination of the position.
        radius (astropy.units.quantity.Quantity, optional): Radius around the position (default is 10 arcseconds).

    Returns:
        tuple: A tuple containing the table of nearby observations and the names of the FUV and NUV tiles.

    """    
    if radius is None:
        radius = 10. * u.arcsec
     
    (fuv_name, nuv_name), choice = identify_galexcoadds (ra, dec, radius)
    if choice is None:
        return None, (None, None)
    dproducts = Observations.get_product_list ( choice )   
    #topull = dproducts[dproducts['productGroupDescription'] == 'Minimum Recommended Products']
    return dproducts, (fuv_name, nuv_name)



def download_galeximages ( ra, dec, name, savedir=None, verbose=True, subdirs=True, **kwargs):
    """
    Download GALEX observations for a single target.

    Parameters:
        ra (float): Right ascension of the target.
        dec (float): Declination of the target.
        name (str): Name of the target.
        savedir (str, optional): Directory to save the downloaded files (default is None, which saves in '~/Downloads/').
        **kwargs: Additional keyword arguments to pass to get_galexobs().

    Returns:
        tuple: A tuple containing the exit status (0 if successful, 1 otherwise) and a message.

    """    
    if savedir is None:
        savedir = f'{os.environ["HOME"]}/Downloads/'
    if subdirs and os.path.exists(f'{savedir}/{name}/'):
        return 0, "Already run", None
    
    start = time.time ()
    topull, names = get_galexobs ( ra, dec, **kwargs )
    if verbose:
        print(f'Identified GALEX observations in {time.time() - start:.2f} seconds.')
        start = time.time ()
        
    if topull is None:
        print(f'No Galex observations found for {name}')
        return 1, f'No Galex observations found for {name}', None
    
    if subdirs:
        target = f'{savedir}/{name}/'
    else:
        target = f'{savedir}/'
    if not os.path.exists(target):
        os.makedirs(target)
        
    open(f'{target}/keys.txt','w').write(f'''FUV,{names[0]}
NUV,{names[1]}''')    
    manifest = Observations.download_products(topull, download_dir=target, mrp_only=True, cache=False )
    if verbose:
        print(f'Downloaded GALEX observations in {time.time() - start:.2f} seconds.')
        start = time.time ()    
    for fname in manifest:
        lpath = fname['Local Path']
        filename = os.path.basename(lpath)
        newname = f'{target}/{filename}'
        if os.path.exists(lpath):
            os.rename ( lpath, newname )
    if verbose:
        print(f'Saved to: {os.path.dirname(lpath)}')
    os.removedirs(os.path.dirname(lpath))
    return 0, manifest, names


def load_galexcutouts ( name, datadir, center, sw, sh, verbose=True, infer_names=False, fits_names=None, subdirs=True, clean=False):
    """
    Load locally saved GALEX cutouts and package as a minimal FITS for a target.

    Args:
        name: str - Name of the target.
        datadir: str - Directory containing the GALEX data.
        center: tuple or astropy.coordinates.SkyCoord - Center coordinates for the cutout, either as pixel coordinates 
                (tuple) or as RA/DEC (astropy.coordinates.SkyCoord).
        sw: int - Width of the cutout.
        sh: int - Height of the cutout.
        verbose: bool - Whether to display verbose information (default is True).
        infer_names: bool - Whether to infer the names of the cutouts if keys.txt is not present (default is False).
        fits_names: dict, list, tuple, or None - FITS names provided directly (default is None).
        subdirs: bool - Whether to use subdirectories within the data directory (default is True).
        clean: bool - Whether to remove FITS files after processing (default is False).

    Return:
        output: dict - A dictionary containing the NUV and FUV cutouts (as fits.HDUList objects) with keys 'nd' and 'fd', respectively.

    Raise:
        ValueError: If more than one FUV or NUV intensity map is found.
        OSError: If no FITS names are supplied, keys.txt is missing, and infer_names is disallowed.
    """
    keypath = f'{datadir}/{name}/keys.txt'
    if fits_names is not None:  #  \\ method 1: just give me the FITS       
        if isinstance(fits_names, dict):
            fuv_name = fits_names['FUV']
            nuv_name = fits_names['NUV']
        elif isinstance(fits_names, tuple) or isinstance(fits_names, list):
            fuv_name = fits_names[0]
            nuv_name = fits_names[1]
    elif os.path.exists(keypath): # \\ method 2: logged mapping        
        keyinfo = open(keypath,'r').read().splitlines()
        fuv_name = keyinfo[0].split(',')[1]
        nuv_name = keyinfo[1].split(',')[1]               
    elif infer_names: # \\ method 3: swing for it from the data structure
        fuv_imap = glob.glob(f'{datadir}/{name}/*fd-int.fits*')
        nuv_imap = glob.glob(f'{datadir}/{name}/*nd-int.fits*')
        
        # \\ check to make sure not ambiguous
        if len(fuv_imap) > 1:
            raise ValueError (f"More than one FUV intensity map found: {fuv_imap}")
        if len(nuv_imap) > 1:
            raise ValueError (f"More than one NUV intensity map found: {nuv_imap}")    
        
        # \\ Check to see if cutouts are present
        if len(fuv_imap) == 0:
            print(f'[{name}] No FUV imaging found')
            fuv_name = None
        else:
            fuv_name = os.path.basename ( fuv_imap[0] ).split('-')[0]
            
        if len(nuv_imap) == 0:
            print(f'[{name}] No NUV imaging found')
            nuv_name = None
        else:
            nuv_name = os.path.basename ( fuv_imap[0] ).split('-')[0]
    else:
        raise OSError ("No FITS names supplied, no keys.txt, and infer_names is disallowed!")

    # \ fix for weird key non-matches with AIS, in particular
    if fuv_name is not None:
        if 'AIS' in fuv_name:
            parts = fuv_name.split('_') 
            fuv_name = '_'.join(parts[:2]) + '_sg' + parts[-1].zfill(2)
    if nuv_name is not None:
        if 'AIS' in nuv_name:
            parts = nuv_name.split('_')        
            nuv_name = '_'.join(parts[:2]) + '_sg' + parts[-1].zfill(2)  
           
    # \\ Fetch cutouts
    band_names = ['nd','fd']
    output = {}
    if subdirs:
        data_path = f'{datadir}/{name}/'
    else:
        data_path = datadir    
    
    for ix,prefix in enumerate([nuv_name, fuv_name]):
        if str(prefix).capitalize() == "None":
            output[band_names[ix]] = None
            continue
        if verbose:
            print(f'{name} maps to {prefix}')      
        

        intmap = fits.open ( f'{data_path}/{prefix}-{band_names[ix]}-int.fits.gz')
        rrhr = fits.open (   f'{data_path}/{prefix}-{band_names[ix]}-rrhr.fits.gz')
        _skybg = fits.open ( f'{data_path}/{prefix}-{band_names[ix]}-skybg.fits.gz')
        #skybg[0].name = 'SKYBG'
        skybg = fits.ImageHDU(_skybg[0].data, _skybg[0].header, name='SKYBG')
        
        intmap[0].name = 'INTENSITY'
        
        # -----------------
        # -- Make cutout --
        # -----------------
        cutout_wcs = wcs.WCS ( intmap[0] )
        #pixscale = cutout_wcs.pixel_scale_matrix[1,1]*3600. # arcsec / pix
        #pixcenter = cutout_wcs.wcs_world2pix ( np.array([center]), 1 )[0].astype(int)
        
        #sw_pix = sw / pixscale
        #sh_pix = sh / pixscale
        
        #slice_a0 = slice ( pixcenter[1]-sh, pixcenter[1]+sh )
        #slice_a1 = slice ( pixcenter[0]-sw, pixcenter[0]+sw )        
                  
        # \\ Following McQuinn+2015 [2015ApJS..218...29M], Table 2 
        # \\ this is Poisson error on the cts (var = cts) divided    
        # \\ by effective exposure time
        cts = intmap[0].data * rrhr[0].data      
        variance = fits.ImageHDU ( cts / rrhr[0].data**2, header=intmap[0].header, name='VARIANCE' )
        
        intmap_cutout = Cutout2D( data=intmap[0].data, position=center, size=[sh,sw], wcs=cutout_wcs )
        intmap_hdu = fits.PrimaryHDU ( data=intmap_cutout.data, header=intmap_cutout.wcs.to_header(),)
        variance_cutout = Cutout2D ( variance.data, position=center, size=[sh,sw], wcs=cutout_wcs )
        variance_hdu = fits.ImageHDU ( data=variance_cutout.data, header=variance_cutout.wcs.to_header(), name='VARIANCE')
        skybg_cutout = Cutout2D ( skybg.data, position=center, size=[sh,sw], wcs=cutout_wcs )
        skybg_hdu = fits.ImageHDU ( data=skybg_cutout.data, header=skybg_cutout.wcs.to_header(), name='SKYBG')
        
        #new_wcs = intmap_cutout.wcs.to_header ()
        #intmap[0].data = intmap_cutout
        #intmap[0].header.update ( new_wcs )
        #variance.data = Cutout2D ( variance.data, position=center, size=[sh,sw], wcs=cutout_wcs )
        #variance.header.update ( new_wcs )
        #skybg.data = Cutout2D ( skybg.data, position=center, size=[sh,sw], wcs=cutout_wcs )
        #skybg.header.update ( new_wcs )
        #variance.data = variance.data[slice_a0,slice_a1]
        #skybg.data = skybg.data[slice_a0,slice_a1]            
        
        ofit = fits.HDUList ( [ intmap_hdu, variance_hdu, skybg_hdu ] )
        
        output[band_names[ix]] = ofit
        
        if clean:
            for product in ['skybg','flags','int','rrhr']:
                fname = f'{data_path}/{prefix}-{band_names[ix]}-{product}.fits.gz'
                if os.path.exists(fname):
                    os.remove(fname)
                    if verbose:
                        print(f'...removed {fname}')                      
    
    if clean:    
        fname = f'{data_path}/{prefix}-xd-mcat.fits.gz'
        if os.path.exists(fname):
            os.remove(fname)     
            if verbose:
                print(f'...removed {fname}')               
    
    return output


def get_legacysurveyimage ( ra, dec, width=100, height=100, pixscale=0.13, layer='ls-dr9', format='fits', savedir='./', savename=None, verbose=True ):
    if hasattr(ra, 'unit'):
        ra = ra.to(u.deg).value
    if hasattr(dec, 'unit'):
        dec = dec.to(u.deg).value
    #url = f"https://www.legacysurvey.org/viewer/cutout.jpg?ra=36.1247&dec=-6.1085&layer=ls-dr9&pixscale=1&width=300&height=300"
    url = f"https://www.legacysurvey.org/viewer/cutout.{format}?ra={ra}&dec={dec}&layer={layer}&pixscale={pixscale}&width={width}&height={height}"
    
    # Send a GET request to the URL
    response = requests.get(url)

    # Write the content of the response to a file
    if savename is None:
        savename=f'cutout'
    file_name = f"{savedir}{savename}.{format}"        
    if verbose:
        print(f'''Saving:
{url}
to {file_name}''')
    with open(file_name, 'wb') as file:
        file.write(response.content)
        
def load_gamacatalogs (gama_dir=None):
    '''
    Just load in GAMA DR4 SpecObj + LAMBDAR stellar masses 
    '''
    if gama_dir is None:
        gama_dir = '/Users/kadofong/work/projects/gama/'
    gama = table.Table(fits.getdata(f'{gama_dir}local_data/SpecObjv27.fits', 1)).to_pandas().set_index("CATAID")
    gama_masses = table.Table(fits.getdata(f'{gama_dir}local_data/StellarMassesLambdarv24.fits', 1)).to_pandas().set_index("CATAID")

    catalog = gama.join(gama_masses[['logmstar','dellogmstar']])
    return catalog

def match_catalogs (
        catA, 
        catB, 
        radius=3.*u.arcsec,
        coordkeysA = ['RA','DEC'],
        coordkeysB = ['RA','DEC']
    ):
    catA_coords = coordinates.SkyCoord(
        catA[coordkeysA[0]].values,
        catA[coordkeysA[1]].values,
        unit='deg'
    )
    catB_coords = coordinates.SkyCoord(
        catB[coordkeysB[0]].values,
        catB[coordkeysB[1]].values,
        unit='deg'
    )
    
    catB_correspondence, d2d, _ = catA_coords.match_to_catalog_sky(catB_coords)
    is_match = d2d < (4.*u.arcsec)
    catB_matching_index = catB.index[catB_correspondence[is_match]]
    catA_matching_index = catA.index[np.arange(len(catA_coords), dtype=int)[is_match]]
    
    return catA.reindex(catA_matching_index), catB.reindex(catB_matching_index)