import time
import os
import glob
import logging
import urllib
import requests
from xml.etree import ElementTree as ET
import numpy as np

import pandas as pd
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

def get_galexobs ( ra, dec, radius=None, verbose=True ):
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
        if verbose:
            print('[ekfparse.query.get_galexobs] No Galex observations!')
        return None, (None, None)
    dproducts = Observations.get_product_list ( choice )   
    #topull = dproducts[dproducts['productGroupDescription'] == 'Minimum Recommended Products']
    return dproducts, (fuv_name, nuv_name)

def hotfix_galex_naming ( obsid ):
    if obsid is None:
        return None
        
    if 'AIS' in obsid:
        parts = obsid.split('_')        
        obsid = '_'.join(parts[:2]) + '_sg' + parts[-1].zfill(2)     
    return obsid

def download_galeximages ( ra, dec, name, savedir=None, verbose=True, subdirs=True, sparse_download=True, obsout=None, **kwargs):
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
    #if subdirs and os.path.exists(f'{savedir}/{name}/'):
    #    return 0, "Already run", None
    
    start = time.time ()
    if obsout is None:        
        topull, names = get_galexobs ( ra, dec, **kwargs )
    else:
        topull, names = obsout

    if np.all([ x is None for x in names]):
        if verbose:
            print('[ekfparse.query.download_galeximages] No Galex coadds identified!')
        return 1, f'No Galex observations found for {name}', None        
        
        
    if sparse_download:
        # \\ AIS filename fix
        anames = [ hotfix_galex_naming(_name) for _name in names ]
                                
        do_pull = np.full(len(topull), False)
        downloaded_images = [ os.path.basename(x) for x in glob.glob(f'{savedir}/{name}/*') ]
        already_downloaded = np.in1d(topull['productFilename'], downloaded_images)
        for _ in range(len(topull)):                        
            is_from_best = False # \\ only pull products from deepest available observations
            is_downloaded = False # \\ do not re-download products
            
            product_filename = topull[_]['productFilename']
            for _name in anames:
                if _name is None:
                    continue
                elif _name in product_filename:
                    is_from_best = True
            
            do_pull[_] = is_from_best
                    
        topull = topull[do_pull&~already_downloaded]

    if verbose:
        print(f'Identified GALEX observations in {time.time() - start:.2f} seconds.')
        start = time.time ()

    if topull is None:
        print(f'No Galex observations found for {name}')
        return 1, f'No Galex observations found for {name}', None
    elif len(topull) == 0:
        print(f'All observations already downloaded for {name}')
        return 2, f'All observations already downloaded for {name}', None
    
    if subdirs:
        target = f'{savedir}/{name}/'
    else:
        target = f'{savedir}/'
    if not os.path.exists(target):
        os.makedirs(target)
    
    with  open(f'{target}/keys.txt','w') as f:
        f.write(f'''FUV,{names[0]}
NUV,{names[1]}''')
            
    manifest = Observations.download_products(topull, download_dir=target, mrp_only=True, cache=False )
    if manifest is None:
        return 2, "Nothing to download!", None
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


def load_galexcutouts ( name, datadir, center=None, sw=60, sh=60, verbose=True, infer_names=False, fits_names=None, subdirs=True, clean=False):
    """
    Load locally saved GALEX cutouts and package as a minimal FITS for a target.

    Args:
        name: str - Name of the target.
        datadir: str - Directory containing the GALEX data.
        center: tuple or astropy.coordinates.SkyCoord - Center coordinates for the cutout, either as pixel coordinates 
                (tuple) or as RA/DEC (astropy.coordinates.SkyCoord).
        sw: int - Width of the cutout on the sky (arcsec).
        sh: int - Height of the cutout on the sky (arcsec).
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
        
        if center is None:
            if verbose:
                print('[query.load_galexcutouts] returning full cutout')
            intmap_hdu = intmap[0]
            variance_hdu = variance
            skybg_hdu = skybg
        else:
            if isinstance(center, tuple):
                print('[query.load_galexcutouts] `center` given as tuple; assuming pixel values')
            intmap_cutout = Cutout2D( data=intmap[0].data, position=center, size=[sh,sw], wcs=cutout_wcs )
            variance_cutout = Cutout2D ( variance.data, position=center, size=[sh,sw], wcs=cutout_wcs )
            skybg_cutout = Cutout2D ( skybg.data, position=center, size=[sh,sw], wcs=cutout_wcs )
            
            intmap_hdu = fits.PrimaryHDU ( data=intmap_cutout.data, header=intmap_cutout.wcs.to_header(),)    
            variance_hdu = fits.ImageHDU ( data=variance_cutout.data, header=variance_cutout.wcs.to_header(), name='VARIANCE')        
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
    
    return file_name
        
def load_gamacatalogs (gama_dir=None):
    '''
    Just load in GAMA DR4 SpecObj + LAMBDAR stellar masses 
    '''
    if gama_dir is None:
        gama_dir = '/Users/kadofong/work/projects/gama/'
    gama = table.Table(fits.getdata(f'{gama_dir}local_data/SpecObjv27.fits', 1)).to_pandas().set_index("CATAID")
    # \\ equatorial & G23 survey region masses
    gama_masses = table.Table(fits.getdata(f'{gama_dir}local_data/StellarMassesLambdarv24.fits', 1)).to_pandas().set_index("CATAID")
    # \\ G02 stellar masses
    g02_masses = table.Table(fits.getdata(f'{gama_dir}local_data/StellarMassesG02SDSSv24.fits', 1)).to_pandas().set_index("CATAID")
    gama_masses = pd.concat([gama_masses, g02_masses])
    gama_lines = table.Table(fits.getdata(f'{gama_dir}local_data/GaussFitComplexv05.fits', 1)).to_pandas()#.set_index("CATAID")
    #gama_lines = gama_lines.loc[~gama_lines.index.duplicated()]
    gama_phot = table.Table(fits.getdata(f'{gama_dir}local_data/InputCatAv07.fits', 1)).to_pandas().set_index("CATAID")
    gama_phot = gama_phot.loc[~gama_phot.index.duplicated()]
    
    #\\photometry
    gama_phot['r_mag'] = gama_phot['PETROMAG_R'] #-2.5*np.log10(gama_phot['r_flux']/3631.)
    #r_circ_pixel = gama_phot['PETRO_RADIUS'] * gama_phot['B_IMAGE']/gama_phot['A_IMAGE']
    #arcsec_per_pizel = 0.339 # arcsec / pixel, see final notes of https://www.gama-survey.org/dr4/schema/table.php?id=445
    #r_circ_arcsec = r_circ_pixel * arcsec_per_pizel
    #gama_phot['radius'] = r_circ_arcsec
    #gama_phot['sb_r'] = gama_phot['r_mag'] + 2.5*np.log10(2.*np.pi*gama_phot['radius']**2)
    
    for band in 'gri':
        gama_phot[f'{band}_flux'] = 10.**(gama_phot[f'PETROMAG_{band.upper()}']/-2.5) * 3631*1e9

    catalog = gama.join(gama_masses[['logmstar','dellogmstar','logage','dellogage','absmag_g','absmag_r']])\
        .reset_index().merge(gama_lines[['SPECID',
                           'HA_FLUX','HA_FLUX_ERR','HA_EW','HA_EW_ERR',
                           'HB_EW','HB_EW_ERR',
                           'NIIB_FLUX','NIIB_FLUX_ERR',
                           'NIIR_FLUX','NIIR_FLUX_ERR',
                           ]], on='SPECID').set_index('CATAID')\
            .join(gama_phot[['g_flux','r_flux','i_flux','r_mag']]) # 'radius','sb_r'
            
    survey_recode = np.zeros(catalog.shape[0])
    survey_recode[catalog['SURVEY_CODE']==1] = 1
    survey_recode[catalog['SURVEY_CODE']==5] = 2     
    
    catalog['is_flux_calibrated'] = False
    catalog.loc[survey_recode>0,'is_flux_calibrated'] = True
    
    catalog = catalog.query("NQ>2")
           
    return catalog

def load_sdsscatalogs (sdss_dir=None, zmax=0.05, use_scratch=True):
    if sdss_dir is None:
        sdss_dir = '/Users/kadofong/work/projects/sdss/'
    sname = f'{sdss_dir}/local_data/scratch_sdss.csv'
    if use_scratch and os.path.exists(sname):
        cat = pd.read_csv(sname)
        return cat
    
    sdss = fits.open(f'{sdss_dir}/local_data/specObj-dr17.fits')
    is_lowz = (sdss[1].data['Z']>0.001)&(sdss[1].data['Z']<zmax)
    cat = np.array([
        sdss[1].data['PLUG_RA'][is_lowz],
        sdss[1].data['PLUG_DEC'][is_lowz],
        sdss[1].data['Z'][is_lowz],
    ])
    cat = pd.DataFrame(cat.T, index=sdss[1].data['SPECOBJID'][is_lowz], columns=['RA','DEC','Z'])
    for key in ['SURVEY','RUN2D','PLATE','MJD','FIBERID']:
        cat[key.lower()] = sdss[1].data[key][is_lowz].byteswap().newbyteorder()
    
    return cat

def download_sdss_spectrum ( row=None, run2d=None, plate=None, mjd=None, fiberid=None, savedir='./sdss_spectra/'):
    if (row is None) and (run2d is None):
        raise ValueError ("Must specify row OR spectrum details")
    elif row is not None:
        run_version = row['run2d']
        plate = row['plate']
        mjd = row['mjd']
        fiber = row['fiberid']
    
    # https://data.sdss.org/sas/dr18/spectro/sdss/redux/26/spectra/
    zfill_plate = str(plate).zfill(4)
    zfill_fiber = str(fiber).zfill(4)
    run_version = run_version.strip()
    # spec-0385-51783-0007.fits
    specname = f'spec-{zfill_plate}-{mjd}-{zfill_fiber}.fits'
    if 'v' in row['run2d']:
        extra = 'full/'
    else:
        extra = ''
    spectrum_url = f'https://data.sdss.org/sas/dr18/spectro/sdss/redux/{run_version}/spectra/{extra}{zfill_plate}/{specname}'

    # Send a GET request to the URL
    response = requests.get(spectrum_url)
    
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        
    file_name = f'{savedir}/{specname}'
    with open(file_name, 'wb') as file:
        file.write(response.content)
        
    return fits.open(file_name)

def match_catalogs (
        catA, 
        catB, 
        radius=3.*u.arcsec,
        coordkeysA = ['RA','DEC'],
        coordkeysB = ['RA','DEC'],
        unitsA = ['deg','deg'],
        unitsB = ['deg','deg']
    ):
    catA_coords = coordinates.SkyCoord(
        catA[coordkeysA[0]].values,
        catA[coordkeysA[1]].values,
        unit=unitsA
    )
    catB_coords = coordinates.SkyCoord(
        catB[coordkeysB[0]].values,
        catB[coordkeysB[1]].values,
        unit=unitsB
    )
    
    catB_correspondence, d2d, _ = catA_coords.match_to_catalog_sky(catB_coords)
    is_match = d2d < (4.*u.arcsec)
    catB_matching_index = catB.index[catB_correspondence[is_match]]
    catA_matching_index = catA.index[np.arange(len(catA_coords), dtype=int)[is_match]]
    
    return catA.reindex(catA_matching_index), catB.reindex(catB_matching_index)