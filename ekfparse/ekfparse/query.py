import os
import glob
import urllib
from xml.etree import ElementTree as ET
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy import table
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

def get_nearbyobs ( ra, dec, radius=None ):
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
     
    # \\ get all nearby MAST regions   
    obs_table = Observations.query_region ( f'{ra} {dec}', radius=radius )
    galex_table = obs_table[obs_table['obs_collection'] == "GALEX"]
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
    #return nc,fc
     
    if fc is None and nc is None:
        return None, None
    elif fc is None:
        choice = nc        
    elif nc is None:
        choice = fc  
    else: 
        choice = table.vstack ([fc,nc])
    
    dproducts = Observations.get_product_list ( choice )   
    #topull = dproducts[dproducts['productGroupDescription'] == 'Minimum Recommended Products']
    return dproducts, (fuv_name, nuv_name)

def download_galeximages ( ra, dec, name, savedir=None, **kwargs):
    """
    Download GALEX observations for a single target.

    Parameters:
        ra (float): Right ascension of the target.
        dec (float): Declination of the target.
        name (str): Name of the target.
        savedir (str, optional): Directory to save the downloaded files (default is None, which saves in '~/Downloads/').
        **kwargs: Additional keyword arguments to pass to get_nearbyobs().

    Returns:
        tuple: A tuple containing the exit status (0 if successful, 1 otherwise) and a message.

    """    
    if savedir is None:
        savedir = f'{os.environ["HOME"]}/Downloads/'
    if os.path.exists(f'{savedir}/{name}/'):
        return 0, "Already run"
    topull, names = get_nearbyobs ( ra, dec, **kwargs )
    if topull is None:
        print(f'No Galex observations found for {name}')
        return 1, f'No Galex observations found for {name}'
    
    target = f'{savedir}/{name}/'
    if not os.path.exists(target):
        os.makedirs(target)
        
    open(f'{target}/keys.txt','w').write(f'''FUV,{names[0]}
NUV,{names[1]}''')
    manifest = Observations.download_products(topull, download_dir=target, mrp_only=True )
    
    for fname in manifest:
        lpath = fname['Local Path']
        filename = os.path.basename(lpath)
        newname = f'{target}/{filename}'
        os.rename ( lpath, newname )
    
    print(os.path.dirname(lpath))
    os.removedirs(os.path.dirname(lpath))
    return 0, manifest


def load_galexcutouts ( name, datadir, center, sw, sh, verbose=True, infer_names=False):
    """
    Load locally saved GALEX cutouts and package as a minimal FITS for a target.

    Parameters:
        name (str): Name of the target.
        datadir (str): Directory containing the GALEX data.
        verbose (bool, optional): Whether to display verbose information (default is True).
        infer_names (bool, optional): Whether to infer the names of the cutouts if keys.txt is not present (default is False).

    Returns:
        dict: A dictionary containing the NUV and FUV cutouts (as fits.HDUList objects) with keys 'nd' and 'fd', respectively.

    """
    keypath = f'{datadir}/{name}/keys.txt'
    if os.path.exists(keypath):
        keyinfo = open(keypath,'r').read().splitlines()
        fuv_name = keyinfo[0].split(',')[1]
        nuv_name = keyinfo[1].split(',')[1]
        
        # \ fix for weird key non-matches with AIS, in particular
        if 'AIS' in fuv_name:
            parts = fuv_name.split('_') 
            fuv_name = '_'.join(parts[:2]) + '_sg' + parts[-1].zfill(2)
        if 'AIS' in nuv_name:
            parts = nuv_name.split('_')        
            nuv_name = '_'.join(parts[:2]) + '_sg' + parts[-1].zfill(2)
    
            
    elif infer_names:
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
        raise OSError ("No keys.txt and infer_names is disallowed!")
    
    # \\ Fetch cutouts
    band_names = ['nd','fd']
    output = {}
    
    for ix,prefix in enumerate([nuv_name, fuv_name]):
        if str(prefix).capitalize() == "None":
            output[band_names[ix]] = None
            continue
        if verbose:
            print(f'{name} maps to {prefix}')      
        
        intmap = fits.open ( f'{datadir}/{name}/{prefix}-{band_names[ix]}-int.fits.gz')
        rrhr = fits.open ( f'{datadir}/{name}/{prefix}-{band_names[ix]}-rrhr.fits.gz')
        skybg = fits.open ( f'{datadir}/{name}/{prefix}-{band_names[ix]}-skybg.fits.gz')
        
        skybg.name = 'SKYBG'
        
        cts = intmap[0].data * rrhr[0].data        
        # \\ Following McQuinn+2015, this is Poisson error on the cts (var = cts) divided
        # \\ by effective exposure time
        variance = fits.ImageHDU ( cts / rrhr[0].data**2, header=intmap[0].header, name='VARIANCE' )
        
        ofit = fits.HDUList ( [ intmap[0], variance, skybg[0] ] )
        
        output[band_names[ix]] = ofit
    return output