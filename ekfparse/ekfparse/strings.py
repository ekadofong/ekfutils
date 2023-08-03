import re
import numpy as np
from astropy import coordinates
from astropy import units as u

def where_substring (arr, substring):
    '''
    Return indices of array [arr] that contain the substring [substring]
    '''
    return np.flatnonzero(np.core.defchararray.find ( arr, substring )!=-1)

def latex_to_float ( x ):
    '''
    Converts LaTeX of the form
    $ A \times 10^{B} $ 
    to 
    float(A * 10.**B)
    '''
    first = float(re.findall( '(?<=\$).*(?=\\\\)', x )[0])
    exponent = float(re.findall( '(?<=10\^\{).*(?=\})', x )[0])
    result = first * 10**exponent
    return result

def hmsdms2deg ( *args, **kwargs ):
    if len(args) == 2:
        ra, dec = args
        
        ra_h, ra_m, ra_s = ra
        dec_sign, dec_d, dec_m, dec_s = dec
    elif len(args) == 1:        
        ra_h = args[0]['RAh']
        ra_m = args[0]['RAm']
        ra_s = args[0]['RAs']
        
        dec_sign = args[0]['DE-']
        dec_d = args[0]['DEd']
        dec_m = args[0]['DEm']
        dec_s = args[0]['DEs']
        


    c_strings = [ f"{ra_h[i]}:{ra_m[i]}:{ra_s[i]} {dec_sign[i]}{dec_d[i]}:{dec_m[i]}:{dec_s[i]}" for i in range(len(ra_h)) ]
    catalog_coords = coordinates.SkyCoord ( c_strings, unit=(u.hourangle, u.deg))
    return catalog_coords