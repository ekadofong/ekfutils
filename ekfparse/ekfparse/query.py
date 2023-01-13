import urllib
from xml.etree import ElementTree as ET

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