import re
import numpy as np

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