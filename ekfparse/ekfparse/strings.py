import numpy as np

def where_substring (arr, substring):
    '''
    Return indices of array [arr] that contain the substring [substring]
    '''
    return np.flatnonzero(np.core.defchararray.find ( arr, substring )!=-1)