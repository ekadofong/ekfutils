import numpy as np

def gini(x, w=None, qt_cut=0.):
    x = x.flatten()
    x = x[x>np.nanquantile(x, qt_cut)]
    sorted_x = np.sort(x)
    n = len(x)
    indices = np.arange(1,1+n)
    interior = np.sum( (2.*indices - n - 1.)*abs(sorted_x) ) 
    factor = (np.mean(abs(x)) * n * (n-1.) )**-1
    return interior * factor
    #cumx = np.cumsum(sorted_x, dtype=float)
    # The above formula, with all weights equal to 1 simplifies to:
    #return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def get_center(arr):
    """
    Calculate the coordinates of the center of the array.

    Parameters:
    - arr (numpy.ndarray): The input array.

    Returns:
    - numpy.ndarray: The coordinates of the center of the array.
    """
    ccoord = np.array(arr.shape) // 2
    return ccoord

def get_centerval(arr):
    """
    Get the value at the center of the array.

    Parameters:
    - arr (numpy.ndarray): The input array.

    Returns:
    - int or float: The value at the center of the array.
    """
    ccoord = get_center(arr)
    return arr[ccoord[0], ccoord[1]]

def rotate_coordinates ( x, y, theta ):
    '''
    Rotate by theta [rad] for rotation matrix
    | cos(theta), -sin(theta) |
    | sin(theta),  cos(theta) |    
    '''
    xr = x*np.cos(theta) - y*np.sin(theta)
    yr = x*np.sin(theta) + y*np.cos(theta)
    return xr, yr

def build_xygrid ( shape, x_0=None, y_0=None, theta=0., ellip=0.):
    if x_0 is None:
        x_0 = shape[1]//2
    if y_0 is None:
        y_0 = shape[1]//2
    
    Y,X = np.mgrid[:shape[0],:shape[1]]
    X_c = (X - x_0)
    Y_c = (Y - y_0)
    X_rot, Y_rot = rotate_coordinates( X_c, Y_c, theta )
    Y_rot /=  1. - ellip
    R = np.sqrt(X_rot**2 + Y_rot**2) + 0.0001    
    return R

def build_ellipse_from_sep ( row, cutout_shape, ellipse_size=9.,  ):
    """Builds an elliptical region based on SExtractor parameters.

    This function creates an elliptical region using parameters from a given
    row. The region is defined within a cutout shape.

    Args:
        row (dict): A dictionary containing SExtractor parameters. Expected keys
            are 'y', 'x', 'cyy', 'cxx', and 'cxy'.
        cutout_shape (tuple): A tuple (height, width) representing the shape of
            the cutout in which the ellipse is created.
        ellipse_size (float, optional): Size of the ellipse. Default is 9.0.

    Returns:
        numpy.ndarray: A boolean array where True represents the region inside
        the ellipse.
    """    
    Y,X = np.mgrid[:cutout_shape[1],:cutout_shape[0]]
        
    yoff = Y-row['y']
    xoff = X-row['x']
    ep = row['cyy']*yoff**2 + row['cxx']*xoff**2 + row['cxy']*xoff*yoff            
    regionauto = ep < ellipse_size # this value comes from SExtractor manual :shrug: 
    return regionauto

def build_ellipsed_segmentationmap ( sep_catalog, cutout_shape ):
    """Builds a segmentation map with elliptical regions.

    This function constructs a segmentation map by adding elliptical regions
    for each entry in the provided SExtractor catalog.

    Args:
        sep_catalog (list of dict): A list of dictionaries, each containing
            SExtractor parameters for a different object. Each dictionary should
            have keys 'y', 'x', 'cyy', 'cxx', and 'cxy'.
        cutout_shape (tuple): A tuple (height, width) representing the shape of
            the cutout in which the ellipses are created.

    Returns:
        numpy.ndarray: An integer array representing the segmentation map, where
        each ellipse is added to the map.
    """    
    segmap = np.zeros(cutout_shape, dtype=int)
    for cid in range(len(sep_catalog)):
        isegmap = build_ellipse_from_sep ( sep_catalog[cid], cutout_shape, )
        segmap[isegmap>0] |= 1 << cid
    return segmap