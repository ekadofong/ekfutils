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