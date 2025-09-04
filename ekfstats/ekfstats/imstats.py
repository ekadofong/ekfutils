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

def bspline_star(x, step):
    """
    FROM https://github.com/broxtronix/pymultiscale/blob/master/pymultiscale/starlet.py
    
    This implements the starlet kernel. Application to different scales is
    accomplished via the step parameter.
    """
    ndim = len(x.shape)
    C1 = 1./16.
    C2 = 4./16.
    C3 = 6./16.

    KSize = 4*step+1
    KS2 = KSize//2
    
    kernel = np.zeros((KSize), dtype = np.float32)
    if KSize == 1:
        kernel[0] = 1.0
    else:
        kernel[0] = C1
        kernel[KSize-1] = C1
        kernel[KS2+step] = C2
        kernel[KS2-step] = C2
        kernel[KS2] = C3

    # Based on benchmarks conducted during January 2015, OpenCV has a far faster
    # seperabable convolution routine than scipy does.  We use it for 2D images
    if ndim == 2:
        import cv2
        result = cv2.sepFilter2D(x, cv2.CV_32F, kernelX = kernel, kernelY = kernel)
        return result

    else:
        result = x
        import scipy.ndimage
        for dim in np.arange(ndim):
            result = scipy.ndimage.filters.convolve1d(result, kernel, axis = dim, mode='reflect', cval = 0.0)
    return result


# -----------------------------------------------------------------------------
#                            FUNCTION API
# -----------------------------------------------------------------------------

def starlet_transform(input_image, num_bands = None, gen2 = True):
    '''
    FROM https://github.com/broxtronix/pymultiscale/blob/master/pymultiscale/starlet.py
    
    Computes the starlet transform of an image (i.e. undecimated isotropic
    wavelet transform).

    The output is a python list containing the sub-bands. If the keyword Gen2 is set,
    then it is the 2nd generation starlet transform which is computed: i.e. g = Id - h*h
    instead of g = Id - h.

    REFERENCES:
    [1] J.L. Starck and F. Murtagh, "Image Restoration with Noise Suppression Using the Wavelet Transform",
    Astronomy and Astrophysics, 288, pp-343-348, 1994.

    For the modified STARLET transform:
    [2] J.-L. Starck, J. Fadili and F. Murtagh, "The Undecimated Wavelet Decomposition
        and its Reconstruction", IEEE Transaction on Image Processing,  16,  2, pp 297--309, 2007.

    This code is based on the STAR2D IDL function written by J.L. Starck.
            http://www.multiresolutions.com/sparsesignalrecipes/software.html

    '''

    if num_bands == None:
        num_bands = int(np.ceil(np.log2(np.min(input_image.shape))) - 3)
        assert num_bands > 0

    ndim = len(input_image.shape)

    im_in = input_image.astype(np.float32)
    step_trou = 1
    im_out = None
    WT = []

    for band in np.arange(num_bands):
        im_out = bspline_star(im_in, step_trou)
        if gen2:  # Gen2 starlet applies smoothing twice
            WT.append(im_in - bspline_star(im_out, step_trou))
        else:
            test = im_in - im_out
            WT.append(im_in - im_out)
        im_in = im_out
        step_trou *= 2

    WT.append(im_out)
    return WT

def inverse_starlet_transform(coefs, gen2 = True):
    '''
    Computes the inverse starlet transform of an image (i.e. undecimated
    isotropic wavelet transform).

    The input is a python list containing the sub-bands. If the keyword Gen2 is
    set, then it is the 2nd generation starlet transform which is computed: i.e.
    g = Id - h*h instead of g = Id - h.

    REFERENCES:
    [1] J.L. Starck and F. Murtagh, "Image Restoration with Noise Suppression Using the Wavelet Transform",
        Astronomy and Astrophysics, 288, pp-343-348, 1994.

    For the modified STARLET transform:
    [2] J.-L. Starck, J. Fadili and F. Murtagh, "The Undecimated Wavelet Decomposition
        and its Reconstruction", IEEE Transaction on Image Processing,  16,  2, pp 297--309, 2007.

    This code is based on the ISTAR2D IDL function written by J.L. Starck.
            http://www.multiresolutions.com/sparsesignalrecipes/software.html
    '''

    # Gen1 starlet can be reconstructed simply by summing the coefficients at each scale.
    if not gen2:
        recon_img = np.zeros_like(coefs[0])
        for i in np.arange(len(coefs)):
            recon_img += coefs[i]

    # Gen2 starlet requires more careful reconstruction.
    else:
        num_bands = len(coefs)-1
        recon_img = coefs[-1]
        step_trou = int(np.power(2, num_bands - 1))

        for i in reversed(range(num_bands)):
            im_temp = bspline_star(recon_img, step_trou)
            recon_img = im_temp + coefs[i]
            step_trou = step_trou//2

    return recon_img