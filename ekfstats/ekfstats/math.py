import numpy as np
from scipy.spatial import cKDTree

def trapz ( y, x):    
    order = np.argsort(x)
    return np.trapz(y[order], x[order])

def ls_2ptcorr(target_catalog, random_catalog, r_bins, pos_cols=['RA', 'DEC']):
    """
    Calculate the Landy & Szalay 1993 2-point correlation function estimator.
    
    Parameters:
    -----------
    target_catalog : pandas.DataFrame or array_like
        Target galaxy catalog with position columns
    random_catalog : pandas.DataFrame or array_like  
        Random catalog with position columns
    r_bins : array_like
        Separation bins (in same units as position columns)
    pos_cols : list, optional
        Position column names, default ['RA', 'DEC']
        
    Returns:
    --------
    xi : array_like
        The 2-point correlation function estimate at bin centers
    r_centers : array_like
        Bin centers
        
    Notes:
    ------
    Implements the Landy & Szalay (1993) estimator: xi = (DD - 2*DR + RR) / RR
    Uses scipy.spatial.cKDTree.count_neighbors for efficient pair counting.
    
    Reference:
    ----------
    Landy, S. D., & Szalay, A. S. 1993, ApJ, 412, 64
    """
    # Extract positions
    if hasattr(target_catalog, 'columns'):  # pandas DataFrame
        target_pos = target_catalog[pos_cols].values
        random_pos = random_catalog[pos_cols].values
    else:  # assume array-like
        target_pos = np.array(target_catalog)
        random_pos = np.array(random_catalog)
    
    # Build KDTrees
    target_tree = cKDTree(target_pos)
    random_tree = cKDTree(random_pos)
    
    # Count pairs in each bin
    r_bins = np.asarray(r_bins)
    
    # DD: data-data pairs
    DD = np.zeros(len(r_bins) - 1)
    for i in range(len(r_bins) - 1):
        if i == 0:
            DD[i] = target_tree.count_neighbors(target_tree, r_bins[i+1])
        else:
            DD[i] = (target_tree.count_neighbors(target_tree, r_bins[i+1]) - 
                    target_tree.count_neighbors(target_tree, r_bins[i]))
    
    # DR: data-random pairs  
    DR = np.zeros(len(r_bins) - 1)
    for i in range(len(r_bins) - 1):
        if i == 0:
            DR[i] = target_tree.count_neighbors(random_tree, r_bins[i+1])
        else:
            DR[i] = (target_tree.count_neighbors(random_tree, r_bins[i+1]) - 
                    target_tree.count_neighbors(random_tree, r_bins[i]))
    
    # RR: random-random pairs
    RR = np.zeros(len(r_bins) - 1)  
    for i in range(len(r_bins) - 1):
        if i == 0:
            RR[i] = random_tree.count_neighbors(random_tree, r_bins[i+1])
        else:
            RR[i] = (random_tree.count_neighbors(random_tree, r_bins[i+1]) - 
                    random_tree.count_neighbors(random_tree, r_bins[i]))
    
    # Calculate Landy-Szalay estimator
    mask = RR > 0
    xi = np.zeros_like(RR, dtype=float)
    xi[mask] = (DD[mask] - 2*DR[mask] + RR[mask]) / RR[mask]
    
    # Bin centers
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    
    return xi, r_centers