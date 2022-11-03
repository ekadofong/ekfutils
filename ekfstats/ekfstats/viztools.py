import numpy as np

def binned_mean ( x, y, bins ):
    assns = np.digitize ( x, bins )
    midpts = 0.5*(bins[1:]+bins[:-1])
    arr = np.zeros([len(midpts),2])
    for idx in range(1, len(bins)):
        mask = np.isfinite(y)
        arr[idx-1,0] = np.nanmean(y[(assns==idx)&mask])
        arr[idx-1,1] = np.nanstd(y[(assns==idx)&mask])
    return midpts, arr

def binned_median ( x, y, bins, alpha=0.16 ):
    assns = np.digitize ( x, bins )
    midpts = 0.5*(bins[1:]+bins[:-1])
    arr = np.zeros([len(midpts),3])
    for idx in range(1, len(bins)):
        mask = np.isfinite(y)
        arr[idx-1,0] = np.nanmedian(y[(assns==idx)&mask])
        arr[idx-1,1] = np.nanquantile(y[(assns==idx)&mask], alpha)
        arr[idx-1,2] = np.nanquantile(y[(assns==idx)&mask], 1.-alpha)
    return midpts, arr