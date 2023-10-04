import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from astropy import units as u

def show_beam ( xy, bmaj, bmin, bpa, ax=None, unit='deg', pixscale=None, **kwargs ):
    if ax is None:
        ax = plt.subplot(111)
    if 'lw' not in kwargs.keys():
        kwargs['lw'] = 2
    if 'facecolor' not in kwargs.keys():
        kwargs['facecolor'] = 'None'
    if 'edgecolor' not in kwargs.keys():
        kwargs['edgecolor'] = 'k'
    if hasattr ( xy[0], 'unit'):
        xy = (xy[0].to(unit.value), xy[1].to(unit.value))
        
    if pixscale is not None:
        bmin = (bmin / pixscale).decompose()        
        assert bmin.unit.is_unity()
        bmin = bmin.value
        bmaj = (bmaj / pixscale).decompose()
        assert bmaj.unit.is_unity()
        bmaj = bmaj.value
    
    ellipse = patches.Ellipse ( 
                               xy,
                               hasattr(bmin,'unit') and bmin.to(unit).value or bmin, 
                               hasattr(bmaj,'unit') and bmin.to(unit).value or bmaj, 
                               hasattr(bpa,'unit') and bpa.to(u.deg).value or bpa,
                               **kwargs
                            )
    ax.add_patch ( ellipse )
    