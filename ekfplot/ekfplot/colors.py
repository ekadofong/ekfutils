import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mpc
import colorsys


class ColorBase ( object ):
    '''
    A utility class that allows us to change aspects of a color around a base.
    '''
    def __init__ ( self, color, system='hexcolor' ):
        if system == 'hexcolor':
            self.base = mpc.to_rgb ( color )
        elif system=='mpl_named':
            self.base = mpc.to_rgb (color)
        elif system == 'rgb':
            if max(color) > 1.:
                self.base = np.array(color)/255.
            else:
                self.base = color
        elif system == 'hsv':
            self.base = colorsys.hsv_to_rgb ( color )
        else:
            raise KeyError (f'Color system {system} not understood!')

    def lighten ( self, value ):
        hls_color = colorsys.rgb_to_hls ( *self.base )
        lightened = [hls_color[0], max(0.,min(1.,hls_color[1]+value)), hls_color[2] ]

        output_rgb = colorsys.hls_to_rgb ( *lightened )
        return output_rgb
        
    def sequential_cmap ( self, end_color='w', end_color_system='mpl_named', reverse=False):
        end_color = ColorBase ( end_color, end_color_system )
        if reverse:
            cmap = mpc.LinearSegmentedColormap.from_list ( 'sequential', [end_color.base,self.base], )
        else:
            cmap = mpc.LinearSegmentedColormap.from_list ( 'sequential', [self.base, end_color.base], )
            
        return cmap