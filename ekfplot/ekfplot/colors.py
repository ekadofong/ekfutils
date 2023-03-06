import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mpc
import colorsys

def my_favorite_colors ():
    '''
    Returns a list with my favorite colors
    '''
    red_base = ColorBase ( "#f05a4f" )
    return red_base

class ColorBase ( object ):
    '''
    A utility class that allows us to change aspects of a color around a base.
    '''
    def __init__ ( self, color, system='hexcolor' ):
        if system == 'hexcolor':
            self.base =  np.array(mpc.to_rgba ( color ))
        elif system=='mpl_named':
            self.base = np.array(mpc.to_rgba (color))
        elif system == 'rgb':
            if max(color) > 1.:
                self.base = np.concatenate([np.array(color)/255.,[1.]])
            else:
                self.base = np.concatenate([color,[1.]])
        elif system == 'hsv':
            self.base = np.concatenate([colorsys.hsv_to_rgb (*color ), [1.]])
        elif system == 'rgba':
            if max(color) > 1.:
                self.base = np.concatenate([np.array(color[:3])/255.,color[-1:]])
            else:
                self.base = color        
        else:
            raise KeyError (f'Color system {system} not understood!')

    @property 
    def rgb_base ( self ):
        return self.base[:-1]

    def modulate ( self, dl=0., ds=0. ):
        hls_color = colorsys.rgb_to_hls ( *self.rgb_base )
        modulated = [
            hls_color[0], 
            max(0.,min(1.,hls_color[1]+dl)), 
            max(0.,min(1.,hls_color[2]+ds)) 
            ]
        
        modulated = colorsys.hls_to_rgb ( *modulated )
        modulated = ColorBase ( modulated, system='rgb' )
        return modulated
    
    def clarify ( self, alpha_new ):
        updated_base = self.base.copy()
        updated_base[-1] = alpha_new
        new_cb = ColorBase ( updated_base, system='rgba' )
        return new_cb
    
    def lighten ( self, value ):
        hls_color = colorsys.rgb_to_hls ( *self.rgb_base )
        lightened = [hls_color[0], max(0.,min(1.,hls_color[1]+value)), hls_color[2] ]

        output_rgb = colorsys.hls_to_rgb ( *lightened )
        return output_rgb
        
    @property
    def complement ( self ):
        complementary = 1. - np.array(self.rgb_base)
        return ColorBase(tuple(complementary), system='rgb')
    
    def evenly_spaced_colors (self, ncolors):
        hsv = colorsys.rgb_to_hsv(*self.rgb_base)
        
        wedge = 1./ncolors
        hues = [ (hsv[0] + wedge*idx) % 1. for idx in range(ncolors)] 
        hsv_l = [ ColorBase( (h, hsv[1], hsv[2]), 'hsv') for h in hues ]
        return hsv_l
        
        
    def sequential_cmap ( self, end_color='w', end_color_system='mpl_named', reverse=False, fade=1. ):
        end_color = ColorBase ( end_color, end_color_system )
        if fade<1.:
            end_color = end_color.clarify ( fade )
            
        if fade:
            end_color_code = end_color
        if reverse:
            cmap = mpc.LinearSegmentedColormap.from_list ( 'sequential', [end_color.base,self.base], )
        else:
            cmap = mpc.LinearSegmentedColormap.from_list ( 'sequential', [self.base, end_color.base], )
            
        return cmap
    
    