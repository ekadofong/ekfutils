import numpy as np
import matplotlib.pyplot as plt
from . import colors as ec

# https://eleanormaclure.wordpress.com/wp-content/uploads/2011/03/colour-coding.pdf
london_rail_colors = {
    "one": '#5A6A91',
    #"gatwick_express": '#151515', # \\ basically black
    #"gt_n._eastern": '#D89EB6', # \\ too pale
    "chiltern": '#B7AECF',
    "central": '#82C8E6',
    "wessex": '#96B17F',
    "south_eastern": '#ECC15F',
    "silverlink": '#E39F54',
    "south_west": '#D5492D',
    "first_gt_western": '#6A0B1E',
    "c2c": '#CF609F',
    "island_line": '#AC4D9F',
    "thameslink": '#943F98',
    "heathrow_express": '#9FCAA9',
    "first_gt_western_link": '#B9B86A',
    "southern": '#6A9B5F',
    "virgin": '#EEAF7F',
    "wagn": '#A66D44',
    "midland_mainline": '#2F4C8F'
}

hcbold = {
    'red':'#e66a4e',
	'orange':'#eb7f2c',
	'green':'#6fb087',
	'blue':'#65a7c5',
	'grey':'#c6caca',    
}

coolwarm2025 = {
    'teal0':'#69d2e6',
	'teal1':'#a7dbd8',
	'neutral':'#e0e4cc',
	'orange1':'#f38630',
	'orange0':'#fa690',
}

slides = {
    'red':'#E06950', 
    'orange':'#E59858', 
    'yellow':'#E9C46A', 
    'blue':'#4BB8CD'
}

merian = {
    'bold':'#14a9e3',
    'darkbold':'#005678',
    'shark':'#3d3736',
    'grey':'#6e6765',
    'dim':'#c9765b',
    'darkdim':'#80665d',
    'highlight':'#e64100'
}

def display_colorlist (clist):
    xs = np.arange(0,len(clist))
    keys = list(clist.keys())
    
    fig = plt.figure(figsize=(len(clist), 1.))
    ax = plt.subplot(111)
    dy=0.05
    for idx in xs:
        key = keys[idx]
        sign = (idx//2 - idx/2.) < 0. and -1. or 1.
        
        base_color = clist[key]
        if isinstance(base_color, ec.ColorBase):
            base_color = base_color.base
            
        ax.scatter(
            idx,
            1,
            color=base_color,
            s=10**2
        )
        ax.text(
            idx,
            1.+sign*dy,
            key,
            color=base_color    ,
            ha='center',
            va=sign < 0 and 'bottom' or 'top',
        )
    
    for spine_direction in ['top','bottom','left','right']:
        ax.spines[spine_direction].set_alpha(0.)
    ax.set_xticks([])
    ax.set_yticks([])