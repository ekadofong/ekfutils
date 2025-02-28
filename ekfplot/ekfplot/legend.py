import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

from . import colors as ec

def add_to_legend_list ( artists, labels, ax, **kwargs ):
    handles, element_labels = ax.get_legend_handles_labels()
    
    handles = handles + artists
    element_labels = element_labels + labels

    ax.legend(handles, element_labels, **kwargs)
    return ax

def add_to_legend ( artist, label, ax, handles, labels, **kwargs ):
    # Append custom elements to the handles and labels
    handles.append(artist)
    labels.append(label)

    # Recreate the legend with the updated handles and labels
    ax.legend(handles, labels)
    return ax

def make_shaded_element ( ax, label, color='tab:red', modulations = [0.4, 0.25], lw=3, **kwargs ):
    if isinstance ( color, str ):
        color = ec.ColorBase(color)
        
    maxwidth = 15
    dlw = (maxwidth - lw)/len(modulations) 
    
    lines  = [ Line2D([], [], 
                      linewidth=maxwidth - dlw*ix, 
                      dashes=(100,1), 
                      color=color.modulate(modulations[ix]).base
                      )\
        for ix in range(len(modulations)) ]
    lines.append(Line2D([], [], linewidth=lw, dashes=(100,1), color=color.base),  )
        
    artist = tuple(lines)
    #handles, labels = add_to_legend ( artist, label, ax, **kwargs )
    return artist, label