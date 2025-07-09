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

def add_to_legend ( artist, label, ax, **kwargs ):
    # Append custom elements to the handles and labels
    handles, labels = ax.get_legend_handles_labels()
    
    handles.append(artist)
    labels.append(label)
    
    # Recreate the legend with the updated handles and labels
    oglhl = ax.get_legend_handles_labels
    def monkeypatch ():
        auto_handles, auto_labels = oglhl()
        new_handles = list(dict.fromkeys(handles + auto_handles))
        new_labels = list(dict.fromkeys(labels + auto_labels))
        return new_handles, new_labels   
    ax.get_legend_handles_labels = monkeypatch

    original_legend = ax.legend
    def monkeypatch_legend ( handles=None, labels=None, **kwargs):
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        original_legend(handles, labels, **kwargs)
    
    ax.legend = monkeypatch_legend
    
    ax.legend()#handles, labels)

    return ax

def make_artist_bounded_estimate ( color='tab:red', modulations = [0.4, 0.25], lw=3, **kwargs ):
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
    lines.append(Line2D([], [], linewidth=lw*1.3, dashes=(100,1), color='w', **kwargs),  )
    lines.append(Line2D([], [], linewidth=lw, dashes=(100,1), color=color.base, **kwargs),  )
        
    artist = tuple(lines)
    #handles, labels = add_to_legend ( artist, label, ax, **kwargs )
    return artist


def make_artist_shades ( color='tab:red', modulations = [0.4, 0.25], lw=3, **kwargs ):
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
    #lines.append(Line2D([], [], linewidth=lw, dashes=(100,1), color=color.base),  )
        
    artist = tuple(lines)
    #handles, labels = add_to_legend ( artist, label, ax, **kwargs )
    return artist
