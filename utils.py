''' A module with a bunch of different utility functions. '''

import numpy as np
from functools import wraps
import matplotlib.pyplot as plt   
import memory as my 
from matplotlib.ticker import MaxNLocator

def line_hist(data, **kwargs):
    ''' Plot a histogram as a line plot instead of a bar plot.  Accepts any keyword 
        arguments that numpy.histogram and pyplot.plot will.
    '''
    
    # kwargs here is a little complicated because they can be either for 
    # np.histogram or for plt.plot.  So, I'll pop the histogram ones first, pass
    # them to the function, and send the rest of the kwargs to plt.plot.
    histo_keys = ['bins', 'range', 'normed', 'weights', 'density']
    histo_kwargs = {key:kwargs.pop(key) for key in histo_keys if kwargs.has_key(key)}
    counts, bins = np.histogram(data, **histo_kwargs)
    
    xs = bins[:-1]
    plt.plot(xs, counts, **kwargs)
    return counts, xs

def memoize(func):
    cache = {}

    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper
    
def filepath_from_dir(dir, ext):
    ''' Returns the absolute path of a file with extension ext in the
        directory dir '''
    
    import os
    import re
    dirpath = os.path.expanduser(dir)
    filelist = os.listdir(dirpath)
    found = re.findall( '\S+.{}\S*'.format(ext), ' '.join(filelist))
    filepath = [ os.path.join(dirpath, file) for file in found ]
    return filepath

def catalog_path(absolute = None):
    ''' This function returns the relative path to the default catalog. '''
    from os.path import expanduser, relpath
    if absolute == None:
        absolute = expanduser('~/Dropbox/Data/catalog.sqlite')
    return '/'+relpath(absolute)

def startup(exclude=None):
    ''' This function runs code that I do really often when I'm looking at data.
        I wrote this so I don't have to type it out every time.
        
        Returns
        -------
        catalog, units, Timelock
    '''
    from spikesort.catalog import Catalog
    from os.path import expanduser
    
    catalog = Catalog(catalog_path())
    units = catalog.query(Unit) \
                   .filter(Unit.rate >= 0.5) \
                   .filter(Unit.cluster != 0) \
                   .all()
    if exclude:
        for unit_id in exclude:
            units.remove(catalog[unit_id])
        
    lkr = my.Timelock(units)
    lkr.lock('onset')
    return catalog, units, lkr

def pub_format(ax, legend=None, spine_width=1.3, spine_color='#111111', 
               hide_spines=['top', 'right'], yticks=6, xticks=5):
    """ Format plots for publication. """
    if hide_spines is None:
        pass
    else:
        for each in hide_spines:
            ax.spines[each].set_color('w')
    ax.set_axis_bgcolor('w')
    
    shown_spines = [each for each in ['top', 'bottom', 'left', 'right'] if each not in hide_spines]
    for each in shown_spines:
        ax.spines[each].set_linewidth(spine_width)
        ax.spines[each].set_color(spine_color)
    
    ax.yaxis.set_major_locator(MaxNLocator(yticks))
    ax.xaxis.set_major_locator(MaxNLocator(xticks))
    ax.xaxis.set_minor_locator(MaxNLocator(xticks*4))
    ax.yaxis.set_minor_locator(MaxNLocator(yticks*2))
    
    ax.tick_params(axis='both', which='minor', length=3, width=1.3, 
                   direction='out', colors='#111111')
    ax.tick_params(which='both', **dict(zip(hide_spines, ['off']*len(hide_spines))))
    ax.tick_params(direction='out', width=spine_width, color=spine_color)
    ax.grid(which='both', b=False)

    if legend:
        leg_frame = legend.get_frame()
        leg_frame.set_facecolor('w')
        leg_frame.set_edgecolor('none')
    
    return ax