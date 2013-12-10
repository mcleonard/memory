''' A module with a bunch of different utility functions. '''

import numpy as np
from functools import wraps
import matplotlib.pyplot as plt   
import memory as my 

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
