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
    
def catalog_path():
    ''' This function returns the relative path to the default catalog. '''
    from os.path import expanduser, relpath
    absolute = expanduser('~/Dropbox/Data/catalog.sqlite')
    return '/'+relpath(absolute)

def startup():
    ''' This function runs code that I do really often when I'm looking at data.
        I wrote this so I don't have to type it out every time.
        
        Returns
        -------
        catalog, units, Timelock
    '''
    from catalog import Catalog, Unit
    from os.path import expanduser
    
    catalog = Catalog(catalog_path())
    units = catalog.query(Unit) \
                   .filter(Unit.rate >= 0.5) \
                   .filter(Unit.cluster != 0) \
                   .all()
    #good_units = [13,14,16,20,25,26,40,50,53,55,58,60,61,62,64,66,67]
    #units = [ catalog[id] for id in good_units ]
    lkr = my.Timelock(units)
    data = lkr.lock('onset').get(units[0])
    return catalog, units, lkr
