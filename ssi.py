''' A module for dealing with Side Selectivity Index analysis '''

import numpy as np
import analyze as al
from bhv import LEFT, RIGHT, HIT, ERROR, RM, WM
from stats import *
import matplotlib.pyplot as plt
from utils import memoize
from scipy.stats import ttest_1samp, ttest_ind

class unitSSI(object):
    ''' An object for storing a Side Selectivity Index (SSI) calculation
        for a single unit.  SSI is the mean rate for 'right' minus the mean 
        rate for 'left' divided by the sum of the mean rates.
        
        Arguments
        ---------
        unit : catalog unit
            Unit for this unitSSI object
        unit_data : pandas DataFrame 
            Timelocked data for this unit, from Timelock.get(unit)
        goal : string
            'PG' or 'FG',
        iters : int
            Number of bootstrap iterations
        
        Attributes
        ----------
        self.wm : np.array
            Working memory SSIs
            A structured numpy array with fields 'mean', 'CI', 'p', holding
            means, confidence intervals (bias-corrected and accelerated),
            and p-values found from a permutation test, respectively.  The
            rows are indexed by intervals, found in intervals attribute.
        self.rm : np.array
            Reference memory SSIs
            A structured numpy array with fields 'mean', 'CI', 'p', holding
            means, confidence intervals (bias-corrected and accelerated),
            and p-values found from a permutation test, respectively.  The
            rows are indexed by intervals, found in intervals attribute.
    '''
    
    def __init__(self, unit, unitData, goal = 'PG', iters = 1000, use_times=False):
        
        self.iters = iters
        self.unit = unit
        self.goal = goal
        self._use_times = use_times
        self.intervals = ['PG port','Early','Middle','Late','C port','C to FG']
        # In this code, wm means working memory and rm means reference memory
        self.wm, self.rm = self._compute_ssis(unitData)
        
    
    def _compute_ssis(self, unitData):
        ''' Computes SSIs from bootstrapping, along with confidence intervals.
            Also gives p-values found from a permutation test.
        '''
        intervals = self.intervals
        dtypes = [ ('mean','f8'), ('CI', 'f8', 2), ('p', 'f8') ]
        mems = {'wm':np.zeros(len(intervals), dtype = dtypes),
                'rm':np.zeros(len(intervals), dtype = dtypes)}
        
        data = al.get_memory_sides(unitData, goal=self.goal)
        for mem in ['wm','rm']:
            left, ldurs = al.slice(self.unit, data[(mem,'left')], times=True)
            right, rdurs = al.slice(self.unit, data[(mem,'right')], times=True)
            if not self._use_times:
                ldata, rdata = left, right
            else:
                ldata, rdata = ldurs, rdurs
            for ii, interval in enumerate(intervals):
                prepared = prepare_memory_sides(ldata[interval], rdata[interval])
                sampled = Bootstrapper(prepared, ssi, iters = self.iters)
                mems[mem][ii]['mean'] = sampled.mean()
                mems[mem][ii]['CI'][:] = sampled.bcaconf()
                mems[mem][ii]['p'] = permutation_test(prepared, sampled.mean(), 
                                                      iters = self.iters)
                                                      
        return mems['wm'], mems['rm']
    
    def __repr__(self):
    
        return "unitSSI(id = %d, iters = %d, goal = %s)" \
            % (self.unit.id, self.iters, self.goal)
    
    def __getitem__(self, key):
        
        if isinstance(key, basestring):
            pass
        else:
            raise KeyError("Key must be a string")
        if key in self._intervals:
            pass
        else:
            raise KeyError("%s is not a valid interval" % key)
        
        index = self.intervals.index(key)
        return {'wm':self.wm[index],'rm':self.rm[index]}

    def __reduce__(self):
        return "unitSSI"
                   
class listSSI(list):
    ''' Okay, going to add memorySSIs to this so I can contain a batch of units
        and their SSIs, then analyze all of them '''
        
    def __init__(self, *args, **kwargs):
        
        super(listSSI, self).__init__(*args)
        
        if 'auto' in kwargs:
            auto = bool(kwargs['auto'])
        else:
            auto = False
        self.autoupdate = auto
        
        if len(self):
            self.autoupdate = True
            self._update()
                 
    def _update(self):
        
        if not self.autoupdate:
            return None
        
        if hasattr(self, 'intervals'):
            pass
        elif hasattr(self[:][0], 'intervals'):
            self.intervals = self[:][0].intervals
        
        self.ids = [ memSSI.unit.id for memSSI in self ]
        self.units = np.array([ memSSI.unit for memSSI in self ])
        self.wm = self._mem_helper('wm')
        self.rm = self._mem_helper('rm')
    
    def update(self):
        save = self.autoupdate
        self.autoupdate = True
        self._update()
        self.autoupdate = save
    
    def _mem_helper(self, memory):
        ''' Calculates a dictionary of arrays storing means, CIs, and p-values
            for each unit, for each interval.  Right now, I loop over all the
            units in the list to get the means and such.  I don't think this 
            part is slow at all, but if it ever is, I should change it so that
            it just updates with the recently added/removed units.
        '''
        
        means = np.array([ getattr(ussi, memory)['mean'] for ussi in self ]).T
        cis = np.concatenate([ getattr(ussi, memory)['CI'] for ussi in self ], axis=1)
        cis = cis.reshape((len(self.intervals), len(self), 2))
        ps = np.array([ getattr(ussi,memory)['p'] for ussi in self ]).T
        
        return {'mean':means, 'CI':cis, 'p':ps}
        
    def append(self, *args):
        super(listSSI, self).append(*args)
        self._update()
    
    def extend(self, *args):
        super(listSSI, self).extend(*args)
        self._update()
    
    def pop(self, *args):
        super(listSSI, self).pop(*args)
        self._update()  
    
    def remove(self, *args):
        super(listSSI, self).remove(*args)
        self._update()
    
    def insert(self, args):
        super(listSSI, self).insert(*args)
        self._update()
    
    def __add__(self, other):
        # self + other
        self.extend(other)
    
    def __sub__(self, other):
        # self - other
        self.remove(other)
            
    def __reduce__(self):
        return 'listSSI'
        
def prepare_memory_sides(left, right):
    
    sides = np.concatenate([np.ones(len(left))*LEFT,
                           np.ones(len(right))*RIGHT])
    rates = np.concatenate([left, right])
    return prepare(rates,sides)
    
def prepare(rates, sides):
    ''' Prepare data for the ssi function 
    
    Arguments
    ---------
    rates : 1D array-like
        Spiking rate for each trial
    sides : 1D array-like
        Response side for each trial, must be same length as rates
        Sides should be values given by the constants from the behavior data
        
    Returns
    -------
    data : numpy structured array
        A structured array with fields 'rates' and 'sides'
        
    '''
    
    if len(rates)==len(sides):
        pass
    else:
        raise ValueError('rates and sides must be the same length')
    
    records = ['rates', 'sides']
    dtypes = [ rates.dtype, sides.dtype ]
    data = np.zeros(len(rates), dtype = zip(records, dtypes))
    data['rates'] = rates
    data['sides'] = sides
    
    return data

def ssi(data):
    ''' Calculates the Side Selectivity Index (SSI).
    
    Arguments
    ---------
    data: numpy structured array
        Must have two fields, 'rates' and 'sides'
        Can get this from prepare function
        
    Returns
    -------
    ssi : float
        The SSI for the data given.  Positive values are selective for RIGHT,
        negative values are selective for LEFT.
    '''
    
    left_mean = data[data['sides']==LEFT]['rates'].mean()
    right_mean = data[data['sides']==RIGHT]['rates'].mean()
    bad = np.any([left_mean + right_mean==0, 
                    np.isnan(left_mean).any(),
                    np.isnan(right_mean).any()])
   
    if not bad:
        ssi = (right_mean - left_mean) / (left_mean + right_mean)
        return ssi
    else:
        return 0
 
def permutation_test(ssi_data, observed, iters = 1000):
    ''' Calculates the p-value of an observed bootstrapped
        statistic using the permutation test.  More specifically,
        it randomly permutes the sides of the rates-sides data, then
        calculated the ssi for the permuted data.  Do this iters times
        and you have a null distribution.  Then, find the p-value of the 
        observed statistic from the null distribution.
    '''
    permute = np.random.permutation
    def nulldata():
        null = ssi_data.copy()
        null['sides'] = permute(ssi_data['sides'])
        return null
    null_dist = np.array([ ssi(nulldata()) for ii in xrange(iters)])
    p = np.sum(null_dist>observed)/float(len(null_dist))
    if observed < 0:
        p = 1 - p
    return p

def ssi_scatter(listSSI, interval):
    ''' Creates a cross scatter plot for SSI values for each unit in the
        desired interval. Data points are colored for significance. Red points
        are significant for the x-axis SSI, blue points are significant for the
        y-axis SSI, purple are both, grey is neither.
        
        Arguments
        ---------
        listSSI : ssi.listSSI
            List of unitSSIs you want to plot
        interval : string
            The time interval you want to plot
            'PG port', 'Early', 'Middle', 'Late', 'C port', 'C to FG'
        
        Returns
        -------
        sig : dict
            Dictionary of significant units.  
            Keys are 'x', 'y', 'both', 'neither'
            'x' gives you the indices of SSIs significant on the x-axis
            'y' gives you the indices of SSIs significant on the y-axis
            'both' gives you the indices of SSIs significant on both axes
            'neither' gives you the indices of not significant SSIs
    
    '''
    ssis = listSSI
    ps = np.concatenate((ssis.wm['p'].flatten(),
                         ssis.rm['p'].flatten()))
    FDR_level = constrain_FDR(ps)
    ii = ssis.intervals.index(interval)
    sigs, ax = al.cross_scatter(ssis.wm['mean'][ii], 
                               ssis.rm['mean'][ii], 
                               ssis.wm['CI'][ii].T, 
                               ssis.rm['CI'][ii].T,
                               ssis.wm['p'][ii],
                               ssis.rm['p'][ii],
                               sig_level = FDR_level)
    
    ax.set_title(interval)
    ax.set_xlabel('WM SSI')
    ax.set_ylabel('RM SSI')
    return sigs, ax

def ssi_grid(*args, **kwargs):
    ''' Creates a plot that shows the SSIs of each unit over time intervals.
        Data points are colored for significance. Red points
        are significant for the x-axis SSI, blue points are significant for the
        y-axis SSI, purple are both, grey is neither.
        
        Arguments
        ---------
        listSSI, or iterable of listSSI objects
        
        Keywords
        --------
        title : string, or iterable of strings
            Title of grid plot, or an iterable of strings the same length
            as the passed listSSI iterable
    '''
    n_plots = len(args)
    fig = plt.figure(figsize=(5,10))
    colors = {'x':'r', 'y':'b', 'both':'purple', 'neither':'grey'}
    axes = [ fig.add_subplot(1,n_plots,ii) for ii in range(n_plots) ]
    
    # Writing it this way so that you can pass in multiple listSSIs and it'll
    # make a grid plot for each of them.
    def grid(arg, ax):
        ssis = arg
        intervals = ssis.intervals
        xs = np.arange(0,len(intervals)*2, 2)
        ys = np.arange(0,len(ssis)*2, 2)
        XX, YY = np.meshgrid(xs, ys)
        SX = XX.copy().astype(float)
        SY = YY.copy().astype(float)
        sig_mat = np.zeros(np.shape(XX), dtype='a6')
        
        ps = np.concatenate((ssis.wm['p'].flatten(),
                             ssis.rm['p'].flatten()))
        FDR_level = constrain_FDR(ps)
        sigs = dict.fromkeys(intervals)
        for interval in intervals:
            ii = ssis.intervals.index(interval)
            sigs[interval] = al.p_value_sig(ssis.wm['p'][ii],
                                            ssis.rm['p'][ii],
                                            sig_p = FDR_level)
        
        for ii, unit in enumerate(ssis):
            SX[ii,:] = XX[ii,:]+unit.wm['mean']
            SY[ii,:] = YY[ii,:]+unit.rm['mean']
            
        for ii, interval in enumerate(intervals):
            for sig in colors.iterkeys():
                try:
                    sig_mat[:,ii][sigs[interval][sig]] = colors[sig]
                except IndexError:
                    pass
        ax.plot(XX,YY,'--', color = 'grey')
        ax.plot(XX.T,YY.T,'--', color = 'grey')
        ax.plot(SX.T,SY.T, '-k')
        ax.scatter(SX,SY, c = sig_mat.flatten(), s=35)
        ax.set_ylim(-2,np.max(ys)+2)
        ax.set_xticks(range(0,len(intervals)*2,2))
        ax.set_yticks(range(0,len(ssis)*2,2))
        ax.set_xticklabels(intervals, rotation=90, size='large')
        ax.set_yticklabels(ssis.ids, size = 'large')
        ax.set_aspect('equal')
    
    for ii, (arg, ax) in enumerate(zip(args, axes)):
        grid(arg, ax)
        ax.set_title(kwargs['title'][ii])
    if len(args)>1:
        axes[1].set_ylabel('Unit id', size='large')
        axes[0].set_yticklabels('')
    else:
        axes[0].set_ylabel('Unit id', size='large')
    
    fig.tight_layout()
    
    return axes