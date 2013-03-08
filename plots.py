''' Module containing functions to generate figures '''

import numpy as np
import matplotlib.pyplot as plt
import memory as my
from .constants import LEFT, RIGHT, WM, RM, HIT, ERROR

def raster(unit, trialData, events = ['PG in', 'FG in'], 
           axis = None, sort_by= 'PG in', prange = None):
    """ A function that creates a raster plot from the units' trial data 
    
    Arguments
    ---------
    unit : catalog Unit
    
    trialData : pandas DataFrame
        The data returned from Timelock.get(unit)
        
    events : list of str
        Plot events of the trial, for instance 'PG in', 'PG out', 'onset', etc.
        Must be valid columns of trialData.
    
    axis : matplotlib axes
        axes object you want to plot the raster on.  Leave as None to create
        a new figure
    
    sort_by : string
        Column in trialData that the trials with be sorted by
    
    prange : tuple of size 2
        Plots from time prange[0] to time prange[1] (in seconds)
        
    Returns
    -------
    ax : matplotlib axes
        Axes the raster is plotted on
    """
    
    # Preparing the data
    srtdata = trialData.sort(sort_by, ascending = False)
    spike_times = srtdata[unit.id]
    cat = np.concatenate # Just for brevity
    spikes = cat([ times for times in spike_times ])
    ys = cat([np.ones(len(stamps))*ii for ii, stamps in enumerate(spike_times)])
    
    events_dict = {event:srtdata[event] for event in events}
    event_colors = {'PG in':'r', 'FG in':'b', 'PG out':'orange', 'onset':'c',
                    'C in':'m', 'C out':'g'}
    
    # Plotting the data
    if axis == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = axis
    ax.scatter(spikes, ys, c = 'k', marker = '|', s = 5) 
    for event, data in events_dict.iteritems():  # Plot the event times
        ax.scatter(data, np.arange(0,len(data)), c=event_colors[event],
                   edgecolor = 'none', marker='o')
    ax.plot([0,0], [0,len(srtdata)], color = 'grey') 
    if prange == None:
        ax.set_xlim(srtdata['PG in'].mean()-3, srtdata['FG in'].mean()+1)
    else:
        ax.set_xlim(prange)
    ax.set_ylim(0,len(data))
    plt.show()
    
    return ax
    
def basic_figs(unit, trialData, prange=(-10,3), smooth = False):
    from itertools import izip
    
    def base_plot(xs, rates, ax, labels = None, title = None):
        
        if labels == None:
            labs = ['']*2
        else:
            labs = labels
        for rate, lab in izip(rates, labs):
            ax.plot(xs, rate, label = lab)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Activity (s)')
        if title != None: ax.set_title(title)
        if labels != None: ax.legend(loc='best')
        ax.plot([0,0], ax.get_ylim(), '-', color = 'grey')
    
    data = trialData
    if not smooth:
        rates = my.analyze.ratehist(unit, data, prange = prange)
    else:
        rates = my.analyze.smooth(unit, data, prange = prange)
    time = rates.columns.values 
    # First plot all data
    ax1 = plt.subplot2grid((2,2), (0,0))
    base_plot(time, [rates.mean()], ax1, labels = ['All trials'])
    
    # Plot PG left vs PG right
    pg = data.groupby(by='PG response')
    pgrates = [rates.ix[values.index].mean() for group, values in pg ]
    ax2 = plt.subplot2grid((2,2), (0,1))
    base_plot(time, pgrates, ax2, labels = ['PG left', 'PG right'])
    
    # Plot FG left vs FG right
    fg = data.groupby(by='FG response')
    fgrates = [rates.ix[values.index].mean() for group, values in fg ]
    ax3 = plt.subplot2grid((2,2), (1,0))
    base_plot(time, fgrates, ax3, labels = ['FG left', 'FG right'])
    
    # Plot cued vs uncued
    mem = data.groupby(by='block')
    memrates = [rates.ix[values.index].mean() for group, values in mem ]
    ax4 = plt.subplot2grid((2,2), (1,1))
    base_plot(time, memrates, ax4, labels = ['cued', 'uncued'])
    
    plt.show()
    fig = plt.gca().figure
    fig.tight_layout()
    
    return fig

def confidence_sig(xerr, yerr):
    ''' Returns a dictionary indicating significant data points, significance
        being defined as 0 falling outside the confidence interval.

        Arguments
        ---------
        xerr : 2d array
            Lower and upper confidence intervals
        yerr : 2d array
            Lower and upper confidence intervals
        
        Returns
        -------
        sigs : dict
            Keys => 'x', 'y', 'both', 'neither'
            Values => indices of significant intervals
    
    '''

    # If the error bounds are both negative or both positive, then 0 is not 
    # between the error bounds. So multiplying the bounds gives a positive 
    # number if significant.
    
    sigs = {'x':np.where((xerr.prod(axis=0)>0)&(yerr.prod(axis=0)<0))[0],
            'y':np.where((xerr.prod(axis=0)<0)&(yerr.prod(axis=0)>0))[0],
            'both':np.where((xerr.prod(axis=0)>0)&(yerr.prod(axis=0)>0))[0],
            'neither':np.where((xerr.prod(axis=0)<0)&(yerr.prod(axis=0)<0))[0]}
    
    return sigs
    
def p_value_sig(p_x, p_y, sig_p = 0.05):
    ''' Returns a dictionary indicating significant data points, significance
        being defined as 0 falling outside the confidence interval.

        Arguments
        ---------
        p_x : 2d array
            p-values for the x dimension
        p_y : 2d array
            p-values for the y dimension
        
        Returns
        -------
        sigs : dict
            Keys => 'x', 'y', 'both', 'neither'
            Values => indices of significant p-values
    
    '''
    sigs = {'x':np.where((p_x<=sig_p)&(p_y>sig_p))[0],
            'y':np.where((p_y<=sig_p)&(p_x>sig_p))[0],
            'both':np.where((p_x<=sig_p)&(p_y<=sig_p))[0],
            'neither':np.where((p_x>sig_p)&(p_y>sig_p))[0]}
    return sigs

def cross_scatter(x, y, xerr, yerr, p1 = None, p2 = None, **kwargs):
    '''  cross_scatter(x, y, xerr, yerr) -> list of bools, makes a figure
    
    Makes a scatter plot with error bars.  Colors data points based on
        significance
    
    Arguments
    ---------
    x : array-like
        An array of the x components of the data
    y : array-like
        An array of the y components of the data
    xerr : 2D array
        First row is the lower error bar, second row in the upper error bar
    yerr : 2D array
        First row is the lower error bar, second row in the upper error bar
    p : array-like
        Array of p-values for each data point. Can be a 2-dimensional 
        list/tuple/array of p-values, e.g. p = (p_x, p_y).
    
    Keywords
    --------
    Hopefully these are obvious
    xlabel, ylabel, xlims, ylims, title, sig_level
        
    Returns
    -------
    sig : list of bools
        Boolean values for the data points that are significant
        Signficance is tested as zero being outside the error bars
    ax : matplotlib.axes.AxesSubplot
        Returns the axes for the scatter plot so you can add labels and such
    '''
    
    valid_kwargs = ['xlabel', 'ylabel', 'xlims', 'ylims', 'title', 'sig_level']
    sig_colors = {'x':'r', 'y':'b', 'both':'purple', 'neither':'grey'}

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    
    # sigs is a dictionary with keys 'x', 'y', 'both', 'neither'
    # The values are the significant data points
    if p1 == None:
        p1, p2 = (np.ones(len(x)), np.ones(len(y)))
        sig_p = 0.01
    elif p2 == None:
        sig_p = my.stats.constrain_FDR(p1)
        sig_p = kwargs.get('sig_level', sig_p)
        p1, p2 = p1.copy(), p1.copy()
        # What I'm doing here is for any points above the unity line can only
        # be colored like 'x' and points below the unity line can only be 
        # colored like 'y', if they are 
        p2[np.where((x-y)<0)] = 1
        p1[np.where((x-y)>0)] = 1
    else:
        sig_p = my.stats.constrain_FDR(np.concatenate([p1,p2]))
        sig_p = kwargs.get('sig_level', sig_p)
    
    sigs = p_value_sig(p1,p2,sig_p)
    
    # Then plot units that are significant for working and reference memory
    xbars = np.abs(xerr - x)
    ybars = np.abs(yerr - y)
    for key in ['neither','both','y','x']:
        sig = sigs[key]
        if len(sig):
            ax.errorbar(x[sig],y[sig],
                        yerr=ybars[:,sig],xerr=xbars[:,sig],
                        fmt='o', color = sig_colors[key])
    
    xlim = kwargs.get('xlims', [-1,1])
    ylim = kwargs.get('ylims', [-1,1])
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    ax.set_autoscale_on(False)
    ax.plot(xlim,[0,0],'-k')
    ax.plot([0,0],ylim,'-k')
    ax.plot([-100,100],[-100,100],'--',color='grey')
    ax.set_aspect('equal')
    
    ax_fns = {'xlabel':ax.set_xlabel, 'ylabel':ax.set_ylabel,
              'title':ax.set_title}
    fns = [ ax_fns[key](value) for key, value in kwargs.iteritems() 
            if key in ax_fns.keys() ]
    
    plt.show()
    
    return sigs, ax