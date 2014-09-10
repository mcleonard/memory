''' Module containing functions to generate figures '''

from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import analyze
import stats
from constants import *

def passed_or_new_ax(func):
    """ This is a decorator for the plots in this module.  Most plots can be
        passed an existing axis to plot on.  If it isn't passed an axis, then
        a new figure and new axis should be generated and passed to the plot.
        This decorator ensures this happens.
    """
    @wraps(func)
    def inner(*args, **kwargs):
        if 'ax' in kwargs:
            return func(*args, **kwargs)
        else:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', None))
            kwargs.update({'ax':ax})
        return func(*args, **kwargs)
    return inner

def differences(units, lkr, intervals, goal, mem=WM, iters=2000):
    """ Creates a plot showing the firing rate difference between right and left trials.  Also
        plots histograms for the rate differences.
        
        Arguments
        ---------
        units : list of catalog units
        lkr : Timelocker
        intervals : list of str, intervals you want plotted, ex. ['Early','Middle','Late']
        goal : 'PG' or 'FG' (past goal or future goal)
        iters : iterations to run for bootstrapping and permutation tests
    """
    from memory.analyze import bootstrap_difference
    from memory.stats import permutation_test, null_difference, constrain_FDR
    
    # This determines the size of the subplot grid
    ROWS = 4
    COLUMNS = len(intervals)
    GRID = (ROWS,COLUMNS)
    
    dtypes = [('pref','i8'), ('mean','f8'), ('CI','f8',2),('p','f8')]
    
    for ii, interval in enumerate(intervals):
        dataFrames = ( lkr.get(unit) for unit in units )
        calc = np.zeros(len(units), dtype = dtypes)
        for jj, data in enumerate(dataFrames):
            calc[jj] = ( data.outcomes(HIT, HIT)
                             .blocks(mem)
                             .preferred_side(goal, interval, iters) )

        # This part is for making the rate differences plot
        ax = plt.subplot2grid(GRID, (0,ii), rowspan = ROWS-1)
        ax, sigs = difference_plot(ax, calc['mean'], calc['CI'].T, calc['p'])
        
        ax.set_yticks(range(0,len(units)))
        ax.set_xticklabels('')
        ax.set_title(interval)
        ax.set_yticklabels([str(unit.id) for unit in units])
        # Get the xlims of this plot to use in the histogram plot
        xl = ax.get_xlim()
        
        # And this part is for the histograms
        ax = plt.subplot2grid(GRID, (ROWS-1,ii))
        ax, _ = sig_histogram(ax, calc['mean'], calc['p'], xlims = xl)
        
        ax.figure.suptitle(goal, size = 'x-large')
    
    return ax, sigs

def difference_plot(ax, values, errs, ps):
    
    sigs, notsigs = sig_ps(ps)
    bars = abs(errs-values)
    ax.errorbar(np.zeros(len(sigs))+values[sigs], sigs,
                xerr=bars[:,sigs], fmt='o', color='k')
    ax.errorbar(np.zeros(len(notsigs))+values[notsigs], 
                notsigs, xerr=bars[:,notsigs], fmt='o', color='grey')
    ax.set_ylim(-1,len(values))
    yl = ax.get_ylim()
    xl = ax.get_xlim()
    ax.plot([0,0],yl,'--', color = 'grey')
    
    return ax, sigs

def sig_histogram(ax, values, ps, xlims = None, bin_width = 0.2):
    
    sigs, notsigs = sig_ps(ps)
    
    if xlims == None:
        xl = (values.min()-1, values.max()-1)
    else:
        xl = xlims
    bins = int((xl[1]-xl[0])/bin_width)
    ax.hist(values[notsigs], bins=bins, range=xl, facecolor = 'w', edgecolor = 'k')
    ax.hist(values[sigs], bins=bins, range=xl, color = 'k')
    
    return ax, sigs

def sig_ps(ps):
    
    p_level = my.stats.constrain_FDR(ps)
    sigs = np.where(ps<=p_level)[0]
    notsigs = np.where(ps>p_level)[0]
    
    return sigs, notsigs
    
def add_sig_colors(x, y, ax, p):
    """ Adds a legend showing significance between bars
        
        Arguments
        ---------
        x, y: ints, location of the upper left corner of the legend
        ax : axis the legend belongs to
        p : p-value to show on the legend
    """
    
    # Setting the height and width as a ratio so they are the same size
    # regardless of the plot scale
    rect_h = np.diff(ax.get_ylim())[0]/20.0
    rect_w = np.diff(ax.get_xlim())[0]/10.0
    
    # The blue bar
    rect = plt.Rectangle((x,y),rect_w,rect_h,color='steelblue')
    ax.add_artist(rect)
    rect.set_clip_box(ax.bbox)
    
    # The orange bar
    rect = plt.Rectangle((x+0.5,y),rect_w,rect_h,color='orangered')
    ax.add_artist(rect)
    rect.set_clip_box(ax.bbox)
    
    # This part indicates the p-value
    line = plt.Line2D([x+1.2*rect_w,x+2.3*rect_w], [y+0.5*rect_h,y+0.5*rect_h], color = 'k', lw = 2)
    ax.add_artist(line)
    line.set_clip_box(ax.bbox)
    ptext = plt.Text(x=x+0.25*rect_w, y = y+1.5*rect_h, text='p = {:01.3f}'.format(p), size='large')
    ax.add_artist(ptext)
    #aster.set_clip_box(ax.bbox)

def trajectories(trialData, interval, mem, legend_loc=(0.5,2.0)):
    """ Plots bars showing mean rates for the four trajectories.  Colors the
        bars to show signficance.  Significance is found from a linear model
        and ANOVA, which are also plotted
        
        Arguments
        ---------
        unit : catalog unit
        unitData : pandas DataFrame, get from lkr.get(unit) for instance
        interval : interval you want to analyze ('Early', 'Middle', or 'Late' for instance)
        mem : my.WM or my.RM for working memory or reference memory, respectively
    """
    from memory.stats import Bootstrapper
    from pandas import DataFrame
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    
    # This is all data calculations
    data = trialData.groupby(by=['block', 'PG outcome', 'FG outcome']).get_group((mem, my.HIT, my.HIT))
    rates = my.slice(data)
    df = DataFrame([data['PG response'], data['FG response'], rates[interval]]).T
    df.columns = ['PG','FG','rate']
    sides = df.groupby(by=['PG','FG'])
    
    # Now we're going to bootstrap our rates, for each PG-FG trajectory
    booted = { group:Bootstrapper(values['rate'].values, np.mean, iters = 2000)
               for group, values in sides  }
    means = [ booted[side].mean() for side,_ in sides ]
    CIs = [ booted[side].prctile() for side,_ in sides ]
    trajectories = [ side for side,_ in sides ]
    
    # And fit a linear model, run ANOVA to test for significant parameters
    ls = ols('rate ~ PG + FG', df).fit()
    anova = anova_lm(ls, type=3)
    lin_fit = [ ls.params['PG']*pg + ls.params['FG']*fg + ls.params['Intercept']
                for (pg, fg), val in sides ]

    sig_p = 0.05
    if anova['PR(>F)']['PG'] < sig_p:
        bar_colors = ['steelblue', 'steelblue', 'orangered', 'orangered']
        p = anova['PR(>F)']['PG']
    elif anova['PR(>F)']['FG'] < sig_p:
        bar_colors = ['steelblue', 'orangered', 'steelblue', 'orangered']
        p = anova['PR(>F)']['FG']
    else:
        if anova['PR(>F)']['PG'] < anova['PR(>F)']['FG']:
            bar_colors = ['steelblue', 'steelblue', 'orangered', 'orangered']
        else:
            bar_colors = ['steelblue', 'orangered', 'steelblue', 'orangered']
        p = min(anova['PR(>F)']['PG'], anova['PR(>F)']['FG'])
    
    # Now we'll do the actual plotting
    fig = plt.figure(figsize=(4,5))
    ax = fig.add_subplot(111)
    xs = np.arange(0,2,0.5)
    y = np.array(means)
    ybars = abs(np.array(CIs).T - y)
    plt.bar(xs-0.20, y, width = 0.4, color = bar_colors, edgecolor='none')
    plt.errorbar(xs, y, yerr=ybars, fmt='o', color = 'k', lw=2, ms = 5)
    plt.plot(xs, lin_fit, '--k')
    traj_map = {(my.LEFT,my.LEFT):'left, left', (my.LEFT,my.RIGHT):'left, right',
                (my.RIGHT,my.RIGHT):'right, right', (my.RIGHT,my.LEFT):'right, left'}
    plt.ylabel('Activity (spikes/s)', size='large')
    plt.yticks(size='large')
    xtlabels = [ traj_map[traj] for traj in trajectories ]
    plt.xticks(xs, xtlabels, rotation=45, ha='right', size='large')
    plt.xlim(-0.25,1.75)
    yl = plt.ylim()
    add_sig_colors(legend_loc[0], legend_loc[1], ax, p)
    fig.tight_layout()
    return ax

def raster(trialData, events=['PG in', 'FG in'], markersize=10,
           ax=None, sort_by='PG in', limit = None, figsize=(8,7)):
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
    
    limit : tuple of size 2
        Plots from time limit[0] to time limit[1] (in seconds)
        
    Returns
    -------
    ax : matplotlib axes
        Axes the raster is plotted on
    """
    
    # Preparing the data
    srtdata = trialData.sort(sort_by, ascending = False)
    spike_times = srtdata['timestamps']
    cat = np.concatenate # Just for brevity
    spikes = cat([ times for times in spike_times ])
    ys = cat([np.ones(len(stamps))*ii for ii, stamps in enumerate(spike_times)])
    
    events_dict = {event:srtdata[event] for event in events}
    event_colors = {'PG in':'r', 'FG in':'b', 'PG out':'orange', 'onset':'c',
                    'C in':'m', 'C out':'g'}
    
    # Plotting the data
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(spikes, ys, c = 'k', marker = '|', s = markersize) 
    for event, data in events_dict.iteritems():  # Plot the event times
        ax.scatter(data, np.arange(0,len(data)), c=event_colors[event],
                   edgecolor = 'none', marker='o')
    ax.plot([0,0], [0,len(srtdata)], color = 'grey') 
    if limit == None:
        ax.set_xlim(srtdata['PG in'].mean()-3, srtdata['FG in'].mean()+1)
    else:
        ax.set_xlim(limit)
    ax.set_ylim(0,len(data))
    #ax.set_ylabel('Trial')
    #ax.set_xlabel('Time (s)')
    ax.grid(False)

    return fig, ax
    
def basic_figs(trial_data, bin_width=0.2, limit=(-10,3), smooth = False, figsize=(8,7)):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    all_peths = analyze.time_histogram(trial_data, bin_width=bin_width, limit=limit)
    if smooth:
        all_peths = smooth(all_peths)
    x = all_peths.columns.values

    titles = ['All', 'Block', 'PG port', 'FG port']

    axes[0,0].plot(x, all_peths.mean(), label='all')

    blocks = trial_data.groupby(by='block')
    axes[0,1].plot(x, all_peths.ix[blocks.get_group(WM).index].mean(), label='uncued')
    axes[0,1].plot(x, all_peths.ix[blocks.get_group(RM).index].mean(), label='cued')

    pg = trial_data.groupby(by='PG port')
    axes[1,0].plot(x, all_peths.ix[pg.get_group(LEFT).index].mean(), label='left')
    axes[1,0].plot(x, all_peths.ix[pg.get_group(RIGHT).index].mean(), label='right')

    fg = trial_data.groupby(by='FG port')
    axes[1,1].plot(x, all_peths.ix[fg.get_group(LEFT).index].mean(), label='left')
    axes[1,1].plot(x, all_peths.ix[fg.get_group(RIGHT).index].mean(), label='right')

    for ax, title in zip(axes.flatten(), titles):
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Rate (spikes/s)')
        ax.legend(loc='best')
        
    fig.tight_layout()

def trajectory_peths(trial_data, limit=(-2,1), bin_width=0.05, figsize=(7,6)):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    title_map = { LEFT:'left', RIGHT:'right' }

    grouped = trial_data.groupby(['PG port', 'FG port'])
    for ax, (group, values) in zip(axes.flatten(), grouped):
        blocks = values.groupby('block')
        peths = analyze.time_histogram(values, limit=limit, bin_width=bin_width)
        x = peths.columns.values
        ax.plot(x, peths.ix[blocks.get_group(WM).index].mean(), label='uncued')
        ax.plot(x, peths.ix[blocks.get_group(RM).index].mean(), label='cued')
        ax.set_title('{} to {}'.format(title_map[group[0]], title_map[group[1]]))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Rate (spikes/s)')
        ax.legend(loc='best')
        
    fig.tight_layout()

    return fig, axes

def trajectory_rasters(trial_data, limit=(-2,1), figsize=(7,6), marker='|', markersize=30):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    title_map = { LEFT:'left', RIGHT:'right' }

    grouped = trial_data.groupby(['PG port', 'FG port'])
    for ax, (group, values) in zip(axes.flatten(), grouped):
        blocks = values.groupby('block')
        wm = blocks.get_group(WM)
        rm = blocks.get_group(RM)
        
        wm_spikes = np.concatenate([ times for times in wm['timestamps'] ])
        rm_spikes = np.concatenate([ times for times in rm['timestamps'] ])
        
        wm_ys = np.concatenate([np.ones(len(stamps))*ii 
                                for ii, stamps in enumerate(wm['timestamps'])])
        rm_ys = np.concatenate([np.ones(len(stamps))*ii 
                                for ii, stamps in enumerate(rm['timestamps'])])
        rm_ys = rm_ys + len(wm)
        
        ax.scatter(wm_spikes, wm_ys, marker=marker, color='k', s=markersize)
        ax.scatter(rm_spikes, rm_ys, marker=marker, color=(.3,.3,.3), s=markersize)
        ax.set_title('{} to {}'.format(title_map[group[0]], title_map[group[1]]))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Trials')
        ax.set_xlim(*limit)
        
    fig.tight_layout()

    return fig, axes

@passed_or_new_ax
def ordered_unit_array(df, ax=None, figsize=(6,5), aspect='auto', cm=plt.cm.RdBu_r):

    #fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(df, cmap=cm, aspect=aspect, interpolation='nearest')
    xticks = range(0,len(df.columns), len(df)/6)
    ax.set_xticks(xticks)
    ax.set_xticklabels(df.columns.values[xticks].astype(float).round(2))
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels('')
    ax.set_ylabel('Units')
    ax.set_xlabel('Time (seconds)')
    delta = np.diff(df.columns.values).astype(float).mean()
    zero = (0.0-df.columns.values[0]/delta).round(1) + 0.5
    ax.plot([zero]*2, [-0.5,len(df)-0.5], '-', color='k')
    ax.grid(False)
    ax.tick_params(axis='y', length=0)
    
    return img, ax

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
    if p1 is None:
        p1, p2 = (np.ones(len(x)), np.ones(len(y)))
        sig_p = 0.01
    elif p2 is None:
        sig_p = stats.constrain_FDR(p1)
        sig_p = kwargs.get('sig_level', sig_p)
        p1, p2 = p1.copy(), p1.copy()
        # What I'm doing here is for any points above the unity line can only
        # be colored like 'x' and points below the unity line can only be 
        # colored like 'y', if they are 
        p2[np.where((x-y)<0)] = 1
        p1[np.where((x-y)>0)] = 1
    else:
        sig_p = stats.constrain_FDR(np.concatenate([p1,p2]))
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