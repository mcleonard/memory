''' Module containing functions to generate figures '''

import numpy as np
import matplotlib.pyplot as plt
import memory as my
from .constants import LEFT, RIGHT, WM, RM, HIT, ERROR

def differences(units, lkr, intervals, goal, iters = 2000):
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
    GRID = (len(intervals),3)
    
    for ii, interval in enumerate(intervals):
        ps=np.ones(len(units))
        diffs = [ bootstrap_difference(lkr.get(unit), interval, my.WM, 
                                       goal, iters = iters) 
                                       for unit in units ]
        for jj, diff in enumerate(diffs):
            _, ps[jj] = permutation_test(diff.mean(), [diff.sample, diff.sample2], null_difference)
        
        means = np.array([diff.mean() for diff in diffs])
        CIs = np.abs(np.array([diff.prctile() for diff in diffs]).T-means)
        p_level = constrain_FDR(ps)
        sigs = np.where(ps<=p_level)[0]
        notsigs = np.where(ps>p_level)[0]
        
        # This part is for making the rate differences plot
        ax = plt.subplot2grid(GRID, (0,ii), rowspan = 2)
        ax.errorbar(np.zeros(len(sigs))+means[sigs],sigs,xerr=CIs[:,sigs], fmt='o', color='k')
        ax.errorbar(np.zeros(len(notsigs))+means[notsigs],notsigs,xerr=CIs[:,notsigs], fmt='o', color='grey')
        ax.set_ylim(-1,len(units))
        yl = ax.get_ylim()
        xl = ax.get_xlim()
        ax.plot([0,0],yl,'--', color = 'grey')
        ax.set_yticks(range(0,len(units)))
        ax.set_yticklabels([str(unit.id) for unit in units])
        ax.set_title(interval)
        
        # And this part is for the histograms
        ax = plt.subplot2grid(GRID, (2,ii))
        BIN_W = 0.2 # Bin width
        BINS = int((xl[1]-xl[0])/BIN_W)
        ax.hist(means[notsigs], bins=BINS, range=xl, facecolor = 'w', edgecolor = 'k')
        ax.hist(means[sigs], bins=BINS, range=xl, color = 'k')

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
        bar_colors = 'steelblue'
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
    #plt.title('Unit 20, Uncued trials (RM)', size='large')
    xtlabels = [ traj_map[traj] for traj in trajectories ]
    plt.xticks(xs, xtlabels, rotation=45, ha='right', size='large')
    plt.xlim(-0.25,1.75)
    yl = plt.ylim()
    add_sig_colors(legend_loc[0], legend_loc[1], ax, p)
    fig.tight_layout()
    return ax

def raster(trialData, events = ['PG in', 'FG in'], 
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
    spike_times = srtdata['timestamps']
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
    
def basic_figs(trialData, prange=(-10,3), smooth = False):
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
        rates = my.analyze.ratehist(data, prange = prange)
    else:
        rates = my.analyze.smooth(data, prange = prange)
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