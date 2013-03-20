import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import memory as my


def bootstrap_difference(trialData, interval, mem_type, goal, iters = 200):
    ''' For the given unit, unitData, interval, mem_type, and goal, returns a Bootstrapper object
        with the bootstrapped distribution of difference in means of right trials minus left trials.
    
    Arguments
    ---------
    unit : catalog Unit
    unitData : pandas DataFrame, get from lkr.get(unit)
    interval : a valid interval string from analyze.slice ('Early', 'Middle', 'Late', etc.)
    mem_type : int, either memory.WM (working memory) or memory.RM (reference memory)
    goal : 'PG' (past goal) or 'FG' (future goal)
    
    Returns
    -------
    out : Bootstrapper with the bootstrapped difference of means, right - left.
        You can get the mean, out.mean(), and CIs, out.prctile().
    '''
    from memory.stats import Bootstrapper
    data = trialData
    hits = data.groupby(by=['block','PG outcome','FG outcome']).get_group((mem_type,my.HIT, my.HIT))
    sides = hits.groupby(by=goal+' response')
    rates = np.array( [my.slice(group)[interval].values for _, group in sides] )
    # I like to define it as right - left, so the value is to the left of zero if left rate is greater
    diff = Bootstrapper(rates[1], s2 = rates[0], stat_func = lambda x,y:np.mean(x)-np.mean(y), iters = iters)
    return diff

def smooth(trialData, width=0.1, prange = (-5,5)):
    ''' Smoothes the unit spiking data by replacing every spike with a gaussian.
        Not convinced this is faster than doing a fftconvolve, but okay.
        Arguments:
        width : float
            width of the gaussians 
        range : 2 element tuple
            range over which to return the smoothed spikes
    '''
    
    # This returns arrays a binary spike train with 1 ms bins
    trains = spike_trains(trialData, prange=prange)
    x = trains.columns.values.astype(float)
    gaussian = lambda u: 1/width/np.sqrt(2*np.pi)*np.exp(-(x-u)**2/2/width**2)
    smoothed = DataFrame(index = trains.index, columns = trains.columns)
    for index, train in trains.iterrows():
        if sum(train)>0:
            spikes = np.where(train==1)[0]
            gausses = [ gaussian(u) for u in x[spikes] ]
            smoothed.ix[index] = np.sum(gausses, axis=0)
        else: smoothed.ix[index] = train
    return smoothed

def get_memory_sides(data, goal = 'PG', outcomes = ('hit','hit')):
    ''' Separates the data into working memory and reference memory,
        then further into response = left and response = right
    '''
    pg, fg = [globals()[outcome.upper()] for outcome in outcomes]
    select = (data['PG outcome']==pg) & (data['FG outcome']==fg)
    selected = data[select]
    
    wm = selected[selected['block']==WM] # Working memory
    rm = selected[selected['block']==RM] # Reference memory
    
    wm_side = {side:wm[wm[goal +' response']==side] for side in [LEFT, RIGHT]}
    rm_side = {side:rm[rm[goal +' response']==side] for side in [LEFT, RIGHT]}
    
    return {('wm','left'):wm_side[LEFT], ('wm','right'):wm_side[RIGHT],
            ('rm','left'):rm_side[LEFT], ('rm','right'):rm_side[RIGHT]}

def get_trajectories(data, mem = 'wm', outcomes = ('hit','hit')):
    pg, fg = [globals()[outcome.upper()] for outcome in outcomes]
    select = (data['PG outcome']==pg) & (data['FG outcome']==fg)
    selected = data[select]
    
    mem_map = {'wm':WM,'rm':RM}
    block = selected[selected['block']==mem_map[mem]]
    pgleft = block[block['PG response']==LEFT]
    fgleft = block[block['FG response']==LEFT]
    pgright = block[block['PG response']==RIGHT]
    fgright = block[block['FG response']==RIGHT]
    def trajectory(pgside, fgside):
        indices = list(set(pgside.index).intersection(set(fgside.index)))
        return block.ix[indices]
    return {('right','left'):trajectory(pgright,fgleft),
            ('left','left'):trajectory(pgleft,fgleft),
            ('left','right'):trajectory(pgleft,fgright),
            ('right','right'):trajectory(pgright,fgright)}

def ratehist(trialData, bin_width=0.200, prange = (-20,2)):
    ''' Output is a DataFrame, time as the columns, index is trial numbers'''
    
    timestamps = trialData['timestamps']
    nbins = np.diff(prange)[0]/bin_width
    columns = np.arange(prange[0], prange[1], bin_width)+bin_width/2.
    rateFrame = DataFrame(index = trialData.index, 
                          columns = columns)
    for ind, times in timestamps.iteritems():
        count, x = np.histogram(times, bins = nbins, range = prange)
        rate = count/bin_width
        rateFrame.ix[ind] = rate
    return rateFrame

def spike_trains(trialData, prange = (-20,2)):
    ''' Returns a binary spike train.  '''
    
    train = ratehist(trialData, bin_width = 0.001, prange = prange)
    train = train/1000.0
    return train

def raster(unit, trialData, sort_by= 'PG in', range = None):
    """ A function that creates a raster plot from the units' trial data 
    
    Arguments
    ---------
    unit : catalog Unit
    
    trialData : pandas DataFrame
        The data returned from Timelock.get(unit)
    
    sort_by : string
        Column in trialData that the trials with be sorted by
    
    range : tuple of size 2
        Plots from time range[0] to time range[1] (in seconds)
    """
    
    print "DEPRECATED: Use mycode.plots.raster"
    
    uid = unit.id
    data = trialData
    
    srtdata = data.sort(sort_by, ascending = False)
    index = srtdata.index
    for trial, times in enumerate(srtdata[uid]):
        ind = index[trial]
        plt.scatter(times, np.ones(len(times))*trial, 
                    c = 'k', marker = '|', s = 5)
        plt.scatter(srtdata['PG in'][ind], trial, color = 'red', marker = 'o')
        plt.scatter(srtdata['FG in'][ind], trial, color = 'blue', marker = 'o')
    
    plt.plot([0,0], [0,len(data)], color = 'grey')
    if range == None:
        plt.xlim(srtdata['PG in'].mean()-2, srtdata['FG in'].mean()+2)
    else:
        plt.xlim(range)
    plt.ylim(0,len(data))
    plt.show()

def basic_figs(trialData, range=(-10,3)):
    from itertools import izip
    print "DEPRECATED: use mycode.plots"
    
    data = trialData
    uid = unit.id
    times = ratehist(data, range= range)
    
    def base_plot(xs, rates, label):
        
        plt.figure()
            
        for rate, lab in izip(rates, label):
            plt.plot(xs, rate, label = lab)
        
        plt.plot([0,0], plt.ylim(), '-', color = 'grey')
        plt.xlabel('Time (s)')
        plt.ylabel('Activity (s)')
        plt.legend()
        
    # First plot all data
    base_plot(times.columns.values,[times.mean()], label = ['All trials'])
    
    # Plot PG left vs PG right
    pgleft = times[data['hits'] & (data['PG response']==consts['LEFT'])]
    pgright = times[data['hits'] & (data['PG response']==consts['RIGHT'])]
    rates = [pgleft.mean(), pgright.mean()]
    base_plot(times.columns.values, rates, label = ['PG left', 'PG right'])
    
    # Plot FG left vs FG right
    fgleft = times[data['hits'] & (data['FG response']==consts['LEFT'])]
    fgright = times[data['hits'] & (data['FG response']==consts['RIGHT'])]
    rates = [fgleft.mean(), fgright.mean()]
    base_plot(times.columns.values, rates, label = ['FG left', 'FG right'])
    
    # Plot cued vs uncued
    cued = times[data['hits'] & (data['block']==RM)]
    uncued = times[data['hits'] & (data['block']==WM)]
    rates = [cued.mean(), uncued.mean()]
    base_plot(times.columns.values, rates, label = ['cued', 'uncued'])

def trajectory(trialData, range= (-10,3)):
    
    data = trialData
    
    # Define the plot
    def base_plot(data, range):
        
        ylims=[0]*4
        titles = ['right->left', 'left->left', 'left->right', 'right->right']
        
        
        traj = [('RIGHT', 'LEFT'), ('LEFT', 'LEFT'), ('LEFT', 'RIGHT'), 
                ('RIGHT', 'RIGHT')]
        
        traj_data = [ data[(data['PG response'] == consts[tr[0]])
                    & (data['FG response'] == consts[tr[1]])] for tr in traj]
        
        subplots = np.arange(221,225)
        
        # Making the rate histograms
        plt.figure()
        for ii, num in enumerate(subplots):
            plt.subplot(num)
            rates = ratehist(unit, traj_data[ii], range = range)
            plt.plot(rates.columns, rates.mean())
            plt.title(titles[ii])
            plt.xlim(range)
            ylims[ii] = plt.ylim()[1]
        
        # Scaling all subplots to the same max y
        ymax = np.max(ylims)
        
        for num in subplots:
            plt.subplot(num)
            plt.ylim(0,ymax)
        
        # Now making the raster plots
        plt.figure()
        for ii, num in enumerate(subplots):
            plt.subplot(num)
            raster(traj_data[ii], range = range)
            plt.title(titles[ii])
            
    cued = data[data['hits'] & (data['block']==RM)]
    uncued = data[data['hits'] & (data['block']==WM)]
    
    # Pass the data to the plot
    base_plot(cued, range)
    base_plot(uncued, range)

def interval(trialData, low, high):
    
    lows = ['PG in','PG out','C in','C out','L reward','R reward','onset']
    highs = ['PG out','C in','C out','L reward','R reward','onset','FG in']
    
    if low in lows:
        pass
    else:
        raise ValueError, '%s not a valid option for low' % low
        
    if high in highs:
        pass
    else:
        raise ValueError, '%s not a valid option for high' % high
    
    times = trialData[['timestamps', low, high]].T
    for ii, series in times.iteritems():
        spiketimes = series.ix[0]
        lowtime = series.ix[1]
        hightime = series.ix[2]
        eptimes = spiketimes[(spiketimes>=lowtime) & (spiketimes<=hightime)]
        times[ii].ix[0] = eptimes
    
    return times.T
    
def slice(trialData, sort_by = None, show = False, times=False):
    ''' Slices the data into six intervals, in the PG port, early delay,
        middle delay, late delay, in the center port, between the center
        port and FG port.  The average rate in each of these intervals is
        calculated for each trial.
        
    Arguments
    ---------
    unit : catalog.Unit
        The Unit object for the unit you want to slice.
    trialData : pandas DataFrame
        DataFrame containing trial information and unit spike timestamps.
        Get this from Timelock().get(unit).
    sort_by : string
        String indicating the trial information column to sort by.
    show : bool
        Show a figure of the rates over intervals and trials
    
    '''
    
    data = trialData
    
    if sort_by in trialData.columns:
        data = trialData.sort(columns=sort_by)
    columns = ['PG port', 'Early', 'Middle', 'Late', 'C port', 'C to FG']
    rates = DataFrame(index=data.index, columns = columns, dtype='float64')
    intervals = DataFrame(index=data.index, columns = columns, dtype='float64')
    
    for ind, row in data.iterrows():
        pg = (row['PG in'], row['PG out'])
        fg = (row['C out'], row['FG in'])
        cent = (row['C in'], row['C out'])
        delay = (row['PG out'], row['C in'])
    
        counts = [ np.histogram(row['timestamps'], bins = 1, range=pg)[0] ]
        
        counts.append(np.histogram(row['timestamps'], bins=3, range=delay)[0])
        counts.extend([np.histogram(row['timestamps'], bins = 1, range=period)[0] 
                        for period in [cent, fg]])
        
        counts = np.concatenate(counts)
        diffs = [pg[1]-pg[0], (delay[1]-delay[0])/3.0, (delay[1]-delay[0])/3.0,
                (delay[1]-delay[0])/3.0, cent[1]-cent[0], fg[1]-fg[0]]
        
        rates.ix[ind] = counts/diffs
        intervals.ix[ind] = diffs
        
    rates.fillna(value=0, inplace=True)
    
    if show:
        plt.imshow(rates.astype(float), 
                    aspect='auto', 
                    interpolation = 'nearest',
                    extent=[0,5,0,len(rates)])
    
    if times:
        return rates, intervals
    else:
        return rates

def confidence_sig(xerr, yerr):
    
    print "DEPRECATED: use mycode.plots.cross_scatter"
    
    from itertools import repeat
    from matplotlib.mlab import find
    def is_sig(test, between):
        is_it = (between[0] <= test) & (between[1] >= test)
        return not is_it
    
    sig = dict.fromkeys(['x','y','both', 'neither'])
    sigx = map(is_sig, repeat(0, xerr.shape[1]), xerr.T)
    sigy = map(is_sig, repeat(0, yerr.shape[1]), yerr.T)
    sig['x'] = set(find(sigx))
    sig['y'] = set(find(sigy))
    sig['both'] = sig['x'].intersection(sig['y'])
    sig['neither'] = \
        set(range(xerr.shape[1])).difference(sig['x'].union(sig['y']))
    
    # I tried doing this with a for loop, but it didn't seem to work.
    sig['x'] = np.array(list(sig['x']))
    sig['y'] = np.array(list(sig['y']))
    sig['both'] = np.array(list(sig['both']))
    sig['neither'] = np.array(list(sig['neither']))
    
    return sig

def p_value_sig(p_x, p_y, sig_p = 0.05):
    
    print "DEPRECATED: use mycode.plots.cross_scatter"
    
    # Test for significance
    sigs = {'x':np.where((p_x<=sig_p)&(p_y>sig_p))[0],
            'y':np.where((p_y<=sig_p)&(p_x>sig_p))[0],
            'both':np.where((p_x<=sig_p)&(p_y<=sig_p))[0],
            'neither':np.where((p_x>sig_p)&(p_y>sig_p))[0]}
    return sigs

def cross_scatter(x, y, xerr, yerr, p_x, p_y, **kwargs):
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
        Array of p-values for each data point.
        
    Returns
    -------
    sig : list of bools
        Boolean values for the data points that are significant
        Signficance is tested as zero being outside the error bars
    ax : matplotlib.axes.AxesSubplot
        Returns the axes for the scatter plot so you can add labels and such
    '''
    
    print "DEPRECATED: use mycode.plots.cross_scatter"

    valid_kwargs = ['xlabel', 'ylabel', 'title', 'sig_level']

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    kwarg_dict = {'xlabel':ax.set_xlabel,
                  'ylabel':ax.set_ylabel,
                  'title':ax.set_title}
    
    sig_colors = {'x':'r', 'y':'b', 'both':'purple', 'neither':'grey'}
    
    xbars = np.abs(xerr - x)
    ybars = np.abs(yerr - y)
    
    if 'sig_level' in kwargs:
        sig_level = kwargs.pop('sig_level')
    else:
        sig_level = 0.5
    sigs = p_value_sig(p_x,p_y,sig_p=sig_level)
    
    # Then plot units that are significant for working and reference memory
    for key in ['neither','both','y','x']:
        sig = sigs[key]
        if len(sig):
            ax.errorbar(x[sig],y[sig],
                        yerr=ybars[:,sig],xerr=xbars[:,sig],
                        fmt='o', color = sig_colors[key])
    
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_autoscale_on(False)
    ax.plot(xlim,[0,0],'-k')
    ax.plot([0,0],ylim,'-k')
    ax.plot([-100,100],[-100,100],'--',color='grey')
    ax.set_aspect('equal')
    for kwarg in kwargs:
        axfunc = kwarg_dict[kwarg]
        axfunc(kwarg)
    
    #plt.show()
    return sigs, ax

