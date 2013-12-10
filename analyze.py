import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp

from bhv import HIT, ERROR, WM, RM, LEFT, RIGHT
import stats


intervals = ['PG port', 'Early', 'Middle', 'Late', 'C port', 'C to FG']

class UnitDict(dict):
    
    def __init__(self, *args, **kwargs):
        super(UnitDict, self).__init__(*args, **kwargs)
    
    def groupby(self, *args, **kwargs):
        grouped = { unit_id:unit.groupby(*args, **kwargs) 
                    for unit_id, unit in self.iteritems() }
        return UnitDict(grouped)
    
    def get_group(self, *args, **kwargs):
        groups = { unit_id:unit.get_group(*args, **kwargs) 
                   for unit_id, unit in self.iteritems() }
        return UnitDict(groups)

    def map(self, func, **kwargs):

        mapped = { unit_id:func(unit, **kwargs) 
                   for unit_id, unit in self.iteritems() }

        return UnitDict(mapped)

    def itergroups(self):
        groups =  self[self.keys()[0]].groups.keys()
        for group in groups:
            grouped = { unit_id:unit.get_group(group) 
                        for unit_id, unit in self.iteritems() }

            yield (group, UnitDict(grouped))
    
    def __repr__(self):
        return "<UnitDict({} units of type {})>".format(len(self), 
                                                  type(self[self.keys()[0]]))

def rate_difference(trial_data, mem=WM, goal='FG', interval='Late', iters=200):
    block = trial_data.groupby('block').get_group(mem)
    rates = interval_rates(block)[interval]
    goals = block.groupby(['{} port'.format(goal)])
    func = lambda s1, s2: np.mean(s1) - np.mean(s2)
    values = { side:rates.ix[group.index].values for side, group in goals }
    boot = stats.bootstrap(values[RIGHT], s2=values[LEFT], 
                              stat_func=func, iters=iters)
    null, p = stats.permutation_test(boot.mean(), 
                                    (values[RIGHT], values[LEFT]), 
                                    stats.null_difference)
    
    return boot, p

def batch_rate_differences(unit_dict, mem=WM, goal='FG', interval='Late', 
                           iters=200, sorted=None):
    boots = unit_dict.map(rate_difference, mem=mem, goal=goal, 
                          interval=interval, iters=iters)
    means = np.array([each[0].mean() for each in boots.itervalues()])
    confs = np.array([each[0].prctile() for each in boots.itervalues()])
    ps = np.array([each[1] for each in boots.itervalues()])
        
    df = pd.DataFrame({'id':boots.keys(),
                       'mean':means, 
                       'conf_low':np.abs(confs[:,0]-means), 
                       'conf_high':np.abs(confs[:,1]-means),
                       'p':ps})

    if sorted is None:
        df = df.sort(columns=['mean'])
    else:
        df = df.ix[sorted.index]
    
    df['x'] = range(len(df))
    df['sig'] = df['p'] <= stats.constrain_FDR(df['p'].values)

    return df

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
    smoothed = pd.DataFrame(index = trains.index, columns = trains.columns)
    for index, train in trains.iterrows():
        if sum(train)>0:
            spikes = np.where(train==1)[0]
            gausses = [ gaussian(u) for u in x[spikes] ]
            smoothed.ix[index] = np.sum(gausses, axis=0)
        else: smoothed.ix[index] = train
    return smoothed

def time_histogram(trialData, bin_width=0.200, prange = (-10,2)):
    ''' Output is a DataFrame, time as the columns, index is trial numbers'''
    
    timestamps = trialData['timestamps']
    nbins = np.around(np.diff(prange)[0]/bin_width).astype(int)
    _, x = np.histogram(timestamps.ix[timestamps.index[0]], 
                        bins = nbins, range = prange)
    columns = x[:-1]
    rateFrame = pd.DataFrame(index = trialData.index, 
                             columns = columns, dtype=float)
    for ind, times in timestamps.iteritems():
        count, x = np.histogram(times, bins = nbins, range = prange)
        rate = count/bin_width
        rateFrame.ix[ind] = rate
    return rateFrame

def spike_trains(trialData, prange = (-20,2)):
    ''' Returns a binary spike train.  '''
    
    train = time_histogram(trialData, bin_width = 0.001, prange = prange)
    train = train/1000.0
    return train

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
    
def interval_rates(trialData, sort_by = None, show = False, times=False):
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
    rates = pd.DataFrame(index=data.index, columns = columns, dtype='float64')
    intervals = pd.DataFrame(index=data.index, columns = columns, dtype='float64')
    
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
