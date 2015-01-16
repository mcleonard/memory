''' Module containing functions to generate figures '''

from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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


@passed_or_new_ax
def raster(trial_data, ax=None, figsize=(6,5), sort_by='FG in', 
           events=['C in', 'FG in'], markersize=5, limit=(-1,1)):

    # set colors for trial events
    event_colors = {'C in':'steelblue', 
                    'FG in':'orange',
                    'PG out':'crimson',
                    'C out':'limegreen'}

    # Sort the trials first, for visual clarity
    sorted_data = trial_data.sort(columns=sort_by)

    timestamps = np.hstack(sorted_data['timestamps'].values)
    trial_nums = np.hstack([[jj]*len(row) for jj, row in 
                                        enumerate(sorted_data['timestamps'])])
    ax.scatter(timestamps, trial_nums, marker='|', color='k', s=markersize)
    for each in events:
        ax.scatter(sorted_data[each].values, range(len(sorted_data)), 
                   marker='.', 
                   c=event_colors[each], 
                   edgecolors=event_colors[each])

    ax.set_xlim(*limit)

    return ax

@passed_or_new_ax
def stepped_histogram(x, y, ax=None, figsize=(6,5), color='#444444'):
    """ Plots a stepped histogram without the vertical lines.

    Parameters
    ----------
    x : bin edges
    y : bin values

    """
    
    bin_edges = np.array(zip(x[:-1], x[1:])).ravel()
    ax.plot(bin_edges, np.repeat(y, 2), color=color)

    return ax

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

def pub_format(ax, legend=None, spine_width=1.3, spine_color='#111111', 
               hide_spines=['top', 'right'], 
               yticks=3, xticks=3, yminorticks=1, xminorticks=1):
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
    ax.yaxis.set_minor_locator(MaxNLocator(yticks*yminorticks))
    ax.xaxis.set_minor_locator(MaxNLocator(xticks*xminorticks))
    
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