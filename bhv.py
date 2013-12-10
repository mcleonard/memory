''' Behavior data analysis '''

# Defining constants here
# These might change but probably not, but just be aware
CHOICE_TIME_UP = 3
HIT = 2
ERROR = 1
LEFT = 1
RIGHT = 2
RM = 1
WM = 2

import bcontrol
import numpy as np
from matplotlib.mlab import find
from pandas import Series, DataFrame

def get_trials_data(Session):
    ''' Accepts a Session from a Catalog and returns the behavior data in a
        nice pandas DataFrame packaged in a named tuple'''
    
    import memory as my
    import cPickle as pkl
    from collections import namedtuple

    Trials = namedtuple("Trials", ['rat','date','data'])
    filename = my.utils.filepath_from_dir(Session.path, 'bhv')
    if len(filename)==1:
        filename = filename[0]
    else:
        pass # TODO: Ad an exception here for zero or multiple files
    
    with open(filename, 'r') as f:
        bdata = pkl.load(f)

    return Trials(Session.rat, Session.date, build_data(bdata))

def build_data(bdata):
    ''' Takes the data from bdata (get this from load_bhv_data) and formats it 
        into a pandas DataFrame for easier analysis 
    '''
    
    # Set up the DataFrame
    records = [ 'stimulus', 'block', 'hits', 'errors', 
                'FG port', '2PG port', 'PG port', 
                'FG outcome', '2PG outcome', 'PG outcome',
                'FG response', '2PG response', 'PG response',
                'PG in', 'PG out', 'FG in','C in', 'C out', 'L reward',
                'R reward', 'onset']
    
    n_trials = len(bdata['onsets'])
    
    #  Create the DataFrame
    trials = DataFrame(index = range(n_trials), columns = records)
    
    trials_info = bdata['TRIALS_INFO'][:n_trials]
    trials['hits'] = Series(trials_info['OUTCOME'] == HIT, dtype=bool)
    trials['errors'] = Series(trials_info['OUTCOME'] == ERROR, dtype=bool)
    trials['FG port'] = trials_info['CORRECT_SIDE']
    trials['PG port'] = trials['FG port'].shift(1)
    trials['2PG port'] = trials['FG port'].shift(2)
    trials['FG outcome'] = trials_info['OUTCOME']
    trials['PG outcome'] = trials['FG outcome'].shift(1)
    trials['2PG outcome'] = trials['FG outcome'].shift(2)
    
    # The FG response is the same as FG port if the rat did it correctly,
    # but it is opposite if the rate did it incorrectly
    # For some reason, I don't think I'm handling timeout cases: TODO: fix this
    trials['FG response'][find(trials['hits'])] = trials['FG port'][trials['hits']]
    incorrect_side = trials['FG port'][find(trials['errors'])]
    swap = {1:2, 2:1}
    swapped_sides = np.array([swap[t] for t in incorrect_side], dtype = 'i2') 
    trials['FG response'][find(trials['errors'])] = swapped_sides
    trials['PG response'] = trials['FG response'].shift(1)
    trials['2PG response'] = trials['FG response'].shift(2)
    
    stimuli = bdata['SOUNDS_INFO']['sound_name']
    stimuli = dict([ (ii, str(sound)) for ii,sound in enumerate(stimuli, 1)])
    trials['stimulus'] = np.array([ stimuli[x] for x in trials_info['STIM_NUMBER']],
                                   dtype='a20')
    # Will need to fix this...
    mem_map = {'cued':RM, 'uncued':WM}
    trials['block'] = np.array([mem_map[tr[3:]] for tr in trials['stimulus']], dtype='i8')
    
    # Now for timing information
    
    trials['onset'] = bdata['onsets']
    
    from itertools import izip
    trial = 1
    # I'm making a decision to skip the first trial since it is worthless
    for previous, current in izip(bdata['peh'], bdata['peh'][1:]):
        
        trials['PG in'][trial] = np.nanmax(previous['states']['choosing_side'])
        
        if trials['PG response'][trial] == LEFT:
            trials['PG out'][trial] = np.nanmax(previous['pokes']['L'])
        elif trials['PG response'][trial] == RIGHT:
            trials['PG out'][trial] = np.nanmax(previous['pokes']['R'])
        else:
            trials['PG out'][trial] = np.nanmax(previous['states']['choice_time_up_istate'])
        
        trials['C in'][trial]= np.nanmin(current['states']['hold_center'])
        
        trials['C out'][trial] = np.nanmin(current['states']['choosing_side'])
        
        trials['FG in'][trial] = np.nanmax(current['states']['choosing_side'])
       
        try:
            trials['L reward'][trial] = np.nanmin(current['states']['left_reward'])
        except ValueError:
            pass
        try:
            trials['R reward'][trial] = np.nanmin(current['states']['right_reward'])
        except ValueError:
            pass
    
        trial += 1
        
    trials.fillna(value = 0, inplace = True)

    # Making sure these columns actually have data types, helps later
    int_cols = ['FG outcome', 'FG response', '2PG port', 'PG port',
                'FG port','PG outcome','2PG outcome', 'PG response', '2PG response']
    for col in int_cols:
        trials[col] = Series(trials[col], dtype='int8')
    flt_cols = ['PG in', 'PG out', 'FG in','C in', 'C out', 'L reward',
                'R reward', 'onset']
    for col in flt_cols:
        trials[col] = Series(trials[col], dtype='float64')
    
    return trials

def load_bhv_data(filename):
    ''' Accepts a file name (the absolute path) to a .mat file containing
        recorded behavior data.  Returns a dictionary with the data in a form
        that can be used in Python '''
        
    bload = bcontrol.Bcontrol_Loader(filename, mem_behavior = True, auto_validate = 0) 
    bdata = bload.load()
    bcontrol.process_for_saving(bdata)
    
    return bdata