import numpy as np

class Timelock(object):
    ''' A class to timelock spike timestamps to an event. 
    
    Parameters
    ----------
    unit_list : a list of units from the catalog.
    
    Methods
    -------
    self.lock(event) :  Aligns the timestamps for each trial such that
        t = 0 s is the time of the event in the trial
    self.get(unit): Returns the DataFrame for unit
    '''
    
    def __init__(self, unit_list):
        
        self.units = unit_list
        self._setup()

    def _setup(self):
        
        import bhv
        from os.path import expanduser
        from itertools import izip
        from scipy import unique
        
        sessions = unique([unit.session for unit in self.units])
        self._sessions = sessions
        self._rawdata = dict.fromkeys(sessions)
        
        for session in sessions:
            
            datadir = expanduser(session.path)
            # We might might be looking at all the units in this session
            session_units = [ unit for unit in self.units if unit in session.units ]
            tetrodes = unique([unit.tetrode for unit in session_units])
            
            data = _load_data(datadir, tetrodes)
            
            # Get all the processed behavior data from a Rat object
            bdata = data['bhv']
            rat = bhv.Rat(session.rat)
            rat.update(session.date, bdata)
            trial_data = rat.sessions['trials'][0]
            
            # Keep only the behavior trials we have neural data for
            sync = data['syn'].map_n_to_b_masked
            trial_data = trial_data.ix[sync.data[~sync.mask]]
            
            # Get the onsets from the neural data
            samp_rate = 30000.0
            n_onsets = data['ons'][~sync.mask]/samp_rate
            trial_data['n_onset'] = n_onsets
            
            for tetrode in tetrodes:
            
                units = [ unit for unit in session_units if unit.tetrode == tetrode ]
                clusters = [ data['cls'][tetrode][unit.cluster] for unit in units ]
                # Using izip saves memory!
                packed = izip(units, clusters)
                
                # For each trial, we want to grab all the spikes between PG and FG,
                # plus 30 seconds on either side
                for unit, cluster in packed:
                    times = _get_times(cluster['peaks'], trial_data)
                    trial_data[unit.id] = times
                
            # Get rid of all trials that take too long, setting it to 20 seconds
            delay_limit = 20
            delay = trial_data['C in'] - trial_data['PG in']
            trial_data = trial_data[delay < delay_limit]
        
            self._rawdata[session] = trial_data
            
    def lock(self,event):
        '''  Sets t = 0 s for spike timestamps to the time of the event in that
        trial 
        
        Parameters
        ----------
        event : the event that you want to set as t = 0.  Currently, the valid
            events are ['PG in', 'PG out', 'C in', 'onset', 'C out', 'FG in'].
            You can add more by creating time information columns in the rat
            class of the bhv module. 
        
        Returns
        -------
        self : object
            Returns self.
        
        '''
        
        self._locked = dict.fromkeys(self._sessions)
        for session in self._sessions:
            
            self.event = event
            
            valid_events = ['PG in', 'PG out', 'C in', 'onset', 'C out', 'FG in']
            if event in valid_events:
                pass
            else:
                raise ValueError('%s is not a valid event' % event)
            
            trial_data = self._rawdata[session].copy()
            
            # I want to subtract time zero from all the time columns
            spikecols = trial_data.columns[[type(col)==type(1) for col in trial_data.columns ]]
            timecolumns = valid_events[:]
            timecolumns.extend(spikecols)
            timecolumns.remove(event)
            tzero = trial_data[event]
            for column in timecolumns:
                trial_data[column] = trial_data[column] - tzero
            trial_data[event] = 0
            
            self._locked[session] = trial_data
    
        return self
    
    def get(self, unit):
        ''' Returns data for the given unit. '''
        
        return self._locked[unit.session]
    
    def __repr__(self):
        
        if hasattr(self, 'event'):
            return "Sessions %s locked to %s" % (self._sessions, self.event)
        else:
            return "Not locked yet"

def _load_data(datadir, tetrodes):
    
    import os
    import re
    import pickle as pkl
    from . import DataSession
    
    filelist = os.listdir(datadir)
    files = re.findall('([a-zA-Z]+\d+[a-zA-Z]*)_(\d+).([a-z]+)', ' '.join(filelist))
    
    ext = ['bhv', 'cls', 'syn', 'ons']
    data = dict.fromkeys(ext)
    data['cls'] = {}
    
    # Load the data files into data dictionary
    for file in files:
        if file[2] in ['bhv', 'syn', 'ons']:
            filename = '{}_{}.{}'.format(*file)
            filepath = os.path.join(datadir, filename)
            with open(filepath,'r') as f:
                    data[file[2]] = pkl.load(f)
        elif file[2] == 'cls':
            for tetrode in tetrodes:
                filename = '{}_{}.{}'.format(*file)
                filepath = '%s/%s.%s' % (datadir, filename, tetrode)
                with open(filepath,'r') as f:
                    data['cls'].update({tetrode:pkl.load(f)})
            
    # Checking to make sure the data files were loaded
    if None in data.viewvalues():
        for key, value in data.iteritems():
            if value == None:
                raise Exception, '%s file wasn\'t loaded properly' % key
    
    return data
    
def _get_times(timestamps, trial_data):
    from matplotlib.mlab import find
    out_times = []
    if type(timestamps) != type(1):
        for ii, trial in trial_data.iterrows():
            
            low = trial['n_onset'] - (trial['onset'] - trial['PG in'])
            high = trial['n_onset'] + (trial['onset'] - trial['FG in'])
            
            in_window = find((timestamps > (low-30)) & (timestamps < (high+30)))
            stamps = sorted(timestamps[in_window])
            
            # And align the neural time stamps with the behavior time stamps
            stamps = stamps + np.abs(trial['n_onset']-trial['onset'])
            out_times.append(stamps)
        
        return out_times
    else:
        return None
    
    