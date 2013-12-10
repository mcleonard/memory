import numpy as np
import pandas as pd
import os
import cPickle as pkl
from spikesort.cluster import load_clusters
import bhv

class Timelock(object):
    ''' A class to timelock spike timestamps to an event. 
    
    Parameters
    ----------
    unit_list : a list of units from the catalog.
    
    Methods
    -------
    self.lock(event) :  Aligns the timestamps for each trial such that
        t = 0 s is the time of the event in the trial
    self.get(unit): Returns the timelocked data for unit
    '''
    
    def __init__(self, unit_list):
        
        self.units = { unit.id:unit for unit in unit_list }
        
        sessions = np.unique([unit.session for unit in self.units.itervalues()])
        self._sessions = sessions
        self._rawdata = self._setup()
        self._locked = dict.fromkeys(sessions)

    def _setup(self):

        processed = dict.fromkeys(self._sessions)

        for session in self._sessions:

            data = _load_session_data(session)
            trial_data = _build_trial_data(data)

            unit_map = {(u.tetrode, u.cluster):u.id for u in session.units}
            sync = trial_data['onset'] - trial_data['n_onset']

            for tetrode in data['cls']:
                clusters = data['cls'][tetrode]
                for cluster in clusters:
                    times = clusters[cluster]['times']
                    timestamps = []
                    for ii in trial_data.index:
                        low = times-30 < trial_data['n_onset'][ii]
                        high = times+30 > trial_data['n_onset'][ii]
                        timestamps.append(times[np.where(low*high)] + sync[ii])
                    unit_id = unit_map[tetrode, cluster]
                    trial_data[unit_id] = pd.Series(timestamps, index=trial_data.index)

            # Get rid of all trials that take too long, setting it to 20 seconds
            delay_limit = 20
            delay = trial_data['C in'] - trial_data['PG in']
            trial_data = trial_data[delay < delay_limit]
        
            processed[session] = trial_data

        return processed
            
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
    
    def __getitem__(self, unit_id):
        ''' Returns data for the given unit id. '''
        unit = self.units[unit_id]
        # Return everything except the first trial because it's junk
        data = self._locked[unit.session][1:]
        data['timestamps'] = data[unit.id]
        spikecols = data.columns[[type(col)==type(1) for col in data.columns ]]
        return data.drop(spikecols, axis=1)

    def __repr__(self):
        
        if hasattr(self, 'event'):
            return "Sessions %s locked to %s" % (self._sessions, self.event)
        else:
            return "Not locked yet"

def timelock(units):
    return Timelock(units)


def _load_session_data(session):
    rat = session.rat
    date = ''.join(session.date.isoformat().split('-'))[2:]
    tetrodes = np.unique([unit.tetrode for unit in session.units])
    data_dir = os.path.expanduser(session.path)
    ext = ['bhv', 'cls', 'syn', 'ons']
    data = dict.fromkeys(ext)
    for each in ext:
        if each == 'cls':
            data[each]={tetrode:None for tetrode in tetrodes}
            for tetrode in tetrodes:
                filepath = os.path.join(data_dir, '{}_{}.{}.{}'.format(rat, date, each, tetrode))
                data[each][tetrode] = load_clusters(filepath)
        else:
            filepath = os.path.join(data_dir, '{}_{}.{}'.format(rat, date, each))
            with open(filepath) as f:
                data[each] = pkl.load(f)
    
    return data

def _build_trial_data(loaded_data):
    bdata = bhv.build_data(loaded_data['bhv'])
    sync = loaded_data['syn'].map_n_to_b_masked
    trial_data = bdata.ix[sync.data[~sync.mask]]
    samp_rate = 30000.0
    n_onsets = loaded_data['ons'][~sync.mask]/samp_rate
    trial_data['n_onset'] = n_onsets
    
    return trial_data

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