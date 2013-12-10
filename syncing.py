''' This script syncs the behavior and neural data.  You'll run this
after spike sorting, before timelocking.  Cheers!
'''

from matplotlib.mlab import find
import ns5
import numpy as np
import pickle as pkl
import os
import sys

import bcontrol
import DataSession
import AudioTools

# Need to create this class to work with the syncing algorithm
class Onsets(object):
    def __init__(self,onsets):
      self.audio_onsets = onsets

# Nov. 9, 2012...  Rewriting this to sync all of my data at once

ratname = sys.argv[1]
topdir = os.environ['HOME'] + '/Dropbox/Data'
NDAQ = '/media/hippocampus/NDAQ'
filelist = os.listdir(os.path.join(topdir,ratname))

dir_test = [ os.path.isdir(os.path.join(topdir, ratname, file)) for file in filelist ]
dates = [ filelist[ind] for ind in find(dir_test) ]

for date in dates:
    
    # Test if this data has already been synced
    data_dir = os.path.join(topdir,ratname,date)
    dir_list = os.listdir(data_dir)
    sync_test = [ '.syn' in file for file in dir_list ]
    if True in sync_test:
        continue
    
    # If not, let's do some syncing
    #behave_file = '/home/mat/Dropbox/Working-memory/ALY1A/data_@TwoAltChoice_Memory_Mat_ALY1A_120926a.mat' 
    #neural_file = '/media/hippocampus/NDAQ/datafile_ML_ALY11_120926_001.ns5'
    
    bfile_name = dir_list[find(['.mat' in file for file in dir_list])]
    
    behave_file = os.path.join(topdir,ratname,date,bfile_name)
    neural_file = os.path.join(NDAQ,'datafile_ML_%(ratname)s_%(date)s_001.ns5' % locals()) 
    save_as = os.path.join(data_dir,'%(ratname)s_%(date)s' % locals())

    loader = ns5.Loader(neural_file)
    loader.load_header()
    loader.load_file()
       
    audio = [ loader.get_analog_channel_as_array(n) for n in [7,8] ] 
    audio = np.array(audio)*4096./2**16
        
    onsets_obj = AudioTools.OnsetDetector(audio, verbose = True, 
        minimum_threshhold=-20)
    onsets_obj.execute()
    n_onsets = Onsets(onsets_obj.detected_onsets)

    bcload = bcontrol.Bcontrol_Loader(filename = behave_file, mem_behavior= True, auto_validate = 0) 
    bcdata = bcload.load()
    b_onsets = Onsets(bcdata['onsets'])
       
    syncer = DataSession.BehavingSyncer()
    syncer.sync(b_onsets,n_onsets, force_run = 1)
    
    with open(save_as + '.bhv','w') as f:
        bcontrol.process_for_saving(bcdata)
        pkl.dump(bcdata,f)

    with open(save_as + '.ons','w') as f:
        pkl.dump(n_onsets.audio_onsets,f)
         
    with open(save_as+ '.syn','w') as f:
        pkl.dump(syncer,f)
