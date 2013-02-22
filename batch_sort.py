""" Here's a script to sort all my data at once, so that I can let it run
    then come back later and win everthing.

    Date: 2/12/2013
"""

import memory as my
import os
import re
import cPickle as pkl
import time

DATA_DIR = '/media/hippocampus/NDAQ'
SAVE_DIR = '/media/hippocampus/Working_mem/Sorted'
RAT_NAME = 'CWM019'

filenames = os.listdir(DATA_DIR)
PATTERN = 'datafile_ML_{}_([0-9]*)_001.ns5'.format(RAT_NAME)
regs = [re.search(PATTERN,filename) for filename in filenames]
my_files = [ reg for reg in regs if reg != None ]
LOG_PATH = os.path.join(SAVE_DIR,'sort_log.txt')

if 'sort_log.txt' in filenames:
    pass
else:
    with open(LOG_PATH,'w') as log:
        header = ', '.join(['rat','date','filename','tetrode',
                            'start','finish\n'])
        log.write(header)

for datafile in my_files:
    print "Sorting {}".format(datafile.group(0))
    
    start_time = time.asctime()
    
    tetrodes = range(1,5)
    for tetrode in tetrodes:
        datapath = os.path.join(DATA_DIR, datafile.group(0))
        data = my.load_data(datapath, tetrode = tetrode)
        spikes = my.detect_spikes(data)
        model = my.cluster.bestmodel(spikes, min_K = 17, max_K = 23, processes = 7)
        
        filename = '_'.join([RAT_NAME, 
                             datafile.group(1), 
                             str(tetrode)])
        path = os.path.join(SAVE_DIR,filename+'.scl')
        
        with open(path, 'w') as out:
            pkl.dump(model.clusters, out)
            print "{} saved as {}".format(model, path)
        
        finish_time = time.asctime()
        with open(LOG_PATH,'a') as log:
            logging = ', '.join([RAT_NAME, datafile.group(1), datafile.group(0),
                                str(tetrode), start_time, finish_time+'\n'])
            log.write(logging)
            
    

    