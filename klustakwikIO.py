# Iterate through all neurons in OE db and dump spiketimes
# We will use the KlustaKwik compatible Feature File format
# Ref: http://klusters.sourceforge.net/UserManual/data-files.html
# Begin format specification (lightly modified from above):
"""
Scans through the date directories under a rat directory and converts
.cls files from ePhys sorter to KlustaKwik files.  When you run the script,
input the rat name as the argument.  Like so:

python klustakwikIO.py 'CWM019'


==============================================
The Feature File

Generic file name: base.fet.n

Format: ASCII, integer values

The feature file lists for each spike the PCA coefficients for each
electrode, followed by the timestamp of the spike (more features can
be inserted between the PCA coefficients and the timestamp). 
The first line contains the number of dimensions. 
Assuming N1 spikes (spike1...spikeN1), N2 electrodes (e1...eN2) and
N3 coefficients (c1...cN3), this file looks like:

nbDimensions
c1_e1_spike1   c2_e1_spike1  ... cN3_e1_spike1   c1_e2_spike1  ... cN3_eN2_spike1   timestamp_spike1
c1_e1_spike2   c2_e1_spike2  ... cN3_e1_spike2   c1_e2_spike2  ... cN3_eN2_spike2   timestamp_spike2
...
c1_e1_spikeN1  c2_e1_spikeN1 ... cN3_e1_spikeN1  c1_e2_spikeN1 ... cN3_eN2_spikeN1  timestamp_spikeN1

The timestamp is expressed in multiples of the sampling interval. For
instance, for a 20kHz recording (50 microsecond sampling interval), a
timestamp of 200 corresponds to 200x0.000050s=0.01s from the beginning
of the recording session.

Notice that the last line must end with a newline or carriage return. 
"""

import cPickle
import numpy as np
import os.path
from os import listdir
from matplotlib.mlab import find
import sys

class UniqueError(Exception):
    pass

def unique_or_error(a):
    u = np.unique(np.asarray(a))
    if len(u) == 0:
        raise UniqueError("no values found")
    if len(u) > 1:
        raise UniqueError("%d values found, should be one" % len(u))
    else:
        return u[0]


ratname = sys.argv[1]
dorun = 1

topdir = '/home/mat/Dropbox/Working-memory'
dirlist = listdir(os.path.join(topdir,ratname))
dir_test = [ os.path.isdir(os.path.join(topdir, ratname, fil)) for fil in dirlist ]
dates = [ dirlist[ind] for ind in find(dir_test) ]
pjoin = os.path.join
psplitext = os.path.splitext

f_samp = 30e3

# I'm going to loop through date folders and tetrodes in those folders
for date in dates:
    
    data_dir = pjoin(topdir,ratname,date)
    
    # Find the tetrodes we're dealing with
    files  = listdir(data_dir)
    
    # Check if we already converted to KK files in this directory
    fetfiles = [files[ind] for ind in find([ '.fet' in f for f in files ])]
    if dorun:
        pass
    elif fetfiles:
        continue
    else:
        pass
    
    # Grab the .cls files
    clsfiles = [files[ind] for ind in find([ '.cls' in f for f in files ])]
    
    # If clsfiles is not empty, keep going
    if clsfiles:
        pass
    # Otherwise, skip this loop and go to the next
    else:
        continue
    
    for filename in clsfiles:
        
        # Assuming the filename is foo.cls.%d, where %d is the tetrode number
        tetrode = int(filename[-1])
        basename = psplitext(filename[:-2])[0]

        with file(os.path.join(data_dir, filename)) as fi:
            data = cPickle.load(fi)
        # Sanitize data...  if a cluster is empty, it'll show up as an int == 0
        int_test = [ type(datum['pca'])!=type(1) for datum in data ]
        valid_clusters = find(int_test)
        data = [ data[ind] for ind in valid_clusters ]
        n_features = unique_or_error([d['pca'].shape[1] for d in data])
        n_clusters = len(valid_clusters)


        # filenames
        fetfilename = os.path.join(data_dir, basename + '.fet.%d' % tetrode)
        clufilename = os.path.join(data_dir, basename + '.clu.%d' % tetrode)
        spkfilename = os.path.join(data_dir, basename + '.spk.%d' % tetrode)

        # write fetfile
        with file(fetfilename, 'w') as fetfile:
            # Write an extra feature for the time
            fetfile.write('%d\n' % (n_features+1))
            
            # Write one cluster at a time
            for d in data:
                to_write = np.hstack([d['pca'], 
                    np.rint(d['peaks'][:, None] * f_samp).astype(np.int)])    
                fmt = ['%f'] * n_features + ['%d']
                np.savetxt(fetfile, to_write, fmt=fmt)

        # write clufile
        with file(clufilename, 'w') as clufile:
            clufile.write('%d\n' % n_clusters)
            for n, d in enumerate(data):
                np.savetxt(clufile, valid_clusters[n] * 
                    np.ones(len(d['peaks']), dtype=np.int), fmt='%d')

        # write spkfile
        with file(spkfilename, 'w') as spkfile:
            
            clst_wvs = [0]*len(data)
            for ii, clst in enumerate(data):
                # This gets each waveform into the format needed for the KlustaKwik
                # spike file format
                cl_spks = np.concatenate( [ np.reshape( np.reshape( wvform, \
                    (4, len(wvform)/4)), len(wvform), order ='F') \
                    for wvform in clst['waveforms'] ] )
                
                clst_wvs[ii] = (cl_spks/(8192.0/2.**16)).astype(np.int16)
                
            spks = np.concatenate(clst_wvs)
            spks.tofile(spkfile)


