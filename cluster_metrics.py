"""  A module containing functions that calculate metrics for testing the 
quality of clustered data.

Load data from KlustaKwik files using the load_spikes() function.
Get false positive and false negative estimates using metrics()

Author:  Mat Leonard
Last modified: 8/2/2012
"""

import numpy as np

def load_spikes(data_dir, group, samp_rate, n_samp, n_chan):
    ''' This function takes the feature, cluster, and spike files in KlustaKwik format 
    and pulls out the features, spike times, spike waveforms for each cluster.
   
    Arguments
    ---------
    data_dir : path to the directory with the KlustaKwik files
    group : the group number you want to load
    samp_rate : the sampling rate of the recording in samples per second
    n_samp : number of samples for each stored spike in the spike file
    n_chan : number of channels stored in the spike file
    
    Returns
    -------
    out : dict of numpy structured arrays 
        A dictionary of the clusters.  The keys are the cluster
        numbers.  The values are numpy structured arrays with fields
        'times', 'waveforms', and 'pca' which give the timestamp,
        tetrode waveform, and pca reduced values, respectively, for each
        spike in the cluster.
        
    '''
    
    from KKFileSchema import KKFileSchema
    import os
    import kkio
    
    # Get the clustered data from Klustakwik files
    kfs = KKFileSchema.coerce(data_dir)
    
    # Get the spike features, time stamps, cluster labels, and waveforms
    feat = kkio.read_fetfile(kfs.fetfiles[group])
    features = feat.values[:,:-1]
    time_stamps = feat.time.values
    cluster_labels = kkio.read_clufile(kfs.clufiles[group])
    spikes = kkio.read_spkfile(kfs.spkfiles[group])
    
    # Reshaping the spike waveforms into a useful form
    spikes = spikes.reshape((len(spikes)/(n_chan*n_samp), (n_chan*n_samp)))
    for ii, spike in enumerate(spikes):
        spikes[ii] = spike.reshape((n_chan, n_samp), order = 'F').reshape(n_chan*n_samp)
    INT_TO_VOLT = 4096.0 / 2.0**15 # uV per bit
    spikes = spikes*INT_TO_VOLT
    
    cluster_ids = np.unique(cluster_labels.values)
    cluster_indices = { cid : np.where(cluster_labels.values == cid)[0] 
                        for cid in cluster_ids }
    
    clusters = dict.fromkeys(cluster_ids)
    dtypes = [('times','f8'), ('waveforms', 'f8', n_chan*n_samp), 
              ('pca', 'f8', len(features[0])) ]
    for cid, indices in cluster_indices.iteritems():
        clusters[cid] = np.zeros(len(indices), dtype = dtypes)
        clusters[cid]['times'] = time_stamps[indices]/np.float(samp_rate)
        clusters[cid]['waveforms'] = spikes[indices]
        clusters[cid]['pca'] = features[indices]
        clusters[cid].sort(order = ['times'])
    
    return clusters

def refractory(clusters, t_ref, t_cen, t_exp):
    ''' Returns an estimation of false positives in a cluster based on the
    number of refractory period violations.  Returns NaN if an estimate can't
    be made, or if the false positive estimate is greater that 0.5.  This is 
    most likely because of too many spikes in the refractory period.
    Unfortunately, this estimate is very sensitive to the number of spikes
    in a cluster, so for low spike counts, it will give you really high
    estimates.
    
    Parameters
    -----------------------------------------
    clusters : A dictionary of the clusters.  The keys are the cluster
        numbers.  The values are numpy structured arrays with fields
        'times', 'waveforms', and 'pca' which give the timestamp,
        tetrode waveform, and pca reduced values, respectively, for each
        spike in the cluster.  Get this from load_spikes().
    t_ref : the width of the refractory period, in seconds
    t_cen : the width of the censored period, in seconds
    t_exp : the total length of the experiment, in seconds
    
    This algorithm follows from D.N. Hill, et al., J. Neuroscience, 2011
    '''
    false_pos = dict.fromkeys(clusters.keys())
    for cid, cluster in clusters.iteritems():
        times = cluster['times']
        # Find the time difference between consecutive peaks
        isi = np.array([ jj - ii for ii, jj in zip(times[:-1], times[1:]) ])
        # Refractory period violations
        ref = np.sum(isi <= t_ref)
        if ref == 0:
            false_pos[cid] = 0
            continue
        
        N = float(len(times))
        # This part calculates the false positive estimate (See the paper)
        long_thing = ref*t_exp/(t_ref-t_cen)/(2.0*N*N)
        false_pos[cid] =(1 - np.sqrt(1-4*long_thing))/2.0
    
    return false_pos

def threshold(clusters, thresh):
    ''' Returns the rate of false negatives caused by spikes below the 
    detection threshold
    
    Arguments
    ---------
    clusters : A dictionary of the clusters.  The keys are the cluster
        numbers.  The values are numpy structured arrays with fields
        'times', 'waveforms', and 'pca' which give the timestamp,
        tetrode waveform, and pca reduced values, respectively, for each
        spike in the cluster.  Get this from load_spikes().
    thresh : detection threshold used for spike sorting
    
    Returns
    -------
    false_neg : dictionary
        false negative percentages for each cluster
    '''
    from scipy.special import erf
    false_neg = dict.fromkeys(clusters.keys())
    for cid, cluster in clusters.iteritems():
        spikes = cluster['waveforms']
        heights = np.min(spikes, axis=1)
        mu = np.mean(heights)
        sigma = np.std(heights)
        # The false negative percentage is the cumulative Gaussian function
        # up to the detection threshold
        cdf = 0.5*(1 + erf((thresh - mu)/np.sqrt(2*sigma**2)))
        false_neg[cid] = (1-cdf)

    return false_neg
    
def overlap(clusters, ignore = [0]):
    ''' Okay, so we are going to calculate the false positives and negatives
    due to overlap between clusters.
    
    This function is a little untrustworthy unless the clusters are pretty
    well behaved.  It calculates the false positives and negatives by
    fitting a gaussian mixture model with two classes to each cluster
    pair.  Then the false positives and negatives between those two clusters
    are calculated.  If you have very non-gaussian shaped clusters, this 
    probably won't work very well.  The work around will be to ignore these
    bad clusters.
    
    Arguments
    ---------
    clusters : A dictionary of the clusters.  The keys are the cluster
        numbers.  The values are numpy structured arrays with fields
        'times', 'waveforms', and 'pca' which give the timestamp,
        tetrode waveform, and pca reduced values, respectively, for each
        spike in the cluster.  Get this from load_spikes().
    ignore: a list of the clusters to ignore in the analysis.  Default is cluster
        zero since that is typically the noise cluster.
    
    Returns
    -------
    false_pos : dictionary
        false positive estimate for each cluster
    false_neg : dictionary
        false negative estimate for each cluster
        
    '''
    from sklearn import mixture
    from itertools import combinations
    from collections import defaultdict
    keys = clusters.keys()
    false_pos = defaultdict(list)
    false_neg = defaultdict(list)
    # This part is going to take out clusters we want to ignore
    popped = [ keys.pop(keys.index(ig)) for ig in ignore 
                                        if ig in keys ]
    # We're going to need to fit models to all pairwise combinations
    # If the first cluster is a noise cluster, ignore it
    pairs = combinations(keys, 2)
    for k, i in pairs:
        features = {k:clusters[k]['pca'], i:clusters[i]['pca']}
        gmm = mixture.GMM(n_components = 2, covariance_type = 'full')
        data = np.concatenate((features[k], features[i]))
        gmm.fit(data)
        gmm.init_params = ''
        while not gmm.converged_:
            gmm.fit(data)
        
        # Calculate the false positives and negatives for each cluster
        # For each pair, there is one model, but four values we can get
        N_k = np.float(len(features[k]))
        N_i = np.float(len(features[i]))

        # This calculates the average probability (over k) that a spike in 
        # cluster k belongs to cluster i - false positives
        f_p_k_i = 1/N_k*np.min(np.sum(gmm.predict_proba(features[k]), axis=0))
        
        # This calculates the average probability (over k)  that a spike in 
        # cluster i belongs to cluster k - false negatives
        f_n_k_i = 1/N_k*np.min(np.sum(gmm.predict_proba(features[i]), axis=0))
        
        # This calculates the average probability (over i) that a spike in 
        # cluster i belongs to cluster k - false positives
        f_p_i_k = 1/N_i*np.min(np.sum(gmm.predict_proba(features[i]), axis=0))
        
        # This calculates the average probability (over i) that a spike in 
        # cluster k belongs to cluster i - false negatives
        f_n_i_k = 1/N_i*np.min(np.sum(gmm.predict_proba(features[k]), axis=0))
        
        false_pos[k].append(f_p_k_i)
        false_pos[i].append(f_p_i_k)
        false_neg[k].append(f_n_k_i)
        false_neg[i].append(f_n_i_k)
    
    # Sum over all values from pairs of clusters
    false_pos_sum = { cid:np.sum(false_pos[cid]) for cid in clusters.keys() }
    false_neg_sum = { cid:np.sum(false_neg[cid]) for cid in clusters.keys() }

    return false_pos_sum, false_neg_sum

def censored(clusters, t_cen, t_exp):
    ''' Returns the estimated false negative rate caused by spikes censored
    after a detected spike
    
    Parameters
    -----------------------------------------
    clusters : A dictionary of the clusters.  The keys are the cluster
        numbers.  The values are numpy structured arrays with fields
        'times', 'waveforms', and 'pca' which give the timestamp,
        tetrode waveform, and pca reduced values, respectively, for each
        spike in the cluster.  Get this from load_spikes().
    t_cen : the width of the censored period, in seconds
    t_exp : the total length of the experiment, in seconds
    '''
    
    N = np.sum([len(cluster['times']) for cluster in clusters.itervalues()])
    false_neg = dict.fromkeys(clusters.keys())
    for cid, cluster in clusters.iteritems():
        times = cluster['times']
        false_neg[cid] = (N - len(times)) * t_cen / t_exp
    
    return false_neg

def metrics(clusters, thresh, t_ref, t_cen, t_exp, ignore = [0], estimates = 'roct'):
    ''' This function runs all the metrics calculations and sums them
    
    Parameters
    -----------------------------------------
    clusters : A dictionary of the clusters.  The keys are the cluster
        numbers.  The values are numpy structured arrays with fields
        'times', 'waveforms', and 'pca' which give the timestamp,
        tetrode waveform, and pca reduced values, respectively, for each
        spike in the cluster.  Get this from load_spikes() or from 
        ePhys.Sorter.clusters.
    thresh : detection threshold used for spike sorting
    t_ref : the width of the refractory period, in seconds
    t_cen : the width of the censored period, in seconds
    t_exp : the total length of the experiment, in seconds
    ignore : a list of the clusters to ignore in the analysis.  Default is cluster
        zero since that is typically the noise cluster.
    estimates : a string of estimates to include
        'r' for refractory
        'o' for overlap
        'c' for censored
        't' for thresholding
        So 'roct' does them all, 'ro' only does refractory and censored,
            'ct' does consored and thresholding, and so on.
    
    Returns
    -----------------------------------------
    f_p : dictionary of false positive estimates for each cluster
        if an estimate can't be made, will be NaN.  This means that the 
        estimate is >50%
    f_n : dictionary of false negative estimates for each cluster
    
    '''
    import pprint as pp # Pretty print!
    from collections import defaultdict
    all_estimates = 'roct'
    false_pos_est = { est:defaultdict(int) for est in all_estimates }
    false_neg_est = { est:defaultdict(int) for est in all_estimates }
    
    if 'r' in estimates:
        f_p_r = refractory(clusters, t_ref, t_cen, t_exp)
        print "Refractory violation false positive estimate:"
        pp.pprint(f_p_r)
        false_pos_est.update({'r':f_p_r})
    if 't' in estimates:
        f_n_t = threshold(clusters, thresh)
        print "Thresholding false negatives estimate:"
        pp.pprint(f_n_t)
        false_neg_est.update({'t':f_n_t})
    if 'o' in estimates:
        f_p_o, f_n_o = overlap(clusters, ignore)
        print "Overlap false positives and negatives estimate:"
        pp.pprint(f_p_o)
        pp.pprint(f_n_o)
        false_pos_est.update({'o':f_p_o})
        false_neg_est.update({'o':f_n_o})
    if 'c' in estimates:
        f_n_c = censored(clusters, t_cen, t_exp)
        print "Censored false negatives estimate:"
        pp.pprint(f_n_c)
        false_neg_est.update({'c':f_n_c})
    
    
    f_p = dict.fromkeys(clusters.keys())
    f_n = dict.fromkeys(clusters.keys())
    for cid in clusters.iterkeys():
        if np.isnan(false_pos_est['r'][cid]):
            f_p[cid] = np.nan
        else:
            f_p[cid] = np.sum([false_pos_est[est][cid] for est in estimates])
            
        if false_neg_est['o'][cid] == None:
            f_n[cid] = np.nan
        else:
            f_n[cid] = np.sum([false_neg_est[est][cid] for est in estimates])
    
    print "Summing everything up"
    return f_p, f_n
    
def batch_metrics(unit_list, threshold, t_ref, t_cen):
    ''' This here function runs metrics on a batch of data.  Pass in units from the catalog.
    '''
    
    from scipy import unique
    
    samp_rate = 30000.
    n_samples = 30
    n_chans = 4
    
    # Find common Sessions
    sessions = unique([unit.session for unit in unit_list])
    
    for session in sessions:
        
        units = session.units
        tetrodes = unique([unit.tetrode for unit in units])
        
        for tetrode in tetrodes:
            data = load_spikes(session.path, tetrode,  samp_rate, n_samples, n_chans)
            f_p, f_n = metrics(data, threshold, t_ref, t_cen, session.duration)
            # Doing this because sometimes there is no cluster 0 sometimes
            f_p.setdefault(1)
            f_n.setdefault(1)
            units = [ unit for unit in session.units if unit.tetrode == tetrode] 
            for unit in units:
                unit.falsePositive = f_p[unit.cluster]
                unit.falseNegative = f_n[unit.cluster]
    