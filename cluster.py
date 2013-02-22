''' This module is used to load neural recordings, detect spikes, and
    sort the spikes into clusters.
    
    Functions:
    load_data(filename, tetrode = _, cluster = _)
        Uses the ns5 module to load and return eletrode voltage signals
        as a numpy array, with each channel as a row.
    
    detect_spikes(data, sigs, low_f, high_f):
        Used to detect spikes from the electrode signals.  Returns a numpy
        structured array with fields 'times' and 'waveforms', the timestamps
        and tetrode waveforms, respectively, of each detected spike.     
    
    Classes:
    Sorter: Used to sort spikes into clusters with a Gaussian mixture model
        Arguments
        ---------
        K : int : default = 9
            Number of clusters to fit to the data
        pca_d : int : default = 10
            Number of PCA components to use in the sorting
        
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from collections import defaultdict

class StateError(Exception):
    def __init__(self, *args, **kwargs):
        super(StateError, self).__init__(*args, **kwargs)

class Viewer(object):
    ''' An object for viewing clusters.
    
        Arguments
        ---------
        clusters : A dictionary of the clusters.  The keys are the cluster
            numbers.  The values are numpy structured arrays with fields
            'times', 'waveforms', and 'pca' which give the timestamp,
            tetrode waveform, and pca reduced values, respectively, for each
            spike in the cluster. 
        
        Methods
        ------------
        scatter() : returns self
                Plots a cluster scatter plot
           
        waveforms(clusters) : returns self
            Plots cluster waveforms
        
        autocorrs(clusters, bin_width, limit) : returns self
            Plots cluster autocorrelations
        
        crosscorrs() : returns self
            Plots cluster crosscorrelations
        
        views(cross = False) : returns self
            Calls scatter, waveforms, and autocorrs
            Set cross = True to also calls crosscorrs
    '''
    
    def __init__(self, clustered_data = None):
        if isinstance(clustered_data, basestring):
            self.clusters = _load_clusters(clustered_data)
        else:
            self.clusters = clustered_data
        self._setColors(len(clustered_data))
        self._xcache = defaultdict(dict) # Cache for xcorr data
        self._acache = defaultdict(dict) # Cache for acorr data
    
    def scatter(self, clusters = None, components = [1,2,3]):
        ''' Make a scatter plot of the waveforms projected on the PCA components,
            colored by cluster.  
            
            Returns self.
            
            Arguments
            ---------
            clusters : list
                List of clusters to plot.  Defaults to None, plots all clusters if
                left as None.
            components : list
                List of PCA components to use
        
        '''
        from itertools import combinations
        clusters = self._sanitizeClusters(clusters)
        # We say the 1st PCA component, but it is the 0th column in the data
        components = np.array(components) - 1
        
        fig, subplots = _createSubplots(1, _binomial(len(components),2))
        # This is an iterator over all pairs of components
        itercomps = combinations(components,2)
        for ii, (comp_1, comp_2) in enumerate(itercomps):
            ax = subplots[ii]
            for k in clusters:
                pca = self.clusters[k]['pca']
                ax.scatter(pca[:,comp_1], pca[:,comp_2], 
                           marker = '.', s = 10,
                           facecolor = self._colors[k],
                           edgecolor = 'face')
                ax.set_xlabel('PCA component %d' % comp_1)
                ax.set_ylabel('PCA component %d' % comp_2)
        
        fig.tight_layout()
        fig.show()
        
        return self
    
    def views(self, views = 'csa', clusters = None):
        ''' Calls viewing methods.  
            
            Returns self.
            
            Arguments
            ---------
            plots : string
                String of views you want. 'c' = scatter, 's' = spikes, 
                    'a' = acorrs, 'x' = xcorrs.  So, views = 'csa' gives you 
                    scatter, spikes(), and acorrs()
            clusters : list
                List of clusters to plot. Plots all clusters if left as None.
        
        '''
        clusters = self._sanitizeClusters(clusters)
        
        view_methods = {'c':self.scatter,'s':self.spikes,
                        'a':self.acorrs,'x':self.xcorrs}
        
        for view in views:
            view_methods[view](clusters)

        return self
        
    def spikes(self, clusters = None):
        ''' Plots spikes waveforms for clusters.
        
            Returns self.
        
            Arguments
            ---------
            clusters : list
                List of clusters to plot.  Defaults to None, plots all clusters if
                left as None.
        
        '''
        from numpy.random import randint

        clusters = self._sanitizeClusters(clusters)
        fig, subplots = _createSubplots(2, len(clusters))
        
        # The waveforms are going to be separated into slices this long
        SLICE = 30 
        N_SAMPLES = 100
        # Build the x values to use in the plots
        xs = np.concatenate([np.arange(SLICE)+ii*(SLICE+10) for ii in range(4)])
        # And build the slices we'll take from the waveforms
        slices = [(SLICE*ii, SLICE*(ii+1)) for ii in range(4)]
        
        waveforms = [ self.clusters[k]['waveforms'] for k in clusters ]
        means = [ np.mean(wf, axis=0) for wf in waveforms ]
        r_wfs = [ wf[randint(0,len(wf),N_SAMPLES)] for wf in waveforms ]
        colors = [ self._colors[k] for k in clusters ]
        
        def wf_plot(ax, waveforms, color):
            if len(np.shape(waveforms)) == 2:
                plots = [ ax.plot(xs[start:end], waveforms[:,start:end].T, 
                                  color = color, alpha = 0.3)
                          for start, end in slices ]
            elif len(np.shape(waveforms)) == 1:
                plots = [ ax.plot(xs[start:end], waveforms[start:end].T, 
                                  color = color)
                          for start, end in slices ]
                
        plot_var = [ wf_plot(ax, r_wf, c)
                     for ax, r_wf, c in zip(subplots, r_wfs, colors) ]
          
        plot_mean = [ wf_plot(ax, mean, 'k')
                      for ax, mean in zip(subplots, means) ]
        
        titles = [ ax.set_title('Cluster {}, N = {}'.format(k, len(wfs)))
                   for ax, k, wfs in zip(subplots, clusters, waveforms) ]
        
        xtl = [ ax.set_xticklabels('') for ax in subplots ]
        
        fig.tight_layout(h_pad=0.1, w_pad=0.1)
        fig.show()
        
        return self
        
    def acorrs(self, clusters = None, bin_width = 0.001, limit = 0.02):
        ''' Plots cluster autocorrelations.
        
            Returns self.
        
            Arguments
            ---------
            clusters : list
                List of clusters to plot.  Defaults to None, plots all clusters if
                left as None.
            
            bin_width : float
                Width of bins used in the autocorrelation
                Defaults to 0.001 seconds (1 ms)
            
            limit : float
                Range over which to calculate the autocorrelation
                Defaults to 0.02 seconds (20 ms)
        
        '''
        
        clusters = self._sanitizeClusters(clusters, hold_out=[0])
        corrs = {k:self._getCorrs((k,k,bin_width,limit), auto=True) 
                 for k in clusters if len(self.clusters[k]['times'])>0}
        fig, subplots = _createSubplots(3, len(corrs))
    
        plots = [ subplots[ii].bar(bins[:-1]*1000, counts, width = bin_width*1000, 
                  color = self._colors[k], edgecolor = 'none') 
                  for ii, (k, (counts, bins)) in enumerate(corrs.iteritems())]
    
        fig.tight_layout(h_pad=0.1, w_pad=0.1)
        fig.show()
        
        return self
    
    def xcorrs(self, clusters = None, bin_width = 0.001, limit = 0.02):
        ''' Plots cluster cross-correlations.
        
            Returns self.
        
            Arguments
            ---------
            clusters : list
                List of clusters to plot.  Defaults to None, plots all clusters if
                left as None.
            
            bin_width : float
                Width of bins used in the autocorrelation
                Defaults to 0.001 seconds (1 ms)
            
            limit : float
                Range over which to calculate the autocorrelation
                Defaults to 0.02 seconds (20 ms)
        
        '''
        clusters = self._sanitizeClusters(clusters, hold_out=[0])
    
        fig = plt.figure(4)
        plt.clf()
        # Set the number of rows and columns to plot
        n_rows = len(clusters)
        n_cols = len(clusters)
        
        # Build a list used to iterate through all the correlograms
        l_rows = np.arange(n_rows+1)
        l_cols = np.arange(n_cols)
        ind = [(x, y) for x in l_rows for y in l_cols if y >= x]
        
        for ii, jj in ind:
            ax = fig.add_subplot(n_rows, n_cols, ii*n_cols + jj + 1)
            params = (ii,jj,bin_width,limit)
            # If cluster 1 is different than cluster 2, get the crosscorrelation
            if ii != jj:
                counts, bins = self._getCorrs(params, auto=False)
                ax.bar(bins[:-1],counts, width = bin_width, color = 'k')
                ax.set_xticklabels('')
                ax.set_yticklabels('')
            
            # If cluster 1 is the same as cluster 2, get the autocorrelation
            else:
                counts, bins = self._getCorrs(params, auto=True)
                ax.bar(bins[:-1],counts, width = bin_width, 
                       color = self._colors[clusters[ii]], edgecolor = 'none')
                ax.set_xlabel(str(clusters[ii]))
                ax.set_ylabel(str(clusters[ii]))
                ax.set_xticklabels('')
                ax.set_yticklabels('')
        
        fig.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.5)
        fig.show()
        
        return self
    
    def _sanitizeClusters(self, clusters, hold_out=[]):
        if clusters == None:
            clusters = self.clusters.keys()
            removed = [ clusters.pop(hold) for hold in hold_out ]
            return clusters
        elif clusters == 'signal':
            return self._sanitize_clusters(hold_out=[0])
        else:
            valid_clusters = set(self.clusters.keys())
            clusters = set(clusters)
            invalid = clusters.difference(valid_clusters)
            if len(invalid) == 0:
                return clusters
            else:
                raise ValueError("Requested clusters that do not exist")
    
    def _setColors(self, max_K):
        ''' Helper method for setting the colors used when plotting clusters '''
        self._colors = plt.cm.Paired(np.arange(0.0,1.0,1.0/max_K))
    
    def _getCorrs(self, params, auto = True):
        ''' This method will return cluster correlations.  Adding in some
            caching so that the thing doesn't calculate the correlations
            everytime you want to see them.
        '''
        if auto == True:
            cache = self._acache[params]
        elif auto == False:
            cache = self._xcache[params]
        
        if params in cache:
            counts, bins = cache[params]
        else:
            ii, jj, bin_width, limit = params
            times = [ self.clusters[k]['times'] for k in self.clusters ]
            counts, bins = _correlogram(times[ii], times[jj], 
                    bin_width = bin_width, limit = limit, auto=auto)
            cache.update({params:(counts, bins)})
            
        return counts, bins
        
    def __len__(self):
            return len(self.clusters)
        
    def __repr__(self):
        return "<Viewer(K = %d)>" % len(self)
    
class Sorter(Viewer):
    ''' This is an object that sorts spike waveforms into clusters.  
        Inherits from Viewer.
        
        Arguments
        ---------
        K : int : default = 9
            Number of clusters to fit to the data
        pca_d : int : default = 10
            Number of dimensions to use for PCA decomposition
        
        Methods
        -------
        sort(data) : returns self
            data should be an numpy structured array with fields 'times' and 
            'waveforms', containing timestamps and spike waveforms.
            This fits a Gaussian mixture model to the data and groups
            similar waveforms into clusters.
        
        combine(source, destination) : returns self
            Combines source cluster into destination cluster
        
        split(source) : returns self
            Splits source cluster into two clusters
        
        autosplit() : returns self
            Splits any cluster with >7000 spikes into clusters with ~4000 spikes
        
        scatter() : returns self
            Plots a cluster scatter plot
       
        waveforms(clusters) : returns self
            Plots cluster waveforms
        
        autocorrs(clusters, bin_width, limit) : returns self
            Plots cluster autocorrelations
        
        crosscorrs() : returns self
            Plots cluster crosscorrelations
        
        views(cross = False) : returns self
            Calls scatter, waveforms, and autocorrs
            Set cross = True to also calls crosscorrs
        
        Attributes
        ----------
        K : Number of clusters fit to the data
        dimensions : Number of PCA components
        state : State of the sorter.  If 'resorted', things like gmm attribute
            won't match the actual clusters any more.
        gmm : The gaussian mixture model used for the initial sorting.  Not
            really valid once the data is resorted, or any clusters are 
            combined or split.
        pca : the pca object used to reduce the detected spikes.
        variance_captured : the variance captured by the number of PCA 
            dimensions used to reduce the spikes. I like to keep it around 80%
        bic : Bayesian information criteria of the initial model.
            Smaller values mean the model is more likely given the data.
        clusters : A dictionary of the clusters.  The keys are the cluster
            numbers.  The values are numpy structured arrays with fields
            'times', 'waveforms', and 'pca' which give the timestamp,
            tetrode waveform, and pca reduced values, respectively, for each
            spike in the cluster.
        
    '''
    
    def __init__(self, K = 9, pca_d = 10):
        self.K = K # Number of clusters to fit to the data
        self.dimensions = pca_d # Number of PCA components used
        self.gmm = mixture.GMM(n_components = K,
                                                covariance_type='full')
        self._setColors(self.K+1)
        self.state = 'initialized'
        self._prob_cutoff = 0.5
        self._xcache = defaultdict(dict)
        self._acache = defaultdict(dict)
        self._dirty = set()
    
    def sort(self, data, fit_once = False):
        ''' Sorts spike data into clusters
        
        Arguments
        ---------
        data : numpy structured array
            data should be a numpy array with fields 'times' and 
            'waveforms', containing timestamps and spike waveforms.
            You can get this from the detect_spikes function.
        
        '''
        prob_cutoff = self._prob_cutoff
        self.pca = _pca(data['waveforms'], self.dimensions)
        reduced = self.pca.transform(data['waveforms'])
        self.variance_captured = np.sum(self.pca.explained_variance_ratio_)
        x = reduced - np.mean(reduced, axis=0)
        
        # Store our data
        alldata = np.zeros(len(data),
                           dtype = [('times','f8'), ('waveforms','f8',120),
                                    ('pca','f8',self.pca.n_components)])
        alldata['times'] = data['times']
        alldata['waveforms'] = data['waveforms']
        alldata['pca'] = x
        self._data = alldata
        
        # Fit the model
        self.gmm.fit(x)
        while not self.gmm.converged_ | fit_once:
            self.gmm.init_params = ''
            self.gmm.fit(x)
        else:
            print 'Model with %d components converged' % self.K
            self.bic = self.gmm.bic(x)
            self.clusters = dict.fromkeys(range(1, self.K+1))
            probs = self.gmm.predict_proba(x)
            index, cluster_id = np.where(probs>=prob_cutoff)
            for k in self.clusters.iterkeys():
                cluster = index[np.where(cluster_id == k-1)]
                self.clusters[k] = self._data[cluster]
            self.state = 'sorted'
        
        # Set the noise cluster
        noise = np.where(probs.max(axis=1)<prob_cutoff)
        self.clusters.update({0:self._data[noise]})
            
        self._cleanUp()
        
        return self
    
    def resort(self, K, hold_out = [0] ):
        ''' Resort the data except for clusters specified by hold_out 
        
            Returns self.
            
            Arguments
            ---------
            K : int
                Number of clusters to fit to the data
            hold_out : list of ints
                Clusters to leave out of te resorting data    
        '''
        
        prob_cutoff = self._prob_cutoff
        
        if (self.state == 'sorted') | (self.state == 'resorted'):
            pass
        else:
            raise StateError('Data must be sorted before resorting')
        
        if type(hold_out) == type([]):
            pass
        else:
            hold_out = list(hold_out)
            
        clusters_to_resort = [ k for k in self.clusters.iterkeys()
                                if k not in hold_out ]
        resorting = [ self.clusters.pop(k) for k in clusters_to_resort ]
        resorting = np.concatenate(resorting)
        
        x = resorting['pca']
        gmm = mixture.GMM(n_components = K, covariance_type='full')
        gmm.fit(x)
        gmm.init_params = ''
        while not gmm.converged_:
            gmm.fit(x)
        else:
            'Model with %d components converged' % K
            probs = gmm.predict_proba(x)
            index, cluster_id = np.where(probs>=prob_cutoff)
            new_ids = [ ii for ii in range(0,len(self)+K) 
                        if ii not in self.clusters ]
            for k in new_ids:
                cluster = index[np.where(cluster_id == k-1)]
                resorted = {k:resorting[cluster]}
                self.clusters.update(resorted)
            self.state = 'resorted'
        
        self._registerDirty(clusters_to_resort.append(new_ids))
        self._cleanUp()
            
        return self
        
    def combine(self, source, destination):
        ''' Combine source cluster into destination cluster 
            
            Arguments
            ---------
            source : int or list of ints
                Cluster number(s) for the cluster you want to move
            destination : int
                Cluster number for the cluster you are moving into
        '''
        
        if type(source) == type([]):
            for s in source:
                self.combine(s, destination)
            return self
            
        moving = self.clusters.pop(source)
        mover = np.concatenate((self.clusters[destination], moving))
        self.clusters.update({destination:mover})
        self.state = 'resorted'
        self._registerDirty([source, destinaton])
        self._cleanUp()
        
        return self
        
    def split(self, source, K=2):
        ''' Splits a cluster 
            
            Arguments
            ---------
            source : int
                The number of the cluster to split
            K : int
                Number of clusters the split into
        '''
        
        gmm = mixture.GMM(n_components = K, covariance_type='full')
        old = self.clusters.pop(source)
        waveforms = old['waveforms']
        new_pca = _pca(waveforms, self.dimensions).transform(waveforms)
        x = new_pca - np.mean(new_pca, axis=0)
        gmm.fit(x)
        self.gmm.init_params = ''
        while not self.gmm.converged_:
            self.gmm.fit(x)
        new_sorting = gmm.predict(x)
        new_ids = [ ii for ii in range(0,len(self)+K)
                        if ii not in self.clusters ]
        for ii, id in enumerate(new_ids):
            self.clusters.update({id:old[np.where(new_sorting == ii)]})
        self.state = 'resorted'
        
        self._registerDirty(new_ids.append(source))
        self._cleanUp()
        
        return self
    
    def autosplit(self):
        ''' Automatically splits clusters into clusters with ~4000 spikes 
            
            Returns self
        
        '''
        to_split = {}
        # Doing this in two loops because it is bad to iterate through
        # a dictionary (self.clusters) while modifying it
        for id, cluster in self.clusters.iteritems():
            if len(cluster) > 7000:
                N = np.ceil(len(cluster)/4000.0)
                to_split.update({id:int(N)})
        for id, N in to_split.iteritems():
            self.split(id,K=N)
        
        return self
    
    def outliers(self):
        ''' This method detects and removes outliers from each cluster.
        
        Outlier removal is done following the method in Hill D.N., et al., 2011
        
        '''     
        inv = np.matrix.inv
        for k, cluster in self.clusters.iteritems():
            
            if k != 0: # Cluster 0 is the noise cluster
                pass
            else:
                continue
            
            spikes = cluster['waveforms']
            mean = spikes.mean(axis = 0)
            invcov = np.matrix(inv(np.cov(spikes.T)))
            diff = np.matrix(spikes - mean)
            # Calculate the chi^2 values for each data point
            chi2 = np.array([ (vec*invcov*vec.T).A1[0] for vec in diff ])
            
            outliers = np.where(chi2 < 1/float(len(diff)))[0]
            inliers = np.where(chi2 > 1/float(len(diff)))[0]
            
            if outliers:
                # Put outliers in the noise cluster
                self.clusters[0] = np.concatenate(self.clusters[0],
                                                  cluster[outliers])
                self.clusters[k] = cluster[inliers]
        
        
        self._registerDirty(self.clusters.keys())
        self._cleanUp()
        
        return self
    
    def saveClusters(directory, rat_name, date, tetrode):
        import cPickle as pkl
        from os import path
        filename = "%s_%s.cls.%d" % (rat_name, date, tetrode)
        filepath = path.join(directory, filename)
        with open(filepath,'w') as f:
            pkl.dump(self.clusters,f)
        print 'Clusters data saved to %s' % filepath
        
    def _cleanUp(self):
        ''' Runs cleaning methods and sets the colors '''
        self._cleanEmpty()
        self._cleanCache()
        max_K = np.max(self.clusters.keys())+1
        self._setColors(max_K)
        
    def _cleanEmpty(self):
        ''' Removes empty clusters '''
        to_remove = [ key for key, value in self.clusters.iteritems()
                        if len(value) == 0 ]
        if 0 in to_remove:
            to_remove.remove(0)
        removed = [ self.clusters.pop(k) for k in to_remove ]
    
    def _registerDirty(self, clusters):
        if np.iterable(clusters):
            pass
        else:
            clusters = [clusters]
        
        self._dirty.update(clusters)
    
    def _cleanCache(self):
        ''' Cleans the cache by removing any entries involving clusters 
            added or removed.  This way we don't plot old clusters
            instead of new clusters.
        '''
        dirty_K = self._dirty
        for cache in [self._acache, self._xcache]:
            keys = cache.keys()
            to_pop = [ key for key in keys for K in dirty_K if K in key[:2] ]
            popped = [ cache.pop(key) for key in to_pop ]
        self._dirty = set()
    
    def __len__(self):
        
        return len(self.clusters)
        
    def __repr__(self):
        
        return "<Sorter(K = %d, d = %d)>" % (len(self), self.dimensions)
    
    def __reduce__(self):
        
        return (Sorter, (self.K,self.dimensions), self.__dict__)
        
def load_data(filename, tetrode = None, channels = None):
    '''  Returns a numpy array with rows being channel data '''
    import ns5
    loader = ns5.Loader(filename)
    loader.load_file()
    tetrodes = {1:[16,18,17,20],2:[19,22,21,24],3:[23,26,25,28],\
        4:[27,30,29,32]}
        
    if tetrode:
        chans = tetrodes[tetrode]
    else:
        chans = channels
        
    data = np.array([ loader.get_channel_as_array(chan) for chan in chans ])
    
    return data

def detect_spikes(rawdata, sigs = 4, detect_hf = 2000, extract_hf = 6000):
    '''  Takes the raw channel data and returns a numpy structured array.
        The array has two fields, 'times' and 'waveforms', the timestamp
        of the spikes and the tetrode waveforms of the spikes, respectively.
    '''
    # I don't think I can make loading the data any faster, but I can
    # definitely make this faster using multiple processes.  It shouldn't be
    # too hard.  First I'll need to profile this to see where it is the slowest.
    samp_rate = 30000.0
    
    # First step is to convert to voltage
    INT_TO_VOLT = 4096.0 / 2.0**15
    rawdata = rawdata * INT_TO_VOLT
    
    # TODO: Spread this out over 4 processes
    detection = np.array([_filter(chan, low = 300, high = detect_hf)
                         for chan in rawdata ])
    
    # Now, get the threshold crossings
    # TODO: Spread this out over 4 processes
    threshold = np.max([ _medthresh(chan, sigs) for chan in detection ])
    print '%s uV threshold' % threshold
    # TODO: Try to parallelize this
    peaks = _crossings(detection, threshold)
    peaks.sort()
    peaks = _censor(peaks,30)
    # TODO: Spread this out over 4 processes
    extraction = np.array([_filter(chan, low = 300, high = extract_hf)
                         for chan in rawdata ])
    # And group into tetrode waveforms
    waveforms = np.array([ _get_tetrode_waveform(extraction, peak) 
                            for peak in peaks ])
    spikes = np.zeros(len(peaks), 
                    dtype=[('times','f8'), ('waveforms', 'f8', 120)])
    spikes['times'] = peaks/samp_rate
    spikes['waveforms'] = waveforms
    
    # Going to remove bad waveforms here
    # Only include waveforms that are below some upper threshold
    # and above some lower threshold
    POS_THRESH = 300
    good = np.where(spikes['waveforms'].max(axis=1)<POS_THRESH)
    spikes = spikes[good]
    NEG_THRESH = -500
    good = np.where(spikes['waveforms'].min(axis=1)>NEG_THRESH)
    spikes = spikes[good]
    
    print "%d spikes detected" % len(spikes)
    
    return spikes        

def bestmodel(spikes, min_K=5, max_K=15, pca_d = 10, processes=2):
    from multiprocessing import Process, Queue
    
    # Creating this function to send to a Process
    # Basically, it puts a Sorter into the queue
    def f(q, K):
        q.put(Sorter(K, pca_d).sort(spikes, fit_once=True))
    
    Ks = range(min_K, max_K+1)
    
    slices = zip(range(0,len(Ks)+1,processes)[:-1],
                 range(0,len(Ks)+1,processes)[1:])
    left_over = len(Ks)%processes
    if left_over != 0:
        slices.append((len(Ks)-left_over,len(Ks)))
    sorters = []
    for x, y in slices:
        sorter_queue = Queue()
        jobs = [ Process(target=f, args=(sorter_queue,K))
                 for K in Ks[x:y] ]
        for job in jobs: job.start()
        sorters.extend([ sorter_queue.get() for ii in Ks[x:y] ])
        for job in jobs: job.join(timeout=30)
        for job in jobs: job.terminate()
        for job in jobs: job.join(timeout=30)
    
    sorter_queue.close()
    
    # Now we're choosing the best model based on lowest BIC
    bics = np.array([ sorter.bic for sorter in sorters ])
    best_sorter = sorters[bics.argmin()]
    
    print "Best model is {}".format(str(best_sorter))
    return best_sorter

def _load_clusters(filepath):
    import cPickle as pkl
    with open(filepath, 'r') as f:
        clusters = pkl.load(f)
    return clusters

def _createSubplots(fig_num, K):
    
    fig = plt.figure(fig_num)
    plt.clf()
    n_axes = K
    n_rows = (n_axes+2)//3
    n_cols = 3
    if n_axes < 3: n_cols = n_axes
        
    subplots = [fig.add_subplot(n_rows, n_cols, ii+1) for ii in range(n_axes)]
    
    return fig, subplots

def _correlogram(t1, t2=None, bin_width=.001, limit=.02, auto=False):
    
    """Return crosscorrelogram of two spike trains.
    
    Essentially, this algorithm subtracts each spike time in `t1` 
    from all of `t2` and bins the results with numpy.histogram, though
    several tweaks were made for efficiency.
    
    Arguments
    ---------
        t1 : first spiketrain, raw spike times in seconds.
        t2 : second spiketrain, raw spike times in seconds.
        bin_width : width of each bar in histogram in sec
        limit : positive and negative extent of histogram, in seconds
        auto : if True, then returns autocorrelogram of `t1` and in
            this case `t2` can be None.
    
    Returns
    -------
        (count, bins) : a tuple containing the bin edges (in seconds) and the
        count of spikes in each bin.

        `bins` is relative to `t1`. That is, if `t1` leads `t2`, then
        `count` will peak in a positive time bin.
    """
    # For auto-CCGs, make sure we use the same exact values
    # Otherwise numerical issues may arise when we compensate for zeros later
    if auto: t2 = t1

    # For efficiency, `t1` should be no longer than `t2`
    swap_args = False
    if len(t1) > len(t2):
        swap_args = True
        t1, t2 = t2, t1

    # Sort both arguments (this takes negligible time)
    t1 = np.sort(t1)
    t2 = np.sort(t2)

    # Determine the bin edges for the histogram
    # Later we will rely on the symmetry of `bins` for undoing `swap_args`
    limit = float(limit)
    bins = np.linspace(-limit, limit, num=(2 * limit/bin_width + 1))

    # This is the old way to calculate bin edges. I think it is more
    # sensitive to numerical error. The new way may slightly change the
    # way that spikes near the bin edges are assigned.
    #bins = np.arange(-limit, limit + bin_width, bin_width)

    # Determine the indexes into `t2` that are relevant for each spike in `t1`
    ii2 = np.searchsorted(t2, t1 - limit)
    jj2 = np.searchsorted(t2, t1 + limit)

    # Concatenate the recentered spike times into a big array
    # We have excluded spikes outside of the histogram range to limit
    # memory use here.
    big = np.concatenate([t2[i:j] - t for t, i, j in zip(t1, ii2, jj2)])

    # Actually do the histogram. Note that calls to numpy.histogram are
    # expensive because it does not assume sorted data.
    count, bins = np.histogram(big, bins=bins)

    if auto:
        # Compensate for the peak at time zero that results in autocorrelations
        # by subtracting the total number of spikes from that bin. Note
        # possible numerical issue here because 0.0 may fall at a bin edge.
        c_temp, bins_temp = np.histogram([0.], bins=bins)
        bin_containing_zero = np.nonzero(c_temp)[0][0]
        count[bin_containing_zero] -= len(t1)

    # Finally compensate for the swapping of t1 and t2
    if swap_args:
        # Here we rely on being able to simply reverse `counts`. This is only
        # possible because of the way `bins` was defined (bins = -bins[::-1])
        count = count[::-1]

    return count, bins

def _binomial(a,b):
    from scipy.special import gamma
    return int(gamma(a+1)/(gamma(b+1)*gamma(a-b+1)))
def _medthresh(data,threshold = 4):
    """ A function that calculates the spike crossing threshold 
        based off the median value of the data.
    
    Arguments
    ---------
    data : your data
    threshold : the threshold multiplier
    """
    return -threshold*np.median(np.abs(data)/0.6745)

def _filter(data, low = 300, high = 6000):
    import scipy.signal as sig

    filter_lo = low #Hz
    filter_hi = high #Hz
    samp_rate = 30000.0
    
    #Take the frequency, and divide by the Nyquist freq
    norm_lo = filter_lo/(samp_rate/2)
    norm_hi = filter_hi/(samp_rate/2)
    
    # Generate a 3-pole Butterworth filter
    b, a = sig.butter(3,[norm_lo,norm_hi], btype="bandpass");
    return sig.filtfilt(b,a,data)

def _censor(data, censor):
    
    return data[np.where(np.diff(data)>=censor)[0]]
    
def _crossings(data, threshold):
    
    peaks = []
    for chan in data:
        
        below = np.where(chan<threshold)[0]
        peaks.append(below[np.where(np.diff(below)==1)])

    return np.concatenate(peaks)

def _get_tetrode_waveform(data, peak_sample):
        
    patch = data[:, peak_sample-15:peak_sample+15]
    patches = []
    for ii, p in enumerate(patch):
        minim = p.min()
        min_samp = np.where(p==minim)[0][0] + peak_sample - 15
        patches.append(data[ii, min_samp-10:min_samp+20])
    waveform = np.concatenate(patches)
    
    return waveform

def _pca(waveforms, components = 5):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=components)
    x = waveforms - np.mean(waveforms, axis=0)
    pca.fit(x)
    return pca

