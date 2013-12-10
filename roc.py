""" Module for ROC analysis. """

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapz
from functools import partial

class ROC(object):

    def __init__(self, data, classifier, points=30):
        self.data = data
        self.classifier = classifier
        self.points = points
        self.pos = self._calculate(points)

    def _calculate(self, points):

        cutoffs = np.linspace(self.data.min(), self.data.max(), points)
        predict = partial(predict_positives, self.data, self.classifier)
        positives = pd.DataFrame([ predict(cutoff) for cutoff in cutoffs ], 
                                  columns=['true', 'false'])
        return positives

    def plot(self, **kwargs):
        
        try:
            fig, ax = plt.subplots(kwargs.pop('figsize'))
        except KeyError:
            fig, ax = plt.subplots()
            
        ax.plot(self.pos['false'], self.pos['true'], **kwargs)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.plot([0,1], [0,1], '--', color='grey')
        
        return fig, ax

    def rerun(self):
        self.pos = self._calculate(self.points)
        return self

    def area(self):
        return trapz(self.pos['true'].values, x=self.pos['false'].values)
    
    def __repr__(self):
        return "ROC object"

def roc(data, classifier, points=30):
    return ROC(data, classifier, points)

def predict_positives(data, classifier, cutoff):
    predicted = data < cutoff
    t_p = predicted[classifier].sum()/float(len(predicted[classifier]))
    f_p = predicted[~classifier].sum()/float(len(predicted[~classifier]))
    return t_p, f_p

