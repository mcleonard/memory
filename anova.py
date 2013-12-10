""" Module for ANOVA and variance explained analysis. """

from collections import defaultdict

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def run_anova(formula, df):
    lm = ols(formula, df).fit()
    anova = anova_lm(lm, type=3)
    return lm, anova

def fraction_of_explainable_variance(factor, anova):
    factors = [ each for each in anova.index if each != 'Residual' ]
    num = anova.ix[factor]['sum_sq']
    denom = anova.ix[factors]['sum_sq'].sum()
    return num/denom

def variance_explained(factor, anova):
    
    return anova.ix[factor]['sum_sq']/anova['sum_sq'].sum()

def variance_vectors(activity, factor_data, formula, var_func=variance_explained):
    results = defaultdict(list)
    for ii, (col, series) in enumerate(activity.iteritems()):
        factor_data['rate'] = series.astype(float)
        lm, anova = run_anova(formula, factor_data)
        factors = [each for each in anova.index if each != 'Residual']
        for factor in factors:
            results[factor].append(var_func(factor, anova))
            results['P({})'.format(factor)].append(anova.ix[factor]['PR(>F)'])
    return pd.DataFrame(results)