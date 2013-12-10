""" Module for decoders.  """

import numpy as np
import pandas as pd
import pymc as mc
from scipy.stats import poisson

def split_data(data, fraction):
    """ Splits the data into training and testing sets. 

        **Arguments**
            *data* : the data you want to split
            *fraction* : fraction of the data in the training set

        **Returns**
            Two pandas Series of indices, first is training, second is testing.

    """
    permuted = np.random.permutation(data.index)
    train, test = permuted[:len(data)*fraction], permuted[len(data)*(1-fraction)]

    return train, test

@np.vectorize
def logistic(x, beta, alpha=0):
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))

def logistic_decoder(rates, goals, samples=30000, burn=10000, thin=5):
    """ Train a decoder using a logistic function. 

        **Returns**
            Arrays of samples from the posterior distributions of alpha and 
            beta logistic parameters, respectively.
    """
    
    beta = mc.Normal("beta", 0, 0.001, value=0)
    alpha = mc.Normal("alpha", 0, 0.001, value=0)

    @mc.deterministic
    def p(r=rates.values, alpha=alpha, beta=beta):
        return 1.0 / (1. + np.exp(beta*r + alpha))

    #Observed goals
    observed = mc.Bernoulli("bernoulli_obs", p, value=goals.values-1, observed=True)
    model = mc.Model([observed, beta, alpha])

    map_ = mc.MAP(model)
    map_.fit()
    mcmc = mc.MCMC(model)
    mcmc.sample(samples, burn, thin)

    return mcmc.trace('alpha')[:, None], mcmc.trace('beta')[:, None]

def logistic_performance(test_data, test_goals, alpha, beta):
    test_p = logistic(test_data.values, beta, alpha)
    predictions = (np.random.rand(*test_p.shape) < test_p).astype(int) + 1
    performance = (predictions == test_goals.values).mean(axis=1)

    return performance

def poisson_decoder(counts, goals, samples=4000, burn=2000, thin=2):
    """ Train a decoder using a poisson distribution for the likelihood. 

        **Returns**
            Arrays of samples from the posterior distributions of lambda 
            parameters.
    """
    
    lambdas = mc.Uniform("lambdas", 0, 10, size=2)

    @mc.deterministic
    def lambda_(g=(goals.values-1).astype(int), lambdas=lambdas):
        out = []
        for each in g:
            if each == 0:
                out.append(lambdas[0])
            elif each == 1:
                out.append(lambdas[1])
        return out
        
    #Observed goals
    observed = mc.Poisson("count_obs", lambda_, value=counts.values, observed=True)
    model = mc.Model([observed, lambdas])

    map_ = mc.MAP(model)
    map_.fit()
    mcmc = mc.MCMC(model)
    mcmc.sample(samples, burn, thin)

    return mcmc.trace('lambdas')[:,:]

def predict_goal(count, lambdas, p, size=(1,):
        likelihoods = poisson.pmf(count, lambdas)
        prob = np.argmax(np.array([p, 1-p])*likelihoods, axis=1).mean()
        return np.random.rand(size=size) < prob

def poisson_performance(counts, goals, lambdas, train, test):
    
    goals_p = (goals.ix[train]==1).mean()
    predictions = np.array([predict_goal(count, lambdas, goals_p) 
        for count in counts.ix[test] ])
    return (predictions == goals.ix[test]-1).mean()

