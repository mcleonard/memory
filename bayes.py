""" This module is for Bayesian data analysis methods """

import numpy as np
import pymc as pm
import scipy as sp

def poisson_rate(t, n, alpha=0.5, beta=0):
    """ Calculates the posterior distribution for the rate of a Poisson
    	process.

    	Uses the Gamma conjugate prior with parameters alpha and beta. Uses
    	the Jeffreys' prior by default. The exponential prior is (1, 1).

    	Parameters
    	----------
    	t: The total number of events
    	n: The number of trials
    	
    	Returns the posterior distribution as a scipy RV object, see 
    	scipy.stats.gamma

    """

    posterior = sp.stats.gamma(alpha + t, scale=(beta + n)**-1)
    return posterior

def binomial_posterior(outcomes, alpha=0.5, beta=0.5):
    successes = np.sum(outcomes)
    posterior = sp.stats.beta(alpha + successes, 
                              beta + len(outcomes) - successes)
    return posterior

def vect_poisson_rate(df, alpha=0.5, beta=0):
	""" Vectorized version of poisson_rate. """

   	counts = df.apply(np.sum).values
   	N = df.shape[0]
   	posteriors = [poisson_rate(each,N,alpha,beta) for each in counts]
   	return posteriors

def poisson_regression(targets, predictors, iters=2000):
    """ Return the posterior of a Bayesian Poisson regression model.

        This function takes the targets and predictors and builds a Poisson
        regression model. The predictor coefficients are found by sampling
        from the posterior using PyMC (NUTS in particular).

        The posterior is returned as an MxN array, where M is the number of
        samples and N is the number of predictors. The first column is the 
        coefficient for the first predictor and so on.

        Requires PyMC3

    """
    with pm.Model() as poisson_model:
        # priors for coefficients
        coeffs = pm.Uniform('coeffs', -10, 10, shape=(1, predictors.shape[1]))
        
        p = t.exp(pm.sum(coeffs*predictors.values, 1))
        
        obs = pm.Poisson('obs', p, observed=targets)

        start = pm.find_MAP()
        step = pm.NUTS(scaling=start)
        poisson_trace = pm.sample(iters, step, start=start, progressbar=False)

    return poisson_trace['coeffs'].squeeze()

