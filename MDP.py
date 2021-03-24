from random import random
import numpy as np


def sample_bernoulli( trueProb ):
    """ Return 1 with `trueProb`, Otherwise return 0 """
    return int( random() <= trueProb )


def normalize_discrete_dist( discreteDist ):
    """ Return a normalized version of `discreteDist` expressed as a list """
    tot = np.sum( discreteDist )
    nrm = np.divide( discreteDist, tot )
    return nrm.tolist()


def sample_discrete_dist( dist ):
    """ Return the index of the event stampled from the discrete (normalized) `dist` """
    # NOTE: This function will have incorrect behavior unless `dist` is normalized to sum to 1.0
    roll  = random()
    total = 0.0
    for i, p_i in enumerate( dist ):
        total += p_i
        if roll <= total:
            return i
    return None
        

def choose_discrete_event( events, dist ):
    """ Return the event stampled from the discrete (normalized) `dist` """
    return events[ sample_discrete_dist( dist ) ]