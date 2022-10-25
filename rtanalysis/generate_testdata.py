import pandas as pd
import numpy as np
import scipy.stats


def generate_test_df(meanRT, sdRT, meanAcc, n=100):
    """
    generate simulated RT data for testing

    Args:
        meanRT (float): mean RT (for correct trials)
        sdRT (float): std deviation of RT (for correct trials)
        meanAcc (float): mean accuracy (proportion, 0 <= meanAcc <= 1)
        sdcutoff ([type]): outlier cutoff (default None for no cutoff)
    """

    rt = pd.Series(scipy.stats.weibull_min.rvs(2, loc=1, size=n))

    # get random accuracy values and threshold for intended proportion
    accuracy_continuous = np.random.rand(n)
    accuracy = pd.Series(
        accuracy_continuous
        < scipy.stats.scoreatpercentile(accuracy_continuous, 100 * meanAcc)
    )

    # scale the correct RTs only
    rt_correct = rt.mask(~accuracy)
    rt_scaled = scale_values(rt_correct, meanRT, sdRT)

    # NB: .where() replaces values where the condition is False
    rt_scaled_with_inaccurate_rts = rt_scaled.where(accuracy, rt)

    return pd.DataFrame({"rt": rt_scaled_with_inaccurate_rts, "accuracy": accuracy})


def scale_values(values, mean, sd):
    """scale values by given mean/sd

    Args:
        values (array-like): values to be scaled
        mean (float): intended mean
        sd (float): intended standard deviation
    """
    values = values * (sd / np.std(values))
    values = (values - np.mean(values)) + mean

    return values
