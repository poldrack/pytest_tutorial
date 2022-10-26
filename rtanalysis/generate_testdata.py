"""Utility module for handling the generation of test data."""
import numpy as np
import pandas as pd
import scipy.stats


def generate_test_df(mean_rt, sd_rt, mean_accuracy, n=100):
    """Generate simulated RT data for testing.

    Parameters
    ----------
    mean_rt : float
        Mean response time for correct trials
    sd_rt : float
        Standard deviation of the response time in correct trials
    mean_accuracy : float
        Mean accuracy across trials (between 0 and 1)
    n : int, optional
        Number of observations to generate, by default 100

    Returns
    -------
    pd.DataFrame
        Generated mock data
    """
    rt = pd.Series(scipy.stats.weibull_min.rvs(2, loc=1, size=n))

    # get random accuracy values and threshold for intended proportion
    accuracy_continuous = np.random.rand(n)
    accuracy = pd.Series(
        accuracy_continuous
        < scipy.stats.scoreatpercentile(accuracy_continuous, 100 * mean_accuracy)
    )

    # scale the correct RTs only
    rt_correct = rt.mask(~accuracy)
    rt_scaled = scale_values(rt_correct, mean_rt, sd_rt)

    # NB: .where() replaces values where the condition is False
    rt_scaled_with_inaccurate_rts = rt_scaled.where(accuracy, rt)

    return pd.DataFrame({"rt": rt_scaled_with_inaccurate_rts, "accuracy": accuracy})


def scale_values(values, mean, sd):
    """Scale values by given mean/SD.

    Parameters
    ----------
    values : array-like
        Values to be scaled
    mean : float
        Target mean
    sd : float
        Target standard deviation

    Returns
    -------
    array-like
        Scaled values
    """
    values = values * (sd / np.std(values))
    values = (values - np.mean(values)) + mean
    return values
