"""example function to analyze reaction times
- given a data frame with RT and accuracy, 
compute mean RT for correct trials and mean accuracy
"""

import pandas as pd
import numpy as np


class RTAnalysis:
    """[summary]
    """
    def __init__(self, outlier_cutoff_sd=None):
        """ 
        RT analysis

        Parameters:
        -----------
        outlier_cutoff_sd: standard deviation cutoff for long RT outliers (default: no cutoff)
        """
        self.outlier_cutoff_sd = outlier_cutoff_sd
        self.meanrt_ = None
        self.meanacc_ = None
        
    def fit(self, rt, accuracy, verbose=True):
        """[summary]

        Args:
            rt (Series of floats): response times for each trial
            accuracy (Series of booleans): accuracy for each trial
        """
        
        assert type(rt) is pd.core.series.Series
        assert type(accuracy) is pd.core.series.Series

        # ensure that RT's are non-negative
        assert rt.min() >= 0
        # ensure that accuracy values are boolean
        assert len(set(acc.unique()).difference([ True, False])) == 0

        if self.outlier_cutoff_sd is not None:
            cutoff = rt.std() * self.outlier_cutoff_sd
            if verbose:
                print(f'outlier rejection excluded {(rt > cutoff).sum()} trials')
            rt = rt.mask(rt > cutoff)

        rt = rt.mask(~accuracy)
        self.meanrt_ = rt.mean()
        self.meanacc_ = accuracy.mean()

        if verbose:
            print(f'mean RT: {self.meanrt_}')
            print(f'mean accuracy: {self.meanacc_}')

def generate_test_df(meanRT, meanAcc, sdcutoff):
    """
    generate simulated RT data for testing

    Args:
        meanRT ([type]): [description]
        meanAcc ([type]): [description]
        sdcutoff ([type]): [description]
    """


if __name__ == "__main__":
    