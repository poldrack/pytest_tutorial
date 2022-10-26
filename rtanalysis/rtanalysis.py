"""Example class to analyze reaction times.

Given a data frame with RT and accuracy, compute mean RT for correct trials and
mean accuracy.
"""
import pandas as pd


class RTAnalysis:
    """Response time (RT) analysis."""

    def __init__(self, outlier_cutoff_sd=None):
        """Initialize a new RTAnalysis instance.

        Parameters
        ----------
        outlier_cutoff_sd : float, optional
            Standard deviation cutoff for long RT outliers, by default None
        """
        self.outlier_cutoff_sd = outlier_cutoff_sd
        self.meanrt_ = None
        self.meanacc_ = None

    def fit(self, rt, accuracy, verbose=True):
        """Fit response time to accuracy.

        Parameters
        ----------
        rt : pd.Series
            Response time per trial
        accuracy : pd.Series
            Accuracy per trial
        verbose : bool, optional
            Whether to print verbose output or not, by default True

        Raises
        ------
        ValueError
            RT/accuracy length mismatch
        ValueError
            Accuracy is 0
        """
        rt = self._ensure_series_type(rt)
        accuracy = self._ensure_series_type(accuracy)

        try:
            assert rt.shape[0] == accuracy.shape[0]
        except AssertionError as e:
            raise ValueError("rt and accuracy must be the same length!") from e

        # ensure that accuracy values are boolean
        assert not set(accuracy.unique()).difference([True, False])

        if self.outlier_cutoff_sd is not None:
            cutoff = rt.std() * self.outlier_cutoff_sd
            if verbose:
                print(f"outlier rejection excluded {(rt > cutoff).sum()} trials")
            rt = rt.mask(rt > cutoff)

        self.meanacc_ = accuracy.mean()
        try:
            assert self.meanacc_ > 0
        except AssertionError as e:
            raise ValueError("accuracy is zero") from e

        rt = rt.mask(~accuracy)
        self.meanrt_ = rt.mean()

        if verbose:
            print(f"mean RT: {self.meanrt_}")
            print(f"mean accuracy: {self.meanacc_}")

    @staticmethod
    def _ensure_series_type(var):
        """Return variable as a pandas Series.

        Parameters
        ----------
        var : Iterable
            Variable to be converted

        Returns
        -------
        pd.Series
            Variable values as a pandas Series
        """
        if type(var) is not pd.core.series.Series:
            var = pd.Series(var)
        return var
