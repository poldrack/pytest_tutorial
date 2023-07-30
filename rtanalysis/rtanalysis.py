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
        self.mean_rt_ = None
        self.mean_accuracy_ = None

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

        self._validate_length(rt, accuracy)

        # Ensure that accuracy values are boolean.
        assert accuracy.dtype == bool

        rt = self.reject_outlier_rt(rt, verbose=verbose)

        self.mean_accuracy_ = accuracy.mean()
        try:
            assert self.mean_accuracy_ > 0
        except AssertionError as e:
            raise ValueError("Accuracy is zero!") from e

        rt = rt.mask(~accuracy)
        self.mean_rt_ = rt.mean()

        try:
            assert rt.min() >  0
        except:
            raise ValueError( "negative response times found")
        if verbose:
            print(f"mean RT: {self.mean_rt_}")
            print(f"mean accuracy: {self.mean_accuracy_}")
    
    @staticmethod
    def _validate_length(rt, accuracy):
        """Validate response time and accuracy series lengths.

        Parameters
        ----------
        rt : pd.Series
            Response time values
        accuracy : _type_
            Accuracy values

        Raises
        ------
        ValueError
            Length mismatch
        """
        same_length = rt.shape[0] == accuracy.shape[0]
        try:
            assert same_length
        except AssertionError as e:
            raise ValueError("RT and accuracy must be the same length!") from e


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
        if not isinstance(var, pd.Series):
            var = pd.Series(var)
        return var

    def reject_outlier_rt(self, rt, verbose=True):
        if self.outlier_cutoff_sd is None:
            return rt
        cutoff = rt.std() * self.outlier_cutoff_sd
        if verbose:
            n_excluded = (rt > cutoff).sum()
            print(f"Outlier rejection excluded {n_excluded} trials.")
        return rt.mask(rt > cutoff)        
