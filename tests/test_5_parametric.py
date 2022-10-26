"""
test for rtanalysis
- in this test, we will try various parameter levels
and make sure that the function properly raises
and exception is accuracy is zero
"""
import numpy as np
import pytest
from rtanalysis.generate_testdata import generate_test_df
from rtanalysis.rtanalysis import RTAnalysis


@pytest.mark.parametrize(
    "meanRT, sdRT, meanAcc", [(1.5, 1.0, 0.9), (1500, 1000, 0.9), (1.5, 1.0, 0)]
)
def test_rtanalysis_parameteric(meanRT, sdRT, meanAcc):
    test_df = generate_test_df(meanRT, sdRT, meanAcc)
    rta = RTAnalysis()
    if meanAcc > 0:
        rta.fit(test_df.rt, test_df.accuracy)
        assert np.allclose(meanRT, rta.mean_rt_)
        assert np.allclose(meanAcc, rta.mean_accuracy_)
    else:
        with pytest.raises(ValueError):
            rta.fit(test_df.rt, test_df.accuracy)
