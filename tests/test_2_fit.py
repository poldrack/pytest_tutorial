"""
test for rtanalysis
- in this test, we will create a simulated dataset and fit
it, ensuring that the answers are correct
"""
import numpy as np
from rtanalysis.generate_testdata import generate_test_df
from rtanalysis.rtanalysis import RTAnalysis


def test_rtanalysis_fit():
    rta = RTAnalysis()
    meanRT = 2.1
    sdRT = 0.9
    meanAcc = 0.8
    test_df = generate_test_df(meanRT, sdRT, meanAcc)
    rta.fit(test_df.rt, test_df.accuracy)
    assert np.allclose(meanRT, rta.mean_rt_)
    assert np.allclose(meanAcc, rta.mean_accuracy_)
