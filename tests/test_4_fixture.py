"""
test for rtanalysis
- in this test, we will create a simulated dataset as a fixture
and use it across multiple tests
- we also create a separate fixture to hold the parameters
"""
import numpy as np
import pytest
from rtanalysis.generate_testdata import generate_test_df
from rtanalysis.rtanalysis import RTAnalysis


@pytest.fixture
def params():
    return {"meanRT": 2.1, "sdRT": 0.9, "meanAcc": 0.8}


@pytest.fixture
def simulated_data(params):
    return generate_test_df(params["meanRT"], params["sdRT"], params["meanAcc"])


def test_rtanalysis_fit(simulated_data, params):
    rta = RTAnalysis()
    rta.fit(simulated_data.rt, simulated_data.accuracy)
    assert np.allclose(params["meanRT"], rta.mean_rt_)
    assert np.allclose(params["meanAcc"], rta.mean_accuracy_)


def test_rtanalysis_checkfail(simulated_data, params):
    rta = RTAnalysis()
    with pytest.raises(ValueError):
        rta.fit(
            simulated_data.rt, simulated_data.accuracy.loc[1:]
        )  # omit first datapoint
