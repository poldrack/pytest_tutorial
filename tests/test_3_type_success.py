"""
test for rtanalysis
- in this test, we will ensure that the function raises a ValueError
"""
import pytest
from rtanalysis.generate_testdata import generate_test_df
from rtanalysis.rtanalysis import RTAnalysis


def test_dataframe_error_with_raises():
    rta = RTAnalysis()
    test_df = generate_test_df(2, 1, 0.8)
    with pytest.raises(ValueError):
        rta.fit(test_df.rt, test_df.accuracy.loc[1:])
