"""
test for rtanalysis
- in this test, we will ensure that the function raises a ValueError
"""

import pytest
from rtanalysis.rtanalysis import RTAnalysis
from rtanalysis.generate_testdata import generate_test_df


@pytest.mark.xfail
def test_dataframe_error():
    rta = RTAnalysis()
    test_df = generate_test_df(2, 1, 0.8)
    rta.fit(test_df.rt, test_df.accuracy.loc[:10])
