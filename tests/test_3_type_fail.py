"""
test for rtanalysis
- in this test, we will ensure that the function raises a ValueError
"""
import pytest
from rtanalysis.generate_testdata import generate_test_df
from rtanalysis.rtanalysis import RTAnalysis


# This xfail decorator tells pytest to run the test
# but don't count it as a failure when it fails
# (since we know it's going to fail)
@pytest.mark.xfail
def test_dataframe_error():
    rta = RTAnalysis()
    test_df = generate_test_df(2, 1, 0.8)
    rta.fit(test_df.rt, test_df.accuracy.loc[1:])  # omit first datapoint
