"""test suite for rtanalysis
"""

import pytest
from rtanalysis import RTAnalysis
from rtanalysis import generate_test_df

def test_rtanalysis_smoke():
    rta = RTAnalysis()
    assert rta is not None