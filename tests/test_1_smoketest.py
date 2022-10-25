"""test suite for rtanalysis
"""
import pytest
from rtanalysis.rtanalysis import RTAnalysis


def test_rtanalysis_smoke():
    rta = RTAnalysis()
    assert rta is not None
