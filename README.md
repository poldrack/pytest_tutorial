# A basic pytest tutorial for data science

![Python application](https://github.com/poldrack/pytest_tutorial/workflows/Python%20application/badge.svg)

Resarchers often wish to know how to implement software testing for data science applications.  This tutorial provides an example of how to get started with software testing in the context of data science, using the pytest library for Python.

The foregoing will assume that you have a fully configured scientific Python installation, and that you have installed the pytest package (``pip install -U pytest``).  For this exercise you should fork a copy of the repository and clone it to your local machine.

## The setup

The goal of this project is to develop a set of tests for a simple Python class that computes the mean response time and accuracy from raw response time and accuracy values, using only the correct response times. The ``RTAnalysis`` class defined in [rtanalysis.py](rtanalysis/rtanalysis.py) uses an interface patterned after the analysis methods in scikit-learn.  To use it, we first instantiate the RTAnalysis object:

    from rtanalysis.rtanalysis import RTAnalysis
    rta = RTAnalysis()

The data to be analyzed should be stored in two pandas Series of the same size, one containing response times (non-negative floating point numbers) and another containing accuracy values for each trial (Boolean values).  Assuming those variables are called ``rt`` and ``accuracy`` respectively, the model can be fit using:

    rta.fit(rt, accuracy)

The resulting estimates are printed to the screen (assuming that the ``verbose`` flag is not set to false) and also stored to internal variables ``rta.meanrt_`` and ``rta.meanacc_``.


## Test 1: A simple smoke test

For our first test, let's simply instantiate the ``RTAnalysis`` class and ensure that the resulting object is not empty.  We call this a "smoke test" since it mostly just makes sure that things run and don't break --- it doesn't actually test the functionality.  This is done in [test_1_smoketest.py](rtanalysis/test_1_smoketest.py):

    import pytest
    from rtanalysis.rtanalysis import RTAnalysis

    def test_rtanalysis_smoke():
        rta = RTAnalysis()
        assert rta is not None

We can run the test using pytest from the command line:

    pytest_tutorial % pytest rtanalysis/test_1_smoketest.py
    ==================================== test session starts =====================================
    platform darwin -- Python 3.8.3, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
    rootdir: /Users/poldrack/Dropbox/code/pytest_tutorial
    plugins: cov-2.10.0
    collected 1 item

    rtanalysis/test_1_smoketest.py .                                                       [100%]

    ===================================== 1 passed in 0.25s ======================================    


## Test 2: Does it get the answer right?

Now we would like to make sure that the function does what it is supposed to do.  In order to test the function we will need to generate some test data where we know the correct answer.  This can be done using the ``generate_test_df()`` function:

    from rtanalysis.generate_testdata import generate_test_df
    meanRT = 2.0
    sdRT = 0.75
    meanAcc = 0.8
    test_df = generate_test_df(meanRT, sdRT, meanAcc)

This data frame includes two series, called ``rt`` and ``accuracy`` that can be used to test the function:

    rta.fit(test_df.rt, test_df.accuracy)

Here is what our test function looks like:

    def test_rtanalysis_fit():
        rta = RTAnalysis()
        meanRT = 2.1
        sdRT = 0.9
        meanAcc = 0.8
        test_df = generate_test_df(meanRT, sdRT, meanAcc)
        rta.fit(test_df.rt, test_df.accuracy)
        assert np.allclose(meanRT, rta.meanrt_)
        assert np.allclose(meanAcc, rta.meanacc_)

We generate the data with known mean and accuracy values, fit the model using our function, and then confirm that our estimates are basically equal to the actual values. We use ``np.allclose()`` rather than a test for equality because sometimes the values will be off by a very small amount due to the numerical precision of the computer; an equality test would treat those as different, but ``np.allclose`` allows some tolerance in its test.

## Test 3: Does it raise the appropriate error if we give it invalid data?

Test 2 checked whether the our program performed as advertised. However,  as Myers et al. state in their book [The Art of Software Testing](http://barbie.uta.edu/~mehra/Book1_The%20Art%20of%20Software%20Testing.pdf):

> Examining a program to see if it does not do what it is supposed to do is only half the battle; the other half is seeing whether the program does what it is not supposed to do.

That is, we need to try to cause the program to make errors, and make sure that it avoids them appropriately.  In this case, we will start by seeing what happens if we give the function rt and accuracy series of different sizes.  Let's first write a test to see what happens if we do this [test_3_type_fail.py](rtanalysis/test_3_type_fail.py):

    def test_dataframe_error():
        rta = RTAnalysis()
        test_df = generate_test_df(2, 1, 0.8)
        rta.fit(test_df.rt, test_df.accuracy.loc[:10])

If we run this test, we will see that it fails, due to the error that is raised by the function when the data are incorrectly sized.  This is the correct behavior on the part of our function, but it's not the correct behavior on the part of our test!  Instead, we want the test to succeed *if and only if* the correct exception is raised.  To do this, we can use the ``pytest.raises`` function as a context manager [test_3_type_success.py](rtanalysis/test_3_type_success.py):

    def test_dataframe_error_with_raises():
        rta = RTAnalysis()
        test_df = generate_test_df(2, 1, 0.8)
        with pytest.raises(ValueError):
            rta.fit(test_df.rt, test_df.accuracy.loc[:10])

This is basically telling pytest that we expect this particular function to raise a ValueError exception, and that the test should fail if this particular exception is *not* raised.

## Exercise 1

The existing code does not check for whether there are any negative response times in the input data.  In this exercise you will first write a new test function that generates an example data set, generate negative response times (e.g. by multiplying the response time variable by -1), and then generating a test that should only pass if the function raises a ValueException when a negative response time is found. Then, you should add an assertion statement to the ``RTAnalysis.fit()`` function that will raise a ValueError exception if there are any negative response times present.


## TBD: Automating tests using Github Actions

It's useful to have our tests run automatically whenever we push a commit to github.  This kind of testing is known as "continuous integration" testing.  The Github Actions system makes this very easy to configure.

1. Click on the "Actions" tab at the top of your repo page.
2. Choose "Set up this workflow" for the "Python Application" suggestion.
3. This will open an editor for a YAML file that defines the workflow. You should be able to use the default workflow, so save it using the "Start Commit" button.
4. We need to push another commit to the repository in order to trigger the workflow.  You can do this by making an edit to any of the files (such as the README.md) file and then committing it.  

## TBD: Fixtures


## TBD: Parametric tests


