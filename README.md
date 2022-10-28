# A basic pytest tutorial for data science

![Python application](https://github.com/poldrack/pytest_tutorial/workflows/Python%20application/badge.svg)

Researchers often wish to know how to implement software testing for data science applications. This tutorial provides an example of how to get started with software testing in the context of data science, using the [`pytest`](https://docs.pytest.org/en/7.1.x/) library for Python.

For this exercise you should fork a copy of the repository and clone it to your local machine. The foregoing will assume that you have a fully configured scientific Python installation, and that you have installed the requirements listed in _requirements.txt_ (`pip install -r requirements.txt`).

Most of the content here is basic Python, but there are several concepts that you should have at least a basic familiarity with:

- [Decorators](https://www.geeksforgeeks.org/decorators-in-python/) are a syntactic element in Python that modify the operation of a function.
- [Exceptions](https://realpython.com/python-exceptions/) are signals that are emitted when an error occurs during the execution of a program. Importantly, exceptions are Python objects that have a specific type that is specified by the code that raises the exception. For example, if one tries to open a non-existent file using `open()`, it will raise a `FileNotFoundError` exception, whereas if one tries to divide by zero the Python interpreter will raise a `ZeroDivisionError` exception.
- [Context managers](https://book.pythontips.com/en/latest/context_managers.html) are a syntatic element that are usually used to control resources like file or database handles, but can also be used to monitor and control the operation of a set of functions, which will be useful when we are looking for specific exceptions.

## The setup

The goal of this project is to develop a set of tests for a simple Python class that computes the mean response time and accuracy from raw response time and accuracy values, using only the correct response times. The `RTAnalysis` class defined in [rtanalysis.py](rtanalysis/rtanalysis.py) uses an interface patterned after the analysis methods in scikit-learn. To use it, we first instantiate the `RTAnalysis` object:

```python
from rtanalysis.rtanalysis import RTAnalysis

rta = RTAnalysis()
```

The data to be analyzed should be stored in two pandas Series of the same size, one containing response times (non-negative floating point numbers) and another containing accuracy values for each trial [(Boolean values)](https://www.scaler.com/topics/python/boolean-operators-in-python/). Assuming those variables are called `rt` and `accuracy` respectively, the model can be fit using:

```python
rta.fit(rt, accuracy)
```

The resulting estimates are printed to the screen (assuming that the `verbose` flag is not set to false) and also stored to internal variables `rta.mean_rt_` and `rta.mean_accuracy_`.

## Test 1: A simple smoke test

For our first test, let's simply instantiate the `RTAnalysis` class and ensure that the resulting object is not empty. We call this a "smoke test" since it mostly just makes sure that things run and don't break --- it doesn't actually test the functionality. This is done in [test_1_smoketest.py](tests/test_1_smoketest.py):

```python
import pytest
from rtanalysis.rtanalysis import RTAnalysis

def test_rtanalysis_smoke():
    rta = RTAnalysis()
    assert rta is not None
```

We can run the test using pytest from the command line:

```sh
python -m pytest tests/test_1_smoketest.py
```

This should return something like:

```
==================================== test session starts =====================================
platform darwin -- Python 3.8.3, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /Users/poldrack/Dropbox/code/pytest_tutorial
plugins: cov-2.10.0
collected 1 item

rtanalysis/test_1_smoketest.py .                                                       [100%]

===================================== 1 passed in 0.25s ======================================
```

## Test 2: Does it get the answer right?

Now we would like to make sure that the function does what it is supposed to do. In order to test the function we will need to generate some test data where we know the correct answer. This can be done using the `generate_test_df()` function:

```python
from rtanalysis.generate_testdata import generate_test_df

meanRT = 2.0
sdRT = 0.75
meanAcc = 0.8

test_df = generate_test_df(meanRT, sdRT, meanAcc)
```

This data frame includes two series, called `rt` and `accuracy` that can be used to test the function:

```python
rta.fit(test_df.rt, test_df.accuracy)
```

Here is what our test function looks like ([test_2_fit.py](tests/test_2_fit.py)):

```python
def test_rtanalysis_fit():
    meanRT = 2.1
    sdRT = 0.9
    meanAcc = 0.8
    test_df = generate_test_df(meanRT, sdRT, meanAcc)

    rta = RTAnalysis()
    rta.fit(test_df.rt, test_df.accuracy)

    assert np.allclose(meanRT, rta.mean_rt_)
    assert np.allclose(meanAcc, rta.mean_accuracy_)
```

We generate the data with known mean and accuracy values, fit the model using our function, and then confirm that our estimates are basically equal to the actual values. We use `np.allclose()` rather than a test for equality because sometimes the values will be off by a very small amount due to the numerical precision of the computer; an equality test would treat those as different, but `np.allclose` allows some tolerance in its test.

## Test 3: Does it raise the appropriate error if we give it invalid data?

Test 2 checked whether our program performed as advertised. However, as Myers et al. state in their book [The Art of Software Testing](http://barbie.uta.edu/~mehra/Book1_The%20Art%20of%20Software%20Testing.pdf):

> Examining a program to see if it does not do what it is supposed to do is only half the battle; the other half is seeing whether the program does what it is not supposed to do.

That is, we need to try to cause the program to make errors, and make sure that it avoids them appropriately. In this case, we will start by seeing what happens if our `rt` and `accuracy` series are of different sizes. Let's first write a test to see what happens if we do this [test_3_type_fail.py](tests/test_3_type_fail.py):

```python
def test_dataframe_error():
    rta = RTAnalysis()
    test_df = generate_test_df(2, 1, 0.8)
    rta.fit(test_df.rt, test_df.accuracy.loc[1:])
```

If we run this test, we will see that it fails, due to the error that is raised by the function when the data are incorrectly sized. (Note that we have told pytest to ignore this failure, so that it won't cause our entire test run to fail, using the `@pytest.mark.xfail` decorator.) This is the correct behavior on the part of our function, but it's not the correct behavior on the part of our test! Instead, we want the test to succeed _if and only if_ the correct exception is raised. To do this, we can use the `pytest.raises` function as a context manager [test_3_type_success.py](tests/test_3_type_success.py):

```python
def test_dataframe_error_with_raises():
    rta = RTAnalysis()
    test_df = generate_test_df(2, 1, 0.8)
    with pytest.raises(ValueError):
        rta.fit(test_df.rt, test_df.accuracy.loc[1:])
```

This is basically telling pytest that we expect this particular function to raise a `ValueError` exception, and that the test should fail if this particular exception is _not_ raised.

## Exercise 1

The existing code does not check for whether there are any negative response times in the input data. In this exercise you will first write a new test function that generates an example data set, generate negative response times (e.g., by multiplying the response time variable by -1), and then generating a test that should only pass if the function raises a `ValueError` exception when a negative response time is found. Then, you should add an assertion statement to the `RTAnalysis.fit()` function that will raise a `ValueError` if there are any negative response times present.

## Automating tests using Github Actions

It's useful to have our tests run automatically whenever we push a commit to Github. This kind of testing is known as "continuous integration" (CI) testing. The Github Actions system makes this very easy to configure. To do the following, you will first need to remove the .github/workflows/python-app.yml from your fork and then commit that change and push it to Github.

1.  Click on the "Actions" tab at the top of your repo page.
2.  Choose "Set up this workflow" for the "Python Application" suggestion.
3.  This will open an editor for a YAML file that defines the workflow. You will need to make one change to the default, in order to install several libraries that our code requires. Replace the following line:

        pip install flake8 pytest

    with:

        pip install -r requirements.txt

    Once you have made that change, save it using the "Start Commit" button.

4.  Return to the Actions tab, and you should see the action running. Once it completes, it should have a green check mark.

If you would like to add a badge to your README file that shows the status of this test, you can copy a snippet of Markdown using the "Create status badge" button the action, and then insert that into your README.

## Test 4: Making a persistent fixture for testing

Let's say that we want to create several tests, all of which use the same object. In this case, let's say that we want to create several tests that use the same simulated dataset. We can do that by creating what we call a _fixture_ in pytest, which is an object that can be passed into a test. In addition to a fixture containing the dataset, we also create a fixture to contain our parameters, so that they can be used for testing (see [test_4_fixture.py](tests/test_4_fixture.py)):

```python
@pytest.fixture
def params():
    return({'meanRT': 2.1,
            'sdRT': 0.9,
            'meanAcc': 0.8})


@pytest.fixture
def simulated_data(params):
    return generate_test_df(
        params['meanRT'], params['sdRT'], params['meanAcc']
    )


def test_rtanalysis_fit(simulated_data, params):
    rta = RTAnalysis()
    rta.fit(simulated_data.rt, simulated_data.accuracy)
    assert np.allclose(params['meanRT'], rta.mean_rt_)
    assert np.allclose(params['meanAcc'], rta.mean_accuracy_)


def test_rtanalysis_checkfail(simulated_data, params):
    rta = RTAnalysis()
    with pytest.raises(ValueError):
        rta.fit(simulated_data.rt,
                simulated_data.accuracy.loc[1:])
```

## Test 5: Parametric tests

Sometimes we wish to test a function across multiple values of a parameter. For example, let's say that we want to make sure that our function works for response times that are coded either in seconds or milliseconds. We can run the same test with different parameters in pytest using the `@pytest.mark.parametrize` decorator ([test_5_parametric.py](tests/test_5_parametric.py)).

```python
@pytest.mark.parametrize(
    "meanRT, sdRT, meanAcc",
    [(1.5, 1.0, 0.9), (1500, 1000, 0.9), (1.5, 1.0, 0)]
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
```

This loops through each of the sets of parameters for the three variables. It checks whether the function runs appropriately for each level, unless accuracy is equal to zero, in which case it ensures that the function raises the appropriate `ValueError` exception.

## Test coverage

It can be useful to know which portions of our code are actually being exercised by our tests. There are various types of test coverage; we will focus here on simply assessing whether each line in the code has been covered, but see [The Art of Software Testing](http://barbie.uta.edu/~mehra/Book1_The%20Art%20of%20Software%20Testing.pdf) for much more on this topic.

We can assess the degree to which our tests cover our code using the [Coverage.py](https://coverage.readthedocs.io/en/6.5.0/) tool with the [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) extension. With these installed, we simply add the `--cov` argument to our pytest commands, which will give us a coverage report. We will specify the code directory so that the coverage is only computed for our code of interest, not for the tests themselves:

```sh
python -m pytest --cov=rtanalysis
```

This should now return:

```sh
==================================================================================== test session starts ====================================================================================
platform darwin -- Python 3.8.3, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /Users/poldrack/Dropbox/code/pytest_tutorial
plugins: cov-2.10.0
collected 9 items

tests/test_1_smoketest.py .                                                                                                                                                           [ 11%]
tests/test_2_fit.py .                                                                                                                                                                 [ 22%]
tests/test_3_type_fail.py x                                                                                                                                                           [ 33%]
tests/test_3_type_success.py .                                                                                                                                                        [ 44%]
tests/test_4_fixture.py ..                                                                                                                                                            [ 66%]
tests/test_5_parametric.py ...                                                                                                                                                        [100%]

---------- coverage: platform darwin, python 3.8.3-final-0 -----------
Name                              Stmts   Miss  Cover
-----------------------------------------------------
rtanalysis/__init__.py                0      0   100%
rtanalysis/generate_testdata.py      15      0   100%
rtanalysis/rtanalysis.py             34      5    85%
-----------------------------------------------------
TOTAL                                49      5    90%


=============================================================================== 8 passed, 1 xfailed in 1.10s ================================================================================
```

Now we see that our pytest output also includes a coverage report, which tells us that we have only covered 85% of the statements in rtanalysis.py. We can look further at which statements we are missing using the `coverage annotate` function, which generates a set of files that are annotated with regard to which statements have been covered:

```sh
$ coverage annotate
$ ls -1 rtanalysis
__init__.py
__init__.py,cover
__pycache__
generate_testdata.py
generate_testdata.py,cover
rtanalysis.py
rtanalysis.py,cover
```

We see here that the annotation function has generated a set of files with the suffix ",cover". Each line in this file is marked with a `>` symbol if it was covered in the testing, and a `!` symbol if it was not. From this, we can see that there were two sections in the code that were not covered:

```python
>         if self.outlier_cutoff_sd is not None:
!             cutoff = rt.std() * self.outlier_cutoff_sd
!             if verbose:
!                 print(f'outlier rejection excluded {(rt > cutoff).sum()} trials')
!             rt = rt.mask(rt > cutoff)
```

and

```python
>         if type(var) is not pd.core.series.Series:
!             var = pd.Series(var)
```

## Exercise 2

Generate two new tests that will cause these two sections of code to be executed and thus raise coverage of rtanalysis.py to 100%.
