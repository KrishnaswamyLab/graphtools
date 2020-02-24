import graphtools
from load_tests import assert_raises_message


def test_check_in():
    graphtools.utils.check_in(["hello", "world"], foo="hello")
    with assert_raises_message(
        ValueError, "foo value bar not recognized. Choose from ['hello', 'world']"
    ):
        graphtools.utils.check_in(["hello", "world"], foo="bar")


def test_check_int():
    graphtools.utils.check_int(foo=5)
    graphtools.utils.check_int(foo=-5)
    with assert_raises_message(ValueError, "Expected foo integer, got 5.3"):
        graphtools.utils.check_int(foo=5.3)


def test_check_positive():
    graphtools.utils.check_positive(foo=5)
    with assert_raises_message(ValueError, "Expected foo > 0, got -5"):
        graphtools.utils.check_positive(foo=-5)
    with assert_raises_message(ValueError, "Expected foo > 0, got 0"):
        graphtools.utils.check_positive(foo=0)


def test_check_if_not():
    graphtools.utils.check_if_not(-5, graphtools.utils.check_positive, foo=-5)
    with assert_raises_message(ValueError, "Expected foo > 0, got -5"):
        graphtools.utils.check_if_not(-4, graphtools.utils.check_positive, foo=-5)


def test_check_between():
    graphtools.utils.check_between(-5, -3, foo=-4)
    with assert_raises_message(ValueError, "Expected foo between -5 and -3, got -6"):
        graphtools.utils.check_between(-5, -3, foo=-6)
    with assert_raises_message(ValueError, "Expected v_max > -3, got -5"):
        graphtools.utils.check_between(-3, -5, foo=-6)
