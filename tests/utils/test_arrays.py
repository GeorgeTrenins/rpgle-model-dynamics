import itertools
import pytest
from rpmdgle.utils.arrays import index_in_slice

BOUNDS = [None] + list(range(-6, 7))
STEPS = [None, -3, -2, -1, 1, 2, 3]


def selected(slc, index, length):
    """Brute-force reference: is seq[index] among the elements of seq[slc]?"""
    seq = list(range(length))
    return -length <= index < length and seq[index] in seq[slc]


@pytest.mark.parametrize("slc, index, expected", [
    (slice(None, -1), 2, True),
    (slice(-2, None), 0, False),
    (slice(None, 0), 3, False),
    (slice(0, 0), 3, False),
    (slice(3, 0, -1), 2, True),
    (slice(None, None, -1), 2, True),
    (slice(1, 4), -2, True),
    (slice(1, 4), 4, False),
    (slice(None), 5, False),
    (slice(None), -6, False),
])
def test_examples_with_length(slc, index, expected):
    assert index_in_slice(slc, index, 5) is expected

def test_against_slicing_with_length():
    for length, start, stop, step in itertools.product(range(7), BOUNDS, BOUNDS, STEPS):
        slc = slice(start, stop, step)
        for index in range(-8, 9):
            assert index_in_slice(slc, index, length) == selected(slc, index, length), (slc, index, length)

def test_without_length_agrees_with_every_length():
    # without a length, the answer must hold for every sequence that contains seq[index]
    for start, stop, step in itertools.product([None, 0, 1, 2, 5], [None, 0, 1, 3, 6], [None, 1, 2, 3]):
        slc = slice(start, stop, step)
        for index in range(8):
            answer = index_in_slice(slc, index)
            assert all(answer == selected(slc, index, length) for length in range(index+1, index+12)), (slc, index)

# negative values are rejected even where the answer happens not to depend on the
# length, e.g. slice(None) contains index -1 and slice(None, None, -1) contains every index
@pytest.mark.parametrize("slc, index", [
    (slice(None), -1),
    (slice(-2, None), 0),
    (slice(None, -1), 0),
    (slice(None, None, -1), 0),
    (slice(4, 0, -2), 2),
])
def test_without_length_rejects_negative_values(slc, index):
    with pytest.raises(ValueError):
        index_in_slice(slc, index)

@pytest.mark.parametrize("length", [None, 5])
def test_zero_step(length):
    with pytest.raises(ValueError):
        index_in_slice(slice(None, None, 0), 1, length)
