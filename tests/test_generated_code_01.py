import pytest
from generated_code import *

def test_multiply_basic():
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[7, 8], [9, 10], [11, 12]]
    expected = [[58, 64], [139, 154]]
    result = multiply_matrices(a, b)
    assert result == expected

def test_multiply_identity():
    a = [[1, 2], [3, 4]]
    identity = [[1, 0], [0, 1]]
    result_left = multiply_matrices(identity, a)
    result_right = multiply_matrices(a, identity)
    assert result_left == a
    assert result_right == a

def test_multiply_1x1():
    a = [[5]]
    b = [[6]]
    assert multiply_matrices(a, b) == [[30]]

def test_zero_matrix():
    a = [[0, 0], [0, 0]]
    b = [[1, 2], [3, 4]]
    assert multiply_matrices(a, b) == [[0, 0], [0, 0]]

def test_large_numbers_and_negatives():
    a = [[10**6, -2], [3, 4]]
    b = [[0, 1], [2, 3]]
    expected = [[-4, 999994], [8, 15]]
    assert multiply_matrices(a, b) == expected

def test_incompatible_dimensions_value_error():
    a = [[1, 2, 3], [4, 5, 6]]  # 2x3
    b = [[1, 2], [3, 4]]       # 2x2 -> incompatible
    try:
        multiply_matrices(a, b)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for incompatible dimensions"

def test_empty_matrix_value_error():
    a = []
    b = [[1]]
    try:
        multiply_matrices(a, b)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for empty matrix 'a'"

def test_rows_must_not_be_empty_value_error():
    a = [[]]  # single empty row
    b = [[1]]
    try:
        multiply_matrices(a, b)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for empty rows"

def test_matrix_not_list_type_error():
    a = "not a matrix"
    b = [[1]]
    try:
        multiply_matrices(a, b)
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError when 'a' is not a list"

def test_rows_not_lists_type_error():
    a = [[1, 2], (3, 4)]  # second row is tuple
    b = [[1, 0], [0, 1]]
    try:
        multiply_matrices(a, b)
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError when a row is not a list"

def test_elements_must_be_ints_type_error():
    a = [[1, 2], [3, 4.0]]  # float element
    b = [[1, 0], [0, 1]]
    try:
        multiply_matrices(a, b)
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError when an element is not an int"

def test_non_rectangular_value_error():
    a = [[1, 2], [3]]  # irregular row lengths
    b = [[1], [2]]
    try:
        multiply_matrices(a, b)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for non-rectangular matrix"