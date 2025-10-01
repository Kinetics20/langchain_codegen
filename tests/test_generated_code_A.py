import pytest
from generated_code import *

def test_multiply_2x2():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    expected = [[19, 22], [43, 50]]
    result = multiply_matrices(a, b)
    assert result == expected, f"Expected {expected}, got {result}"

def test_multiply_rectangular_2x3_by_3x2():
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[7, 8], [9, 10], [11, 12]]
    expected = [[58, 64], [139, 154]]
    assert multiply_matrices(a, b) == expected

def test_multiply_1x1():
    a = [[7]]
    b = [[3]]
    assert multiply_matrices(a, b) == [[21]]

def test_multiply_with_zero_and_negative():
    a = [[0, -1], [2, 3]]
    b = [[4, -5], [6, 7]]
    expected = [[-6, -7], [26, 11]]
    assert multiply_matrices(a, b) == expected

def test_incompatible_dimensions_raises_value_error():
    a = [[1, 2, 3], [4, 5, 6]]  # 2x3
    b = [[1, 2], [3, 4], [5, 6], [7, 8]]  # 4x2 -> inner dims 3 vs 4 incompatible
    try:
        multiply_matrices(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for incompatible dimensions")

def test_non_int_entries_in_a_raise_value_error():
    a = [[1.0, 2], [3, 4]]  # float present
    b = [[1, 0], [0, 1]]
    try:
        multiply_matrices(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-int entries in 'a'")

def test_non_int_entries_in_b_raise_value_error():
    a = [[1, 2], [3, 4]]
    b = [[1, "x"], [0, 1]]  # string present
    try:
        multiply_matrices(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-int entries in 'b'")

def test_empty_matrix_raises_value_error():
    a = []
    b = [[1]]
    try:
        multiply_matrices(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty matrix 'a'")

def test_empty_row_raises_value_error():
    a = [[]]  # row with zero columns
    b = [[1]]
    try:
        multiply_matrices(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for matrix with empty row")

def test_non_rectangular_matrix_raises_value_error():
    a = [[1, 2], [3]]  # non-rectangular
    b = [[1], [2]]
    try:
        multiply_matrices(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-rectangular matrix 'a'")

def test_result_does_not_mutate_inputs():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    a_copy = [row[:] for row in a]
    b_copy = [row[:] for row in b]
    _ = multiply_matrices(a, b)
    assert a == a_copy, "Input matrix 'a' was mutated"
    assert b == b_copy, "Input matrix 'b' was mutated"

def test_large_integer_entries():
    big1 = 10**50
    big2 = 10**60
    a = [[big1, big2]]
    b = [[big2], [big1]]
    # result is 1x1: big1*big2 + big2*big1 = 2 * big1 * big2
    expected = [[2 * big1 * big2]]
    assert multiply_matrices(a, b) == expected