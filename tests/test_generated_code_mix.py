import pytest
from generated_code import *

def test_validate_matrix_valid_returns_dimensions():
    mat = [[1, 2], [3, 4]]
    dims = _validate_matrix(mat, "M")
    assert dims == (2, 2)

def test_validate_matrix_empty_raises_value_error():
    try:
        _validate_matrix([], "M")
    except ValueError as e:
        assert str(e) == "M must be a non-empty matrix."
    else:
        assert False, "Expected ValueError for empty matrix"

def test_validate_matrix_not_list_of_lists_raises():
    try:
        _validate_matrix([(1, 2), (3, 4)], "T")  # rows are tuples, not lists
    except ValueError as e:
        assert str(e) == "T must be a list of lists."
    else:
        assert False, "Expected ValueError when rows are not lists"

def test_validate_matrix_row_empty_raises():
    try:
        _validate_matrix([[]], "R")
    except ValueError as e:
        assert str(e) == "R rows must be non-empty."
    else:
        assert False, "Expected ValueError for empty rows"

def test_validate_matrix_nonrectangular_raises_with_detail():
    mat = [[1, 2], [3]]
    try:
        _validate_matrix(mat, "N")
    except ValueError as e:
        expected = "All rows in N must have the same length; row 0 has length 2 but row 1 has length 1."
        assert str(e) == expected
    else:
        assert False, "Expected ValueError for non-rectangular matrix"

def test_validate_matrix_nonint_element_raises_with_location():
    mat = [[1, "a"]]
    try:
        _validate_matrix(mat, "X")
    except ValueError as e:
        assert str(e) == "All elements of X must be integers; found str at (0, 1)."
    else:
        assert False, "Expected ValueError for non-integer element"

def test_validate_matrix_float_element_is_invalid():
    mat = [[1.0]]
    try:
        _validate_matrix(mat, "F")
    except ValueError as e:
        assert str(e) == "All elements of F must be integers; found float at (0, 0)."
    else:
        assert False, "Expected ValueError for float element"

def test_multiply_matrices_basic():
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8], [9, 10], [11, 12]]
    result = multiply_matrices(A, B)
    assert result == [[58, 64], [139, 154]]

def test_multiply_matrices_identity_behavior():
    I = [[1, 0], [0, 1]]
    M = [[5, 6], [7, 8]]
    assert multiply_matrices(I, M) == M
    assert multiply_matrices(M, I) == M

def test_multiply_1x1_matrices():
    assert multiply_matrices([[3]], [[4]]) == [[12]]

def test_multiply_with_negatives_and_zeros():
    A = [[0, -1], [2, 3]]
    B = [[4, 5], [-6, 7]]
    # Compute manually:
    # row0: [0*4 + -1*-6 = 6, 0*5 + -1*7 = -7]
    # row1: [2*4 + 3*-6 = 8 - 18 = -10, 2*5 + 3*7 = 10 + 21 = 31]
    assert multiply_matrices(A, B) == [[6, -7], [-10, 31]]

def test_multiply_incompatible_dimensions_raises_with_message():
    A = [[1, 2, 3], [4, 5, 6]]  # 2x3
    B = [[1, 2], [3, 4], [5, 6], [7, 8]]  # 4x2
    try:
        multiply_matrices(A, B)
    except ValueError as e:
        assert str(e) == "Incompatible dimensions for multiplication: A is 2x3, B is 4x2."
    else:
        assert False, "Expected ValueError for incompatible dimensions"