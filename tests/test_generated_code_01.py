import pytest
from generated_code_01 import *

def test_validate_matrix_valid_rectangular():
    m = [[1, 2, 3], [4, 5, 6]]
    rows, cols = _validate_matrix(m, "M")
    assert (rows, cols) == (2, 3)

def test_validate_matrix_single_row_and_single_column():
    m1 = [[7]]
    assert _validate_matrix(m1, "M1") == (1, 1)
    m2 = [[1], [2], [3]]
    assert _validate_matrix(m2, "M2") == (3, 1)

def test_validate_matrix_empty_matrix_error():
    try:
        _validate_matrix([], "M")
        assert False, "Expected ValueError for empty matrix"
    except ValueError as e:
        assert str(e) == "M must be a non-empty list of rows."

def test_validate_matrix_row_not_list_error():
    try:
        _validate_matrix([1, 2, 3], "M")
        assert False, "Expected ValueError when rows are not lists"
    except ValueError as e:
        assert str(e) == "All rows of M must be lists."

def test_validate_matrix_empty_row_error():
    try:
        _validate_matrix([[]], "M")
        assert False, "Expected ValueError for empty row"
    except ValueError as e:
        assert str(e) == "Rows of M must be non-empty."

def test_validate_matrix_non_integer_elements_error():
    try:
        _validate_matrix([[1, "a"]], "M")
        assert False, "Expected ValueError for non-integer element"
    except ValueError as e:
        assert str(e) == "All elements of M must be integers."

def test_validate_matrix_non_rectangular_error():
    try:
        _validate_matrix([[1, 2], [3]], "M")
        assert False, "Expected ValueError for non-rectangular matrix"
    except ValueError as e:
        expected = (
            "All rows of M must have the same length; row 0 has length 2, "
            "row 1 has length 1."
        )
        assert str(e) == expected

def test_multiply_matrices_basic():
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[7, 8], [9, 10], [11, 12]]
    result = multiply_matrices(a, b)
    assert result == [[58, 64], [139, 154]]

def test_multiply_by_identity_returns_same_matrix():
    a = [[5, 6], [7, 8]]
    identity = [[1, 0], [0, 1]]
    assert multiply_matrices(a, identity) == a
    assert multiply_matrices(identity, a) == a

def test_multiply_with_zero_matrix():
    a = [[1, 2, 3], [4, 5, 6]]  # 2x3
    zero = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # 3x4
    result = multiply_matrices(a, zero)
    assert result == [[0, 0, 0, 0], [0, 0, 0, 0]]

def test_multiply_with_negative_numbers():
    a = [[-1, 2], [3, -4]]
    b = [[5, -6], [-7, 8]]
    # compute expected manually
    expected = [
        [(-1) * 5 + 2 * (-7), (-1) * (-6) + 2 * 8],
        [3 * 5 + (-4) * (-7), 3 * (-6) + (-4) * 8],
    ]
    assert multiply_matrices(a, b) == expected

def test_multiply_incompatible_dimensions_error():
    a = [[1, 2, 3], [4, 5, 6]]  # 2x3
    b = [[1, 2], [3, 4], [5, 6], [7, 8]]  # 4x2
    try:
        multiply_matrices(a, b)
        assert False, "Expected ValueError for incompatible dimensions"
    except ValueError as e:
        assert str(e) == "Incompatible dimensions: A has 3 columns but B has 4 rows."

def test_multiply_propagates_validation_error_from_a():
    # A contains a non-integer element -> validation for "A" should fail
    a = [[1, "x"]]
    b = [[1], [2]]
    try:
        multiply_matrices(a, b)
        assert False, "Expected ValueError from validating A"
    except ValueError as e:
        assert str(e) == "All elements of A must be integers."

def test_multiply_propagates_validation_error_from_b():
    # B has rows that are not lists -> validation for "B" should fail
    a = [[1, 2]]
    b = [1, 2]  # invalid
    try:
        multiply_matrices(a, b)
        assert False, "Expected ValueError from validating B"
    except ValueError as e:
        assert str(e) == "All rows of B must be lists."