import pytest
from generated_code import _validate_matrix, multiply_matrices

def test_validate_matrix_valid_single():
    dims = _validate_matrix([[5]], "single")
    assert dims == (1, 1)

def test_validate_matrix_valid_rectangular():
    matrix = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    dims = _validate_matrix(matrix, "rect")
    assert dims == (2, 3)

def test_validate_matrix_non_list_raises_value_error():
    try:
        _validate_matrix("not a matrix", "bad")
        assert False, "Expected ValueError for non-list matrix"
    except Exception as e:
        assert isinstance(e, ValueError)
        assert "must be a non-empty list of rows" in str(e)

def test_validate_matrix_empty_list_raises_value_error():
    try:
        _validate_matrix([], "empty")
        assert False, "Expected ValueError for empty matrix"
    except Exception as e:
        assert isinstance(e, ValueError)
        assert "must be a non-empty list of rows" in str(e)

def test_validate_matrix_row_not_list_or_empty_raises_value_error():
    # One row is an empty list
    try:
        _validate_matrix([[1, 2], []], "rows")
        assert False, "Expected ValueError for empty row"
    except Exception as e:
        assert isinstance(e, ValueError)
        assert "must contain non-empty rows as lists" in str(e)

    # One row is not a list
    try:
        _validate_matrix([[1, 2], "not a row"], "rows")
        assert False, "Expected ValueError for non-list row"
    except Exception as e:
        assert isinstance(e, ValueError)
        assert "must contain non-empty rows as lists" in str(e)

def test_validate_matrix_inconsistent_row_length_raises_value_error():
    try:
        _validate_matrix([[1, 2], [3, 4, 5]], "inconsistent")
        assert False, "Expected ValueError for inconsistent row lengths"
    except Exception as e:
        assert isinstance(e, ValueError)
        assert "All rows in inconsistent must have the same length" in str(e)

def test_validate_matrix_non_integer_elements_raises_value_error():
    try:
        _validate_matrix([[1, 2], [3, "a"]], "contains_non_int")
        assert False, "Expected ValueError for non-integer element"
    except Exception as e:
        assert isinstance(e, ValueError)
        assert "All elements of contains_non_int must be integers" in str(e)

def test_multiply_single_elements():
    result = multiply_matrices([[2]], [[3]])
    assert result == [[6]]

def test_multiply_2x3_by_3x2():
    a = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    b = [
        [7, 8],
        [9, 10],
        [11, 12]
    ]
    expected = [
        [58, 64],
        [139, 154]
    ]
    assert multiply_matrices(a, b) == expected

def test_multiply_1x3_by_3x1():
    a = [[1, 0, -1]]
    b = [[2], [3], [4]]
    # 1*2 + 0*3 + (-1)*4 = -2
    assert multiply_matrices(a, b) == [[-2]]

def test_multiply_incompatible_dimensions_raises_value_error():
    a = [
        [1, 2],
        [3, 4]
    ]  # 2x2
    b = [
        [1, 2, 3]
    ]  # 1x3
    try:
        multiply_matrices(a, b)
        assert False, "Expected ValueError for incompatible dimensions"
    except Exception as e:
        assert isinstance(e, ValueError)
        assert "Incompatible dimensions" in str(e)

def test_multiply_with_malformed_matrices_raises_value_error():
    # matrix_a malformed
    try:
        multiply_matrices([], [[1]])
        assert False, "Expected ValueError for malformed matrix_a"
    except Exception as e:
        assert isinstance(e, ValueError)
        assert "must be a non-empty list of rows" in str(e)

    # matrix_b malformed
    try:
        multiply_matrices([[1]], [["bad"]])
        assert False, "Expected ValueError for malformed matrix_b"
    except Exception as e:
        assert isinstance(e, ValueError)
        assert "must be a non-empty list of rows" in str(e) or "must contain non-empty rows as lists" in str(e) or "must be integers" in str(e)

def test_multiply_with_identity_returns_same_matrix():
    matrix = [
        [3, 5, 2],
        [1, 0, -1],
        [4, 2, 6]
    ]
    identity = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    assert multiply_matrices(matrix, identity) == matrix
    assert multiply_matrices(identity, matrix) == matrix