import pytest
from generated_code_agent import _validate_matrix, multiply_matrices

def test_validate_matrix_valid():
    m = [[1, 2], [3, 4]]
    result = _validate_matrix(m, "m")
    assert result is None

def test_validate_matrix_not_list_raises_typeerror():
    try:
        _validate_matrix(None, "m")
        assert False, "Expected TypeError for non-list matrix"
    except Exception as e:
        assert isinstance(e, TypeError)
        assert "must be a list of lists of ints" in str(e)

def test_validate_matrix_empty_matrix_raises_valueerror():
    try:
        _validate_matrix([], "m")
        assert False, "Expected ValueError for empty matrix"
    except Exception as e:
        assert isinstance(e, ValueError)
        assert "must have at least one row" in str(e)

def test_validate_matrix_row_not_list_raises_typeerror():
    try:
        _validate_matrix([1, 2, 3], "m")
        assert False, "Expected TypeError for a row that is not a list"
    except Exception as e:
        assert isinstance(e, TypeError)
        assert "m[0] must be a list" in str(e)

def test_validate_matrix_empty_row_raises_valueerror():
    try:
        _validate_matrix([[]], "m")
        assert False, "Expected ValueError for an empty row"
    except Exception as e:
        assert isinstance(e, ValueError)
        assert "m[0] must have at least one column" in str(e)

def test_validate_matrix_inconsistent_row_lengths_raises_valueerror():
    try:
        _validate_matrix([[1, 2], [3]], "m")
        assert False, "Expected ValueError for inconsistent row lengths"
    except Exception as e:
        assert isinstance(e, ValueError)
        assert "All rows of m must have the same length" in str(e)

def test_validate_matrix_element_not_int_raises_typeerror():
    try:
        _validate_matrix([[1, "a"]], "m")
        assert False, "Expected TypeError for non-int element"
    except Exception as e:
        assert isinstance(e, TypeError)
        assert "must be an int" in str(e)
        assert "str" in str(e)

def test_multiply_matrices_1x1():
    a = [[3]]
    b = [[4]]
    assert multiply_matrices(a, b) == [[12]]

def test_multiply_matrices_2x3_by_3x2():
    a = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    b = [
        [7, 8],
        [9, 10],
        [11, 12],
    ]
    expected = [
        [58, 64],
        [139, 154],
    ]
    assert multiply_matrices(a, b) == expected

def test_multiply_by_identity_returns_same_matrix():
    a = [
        [5, -2, 3],
        [0, 7, 1],
    ]
    identity = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    result = multiply_matrices(a, identity)
    assert result == a

def test_multiply_with_zero_matrix_results_zero_matrix():
    a = [
        [1, 2],
        [3, 4],
    ]
    zero = [
        [0, 0],
    ]
    # adjust to valid dimensions: a is 2x2, zero must be 2xP to multiply; test a * zero(2x2)
    zero2 = [
        [0, 0],
        [0, 0],
    ]
    result = multiply_matrices(a, zero2)
    assert result == [[0, 0], [0, 0]]

def test_multiply_vector_forms():
    a = [[1, 2, 3]]      # 1x3
    b = [[4], [5], [6]]  # 3x1
    assert multiply_matrices(a, b) == [[32]]
    # Nx1 times 1xM -> NxM
    c = [[2], [3]]       # 2x1
    d = [[7, 8]]         # 1x2
    assert multiply_matrices(c, d) == [[14, 16], [21, 24]]

def test_multiply_incompatible_dimensions_raises_valueerror():
    a = [[1, 2]]  # 1x2
    b = [[1, 2]]  # 1x2 -> incompatible (a_cols=2, b_rows=1)
    try:
        multiply_matrices(a, b)
        assert False, "Expected ValueError for incompatible dimensions"
    except Exception as e:
        assert isinstance(e, ValueError)
        assert "Incompatible dimensions" in str(e)

def test_multiply_raises_typeerror_for_invalid_input():
    a = "not a matrix"
    b = [[1]]
    try:
        multiply_matrices(a, b)
        assert False, "Expected TypeError for invalid input 'a'"
    except Exception as e:
        assert isinstance(e, TypeError)
    # invalid 'b'
    try:
        multiply_matrices([[1]], None)
        assert False, "Expected TypeError for invalid input 'b'"
    except Exception as e:
        assert isinstance(e, TypeError)

def test_multiply_does_not_mutate_inputs():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    # make manual deep copies to compare after operation
    a_copy = [row[:] for row in a]
    b_copy = [row[:] for row in b]
    _ = multiply_matrices(a, b)
    assert a == a_copy
    assert b == b_copy

def test_multiply_with_negative_numbers():
    a = [[-1, 2], [3, -4]]
    b = [[5, -6], [-7, 8]]
    # compute expected manually
    expected = [
        [(-1) * 5 + 2 * (-7), (-1) * -6 + 2 * 8],
        [3 * 5 + (-4) * (-7), 3 * -6 + (-4) * 8],
    ]
    assert multiply_matrices(a, b) == expected