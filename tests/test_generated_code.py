import pytest
from generated_code import _validate_matrix, multiply_matrices

def test_validate_matrix_valid():
    m = [[1, 2, 3], [4, 5, 6]]
    dims = _validate_matrix(m, "m")
    assert dims == (2, 3)

def test_validate_matrix_single_row():
    m = [[42, -1, 0]]
    assert _validate_matrix(m, "single") == (1, 3)

def test_validate_matrix_single_column():
    m = [[1], [2], [3]]
    assert _validate_matrix(m, "col") == (3, 1)

def test_validate_matrix_empty_matrix_raises():
    try:
        _validate_matrix([], "empty")
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        assert False, "Expected ValueError for empty matrix"

def test_validate_matrix_empty_rows_raises():
    try:
        _validate_matrix([[]], "erow")
    except ValueError as exc:
        assert "must have non-empty rows" in str(exc)
    else:
        assert False, "Expected ValueError for empty rows"

def test_validate_matrix_non_rectangular_raises():
    try:
        _validate_matrix([[1, 2], [3]], "nr")
    except ValueError as exc:
        assert "rectangular" in str(exc)
    else:
        assert False, "Expected ValueError for non-rectangular matrix"

def test_validate_matrix_non_integer_element_raises():
    try:
        _validate_matrix([[1, 2], [3, 4.5]], "nin")
    except ValueError as exc:
        assert "matrix elements must be integers" in str(exc)
        assert "nin[" in str(exc)  # check that the name appears in message
    else:
        assert False, "Expected ValueError for non-integer element"

def test_multiply_matrices_1x1():
    a = [[7]]
    b = [[3]]
    result = multiply_matrices(a, b)
    assert result == [[21]]

def test_multiply_matrices_identity():
    a = [[1, 2], [3, 4]]
    identity = [[1, 0], [0, 1]]
    res_left = multiply_matrices(identity, a)
    res_right = multiply_matrices(a, identity)
    assert res_left == a
    assert res_right == a

def test_multiply_matrices_rectangular():
    a = [
        [1, 2, 3],
        [4, 5, 6]
    ]  # 2x3
    b = [
        [7, 8],
        [9, 10],
        [11, 12]
    ]  # 3x2
    expected = [
        [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12],
        [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]
    ]
    result = multiply_matrices(a, b)
    assert result == expected

def test_multiply_matrices_negative_and_zero():
    a = [[0, -1], [2, 3]]
    b = [[4, 0], [-1, 5]]
    # compute by hand
    expected = [
        [0*4 + -1*-1, 0*0 + -1*5],
        [2*4 + 3*-1, 2*0 + 3*5]
    ]
    assert multiply_matrices(a, b) == expected

def test_multiply_incompatible_dimensions_raises():
    a = [[1, 2], [3, 4]]  # 2x2
    b = [[1], [2], [3]]   # 3x1
    try:
        multiply_matrices(a, b)
    except ValueError as exc:
        assert "incompatible dimensions" in str(exc)
        assert "a has" in str(exc) and "cols" or "rows"  # basic sanity on message content
    else:
        assert False, "Expected ValueError for incompatible dimensions"

def test_multiply_does_not_mutate_inputs():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    a_copy = [row[:] for row in a]
    b_copy = [row[:] for row in b]
    _ = multiply_matrices(a, b)
    assert a == a_copy
    assert b == b_copy

def test_validate_matrix_error_message_indices():
    # ensure that the error message contains the correct index for a non-int
    m = [[1, 2], [3, "x"]]
    try:
        _validate_matrix(m, "testname")
    except ValueError as exc:
        msg = str(exc)
        assert "matrix elements must be integers" in msg
        assert "testname[1][1]" in msg
    else:
        assert False, "Expected ValueError for non-integer at specific index"