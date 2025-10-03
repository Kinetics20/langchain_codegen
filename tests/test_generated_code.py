import pytest
from generated_code import _validate_matrix, multiply_matrices

def test_validate_matrix_returns_dimensions_for_rectangular_matrix():
    m = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    rows, cols = _validate_matrix(m, "Matrix")
    assert (rows, cols) == (2, 3)

def test_validate_matrix_single_row_and_single_col():
    m1 = [[42]]
    assert _validate_matrix(m1, "M1") == (1, 1)
    m2 = [[1, 2, 3]]
    assert _validate_matrix(m2, "M2") == (1, 3)
    m3 = [[1], [2], [3]]
    assert _validate_matrix(m3, "M3") == (3, 1)

def test_validate_matrix_raises_for_non_list_matrix():
    with __import__('pytest').raises(ValueError):
        _validate_matrix(None, "X")
    with __import__('pytest').raises(ValueError):
        _validate_matrix(123, "X")
    with __import__('pytest').raises(ValueError):
        _validate_matrix("not a matrix", "X")

def test_validate_matrix_raises_for_empty_matrix_or_empty_first_row():
    with __import__('pytest').raises(ValueError):
        _validate_matrix([], "Empty")
    with __import__('pytest').raises(ValueError):
        _validate_matrix([[]], "EmptyRow")

def test_validate_matrix_raises_if_row_is_not_list():
    bad = [[1, 2], "not a list"]
    with __import__('pytest').raises(ValueError):
        _validate_matrix(bad, "Bad")

def test_validate_matrix_raises_for_inconsistent_row_lengths():
    bad = [[1, 2, 3], [4, 5]]
    with __import__('pytest').raises(ValueError):
        _validate_matrix(bad, "Inconsistent")

def test_validate_matrix_raises_for_non_integer_items():
    bad1 = [[1, 2], [3, 4.5]]
    with __import__('pytest').raises(ValueError):
        _validate_matrix(bad1, "NonInt")
    bad2 = [[1, "2"]]
    with __import__('pytest').raises(ValueError):
        _validate_matrix(bad2, "NonInt2")

def test_multiply_matrices_basic_2x3_by_3x2():
    a = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    b = [
        [7, 8],
        [9, 10],
        [11, 12],
    ]
    result = multiply_matrices(a, b)
    assert result == [
        [58, 64],
        [139, 154],
    ]

def test_multiply_matrices_1x1():
    a = [[6]]
    b = [[7]]
    assert multiply_matrices(a, b) == [[42]]

def test_multiply_matrices_row_by_column_produces_1x1():
    a = [[1, 2, 3]]      # 1x3
    b = [[4], [5], [6]]  # 3x1
    assert multiply_matrices(a, b) == [[32]]  # 1*4 + 2*5 + 3*6 = 32

def test_multiply_matrices_identity_preserves_matrix():
    identity = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    mat = [
        [2, 3, 5],
        [7, 11, 13],
        [17, 19, 23],
    ]
    assert multiply_matrices(identity, mat) == mat
    assert multiply_matrices(mat, identity) == mat

def test_multiply_matrices_with_negatives_and_zeros():
    a = [
        [0, -1],
        [2, 3],
    ]
    b = [
        [4, 0],
        [-5, 6],
    ]
    # manual calculation:
    # [0*4 + -1*-5, 0*0 + -1*6] = [5, -6]
    # [2*4 + 3*-5, 2*0 + 3*6] = [8 -15 = -7, 18]
    assert multiply_matrices(a, b) == [
        [5, -6],
        [-7, 18],
    ]

def test_multiply_matrices_raises_on_incompatible_dimensions():
    a = [
        [1, 2],
        [3, 4],
    ]  # 2x2
    b = [
        [5],
        [6],
        [7],
    ]  # 3x1
    with __import__('pytest').raises(ValueError):
        multiply_matrices(a, b)

def test_multiply_matrices_uses_validation_on_inputs():
    # invalid second matrix should cause validation error to bubble up
    a = [[1]]
    b = [[]]  # invalid: empty row
    with __import__('pytest').raises(ValueError):
        multiply_matrices(a, b)

def test_multiply_matrices_does_not_mutate_inputs_and_returns_new_structure():
    a = [
        [1, 2],
        [3, 4],
    ]
    b = [
        [5, 6],
        [7, 8],
    ]
    a_copy = [row[:] for row in a]
    b_copy = [row[:] for row in b]
    result = multiply_matrices(a, b)
    # inputs unchanged
    assert a == a_copy
    assert b == b_copy
    # result is a new nested list (mutating result should not affect inputs)
    original_result_val = result[0][0]
    result[0][0] = 99999
    assert a[0][0] == a_copy[0][0]
    assert b[0][0] == b_copy[0][0]
    # restore and verify original_result_val was a valid integer product
    result[0][0] = original_result_val
    expected = [
        [1*5 + 2*7, 1*6 + 2*8],
        [3*5 + 4*7, 3*6 + 4*8],
    ]
    assert result == expected