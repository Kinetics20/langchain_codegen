import pytest
from generated_code_mix import *

def test_multiply_basic_2x3_by_3x2():
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
    result = multiply_matrices(a, b)
    assert result == expected

def test_multiply_identity_2x2():
    a = [
        [1, 2],
        [3, 4],
    ]
    identity = [
        [1, 0],
        [0, 1],
    ]
    assert multiply_matrices(a, identity) == a
    assert multiply_matrices(identity, a) == a

def test_multiply_1x1():
    a = [[5]]
    b = [[-3]]
    assert multiply_matrices(a, b) == [[-15]]

def test_multiply_with_negative_and_zero():
    a = [
        [0, -1],
        [2, 3],
    ]
    b = [
        [4, -2],
        [1, 5],
    ]
    expected = [
        [-1, -5],
        [11, 11],
    ]
    assert multiply_matrices(a, b) == expected

def test_multiply_both_empty_returns_empty():
    assert multiply_matrices([], []) == []

def test_multiply_empty_a_nonempty_raises():
    a = []
    b = [[1]]
    try:
        multiply_matrices(a, b)
        assert False, "Expected ValueError for empty left operand with non-empty right operand"
    except ValueError:
        pass

def test_multiply_nonempty_a_empty_raises():
    a = [[1]]
    b = []
    try:
        multiply_matrices(a, b)
        assert False, "Expected ValueError for non-empty left operand with empty right operand"
    except ValueError:
        pass

def test_multiply_row_with_zero_columns_and_empty_b_returns_1x0():
    # a is 1x0, b is 0x0 -> result should be 1x0 ([[]])
    a = [[]]
    b = []
    result = multiply_matrices(a, b)
    assert result == [[]]

def test_multiply_1x0_by_1x0_raises_due_to_incompatible_shapes():
    # a is 1x0, b is 1x0 -> since a_cols(0) != b_rows(1) should raise
    a = [[]]
    b = [[]]
    try:
        multiply_matrices(a, b)
        assert False, "Expected ValueError for incompatible shapes (1x0 * 1x0)"
    except ValueError:
        pass

def test_validate_matrix_non_list_raises():
    try:
        _validate_matrix(123)
        assert False, "Expected ValueError for non-list matrix"
    except ValueError:
        pass

def test_validate_matrix_row_not_list_raises():
    try:
        _validate_matrix([1, 2, 3])  # rows are ints, not lists
        assert False, "Expected ValueError when rows are not lists"
    except ValueError:
        pass

def test_validate_matrix_inconsistent_row_length_raises():
    try:
        _validate_matrix([[1, 2], [3]])
        assert False, "Expected ValueError for rows with inconsistent lengths"
    except ValueError:
        pass

def test_validate_matrix_non_int_entry_raises():
    try:
        _validate_matrix([[1, "a"]])
        assert False, "Expected ValueError for non-int matrix entry"
    except ValueError:
        pass

def test_multiply_incompatible_shapes_raises():
    a = [
        [1, 2, 3],
        [4, 5, 6],
    ]  # 2x3
    b = [
        [1, 2],
        [3, 4],
    ]  # 2x2 -> incompatible
    try:
        multiply_matrices(a, b)
        assert False, "Expected ValueError for incompatible shapes (2x3 * 2x2)"
    except ValueError:
        pass