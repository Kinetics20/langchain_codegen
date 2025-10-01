import pytest
from generated_code import *

def test_multiply_basic():
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

def test_multiply_identity():
    a = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    m = [
        [5, -2, 3],
        [0, 7, 1],
        [4, 0, -6],
    ]
    assert multiply_matrices(a, m) == m
    assert multiply_matrices(m, a) == m

def test_multiply_1x1_and_vector_cases():
    # 1x1 * 1x1
    assert multiply_matrices([[5]], [[3]]) == [[15]]

    # 1x3 * 3x1 -> 1x1
    a = [[1, 2, 3]]
    b = [[4], [5], [6]]
    assert multiply_matrices(a, b) == [[32]]

    # 3x1 * 1x3 -> 3x3
    a2 = [[1], [2], [3]]
    b2 = [[7, 8, 9]]
    assert multiply_matrices(a2, b2) == [
        [7, 8, 9],
        [14, 16, 18],
        [21, 24, 27],
    ]

def test_multiply_with_negatives_and_zeros():
    a = [
        [0, -1],
        [-2, 3],
    ]
    b = [
        [4, 0],
        [1, -5],
    ]
    # Manually computed:
    # row0 dot col0 = 0*4 + -1*1 = -1
    # row0 dot col1 = 0*0 + -1*-5 = 5
    # row1 dot col0 = -2*4 + 3*1 = -8 + 3 = -5
    # row1 dot col1 = -2*0 + 3*-5 = -15
    assert multiply_matrices(a, b) == [
        [-1, 5],
        [-5, -15],
    ]

def test_invalid_a_not_list_raises_typeerror():
    try:
        multiply_matrices("not a matrix", [[1]])
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError for non-list 'a'"

def test_invalid_b_not_list_raises_typeerror():
    try:
        multiply_matrices([[1]], "not a matrix")
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError for non-list 'b'"

def test_empty_matrix_raises_valueerror():
    try:
        multiply_matrices([], [[1]])
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for empty matrix 'a'"

    try:
        multiply_matrices([[1]], [])
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for empty matrix 'b'"

def test_row_not_list_raises_typeerror():
    try:
        multiply_matrices([[1, 2], 3], [[1]])
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError when a row is not a list"

    try:
        multiply_matrices([[1]], [[1, 2], 3])
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError when b row is not a list"

def test_empty_row_raises_valueerror():
    try:
        multiply_matrices([[]], [[1]])
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for empty rows in 'a'"

    try:
        multiply_matrices([[1]], [[]])
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for empty rows in 'b'"

def test_non_rectangular_raises_valueerror():
    try:
        multiply_matrices([[1, 2], [3, 4, 5]], [[1, 2], [3, 4], [5, 6]])
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for non-rectangular matrix 'a'"

    try:
        multiply_matrices([[1, 2, 3], [4, 5, 6]], [[1, 2], [3], [4, 5]])
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for non-rectangular matrix 'b'"

def test_non_int_entries_raises_typeerror():
    try:
        multiply_matrices([[1.0, 2], [3, 4]], [[1]])
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError for float in 'a'"

    try:
        multiply_matrices([[1]], [["2"]])
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError for non-int in 'b'"

def test_incompatible_dimensions_raises_valueerror():
    a = [
        [1, 2, 3],
        [4, 5, 6],
    ]  # 2x3
    b = [
        [7, 8],
        [9, 10],
    ]  # 2x2 -> incompatible
    try:
        multiply_matrices(a, b)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for incompatible dimensions"

def test_inputs_not_mutated_by_operation():
    a = [
        [1, 2],
        [3, 4],
    ]
    b = [
        [5, 6],
        [7, 8],
    ]
    # shallow copy of rows and elements to compare after operation
    a_copy = [row[:] for row in a]
    b_copy = [row[:] for row in b]
    _ = multiply_matrices(a, b)
    assert a == a_copy, "Input matrix 'a' was mutated"
    assert b == b_copy, "Input matrix 'b' was mutated"