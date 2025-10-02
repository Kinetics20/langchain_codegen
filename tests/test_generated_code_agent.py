import pytest
from generated_code_agent import _matrix_dimensions, _validate_matrix_entries, multiply_matrices

def test_matrix_dimensions_rectangular():
    matrix = [[1, 2], [3, 4]]
    assert _matrix_dimensions(matrix) == (2, 2)

def test_matrix_dimensions_single_row_and_single_column():
    assert _matrix_dimensions([[1, 2, 3]]) == (1, 3)
    assert _matrix_dimensions([[1], [2], [3]]) == (3, 1)

def test_matrix_dimensions_empty_raises_value_error():
    try:
        _matrix_dimensions([])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty matrix")

def test_matrix_dimensions_zero_columns_raises_value_error():
    try:
        _matrix_dimensions([[], []])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for rows with zero columns")

def test_matrix_dimensions_non_rectangular_raises_value_error():
    try:
        _matrix_dimensions([[1, 2], [3]])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-rectangular matrix")

def test_validate_matrix_entries_accepts_integers():
    matrix = [[0, -1], [2, 3]]
    # Should not raise
    _validate_matrix_entries(matrix)

def test_validate_matrix_entries_empty_matrix_accepts():
    # Empty matrix should not raise when validating entries
    _validate_matrix_entries([])

def test_validate_matrix_entries_non_integer_raises_type_error():
    try:
        _validate_matrix_entries([[1, 2.5]])
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError for non-integer matrix entry")

def test_multiply_matrices_2x2():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    expected = [[19, 22], [43, 50]]
    assert multiply_matrices(a, b) == expected

def test_multiply_matrices_1x3_by_3x1():
    a = [[1, 2, 3]]
    b = [[4], [5], [6]]
    assert multiply_matrices(a, b) == [[32]]

def test_multiply_matrices_3x2_by_2x4():
    a = [[1, 2], [3, 4], [5, 6]]
    b = [
        [7, 8, 9, 10],
        [11, 12, 13, 14],
    ]
    expected = [
        [29, 32, 35, 38],
        [65, 72, 79, 86],
        [101, 112, 123, 134],
    ]
    assert multiply_matrices(a, b) == expected

def test_multiply_matrices_identity():
    I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    m = [[2, -1, 3], [0, 5, 4], [7, 8, 9]]
    assert multiply_matrices(I, m) == m
    assert multiply_matrices(m, I) == m

def test_multiply_matrices_incompatible_dimensions_raises_value_error():
    a = [[1, 2, 3], [4, 5, 6]]  # 2x3
    b = [[7, 8], [9, 10]]       # 2x2 -> incompatible
    try:
        multiply_matrices(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for incompatible dimensions")

def test_multiply_matrices_non_rectangular_input_raises_value_error():
    a = [[1, 2], [3]]  # non-rectangular
    b = [[1, 2], [3, 4]]
    try:
        multiply_matrices(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-rectangular input matrix")

def test_multiply_matrices_non_integer_entries_raises_type_error():
    a = [[1, 2]]  # 1x2
    b = [[3.0], [4.0]]  # entries are floats -> should raise TypeError
    try:
        multiply_matrices(a, b)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError for non-integer entries in matrices")

def test_multiply_matrices_with_boolean_entries_treated_as_ints():
    # bool is subclass of int in Python; behavior: True==1, False==0
    a = [[True, False]]  # 1x2
    b = [[2], [3]]       # 2x1
    # Expected: 1*2 + 0*3 = 2
    assert multiply_matrices(a, b) == [[2]]