import pytest
from generated_code import _validate_matrix, multiply_matrices

def test_validate_matrix_single_element():
    assert _validate_matrix([[42]], "M") == (1, 1)

def test_validate_matrix_rectangular_matrix():
    m = [[1, 2, 3], [4, 5, 6]]
    assert _validate_matrix(m, "X") == (2, 3)

def test_validate_matrix_not_list_raises_type_error():
    with pytest.raises(TypeError) as exc:
        _validate_matrix((1, 2, 3), "M")
    assert "M must be a list of lists of ints" in str(exc.value)

def test_validate_matrix_row_not_list_raises_type_error():
    with pytest.raises(TypeError) as exc:
        _validate_matrix([1, 2, 3], "A")
    assert "A[0] must be a list of ints" in str(exc.value)

def test_validate_matrix_element_not_int_raises_type_error():
    with pytest.raises(TypeError) as exc:
        _validate_matrix([[1, 2.5], [3, 4]], "B")
    assert "B[0][1] must be an int" in str(exc.value)

def test_validate_matrix_empty_matrix_raises_value_error():
    with pytest.raises(ValueError) as exc:
        _validate_matrix([], "Empty")
    assert "Empty must not be empty" in str(exc.value)

def test_validate_matrix_empty_row_raises_value_error():
    with pytest.raises(ValueError) as exc:
        _validate_matrix([[]], "R")
    assert "R rows must not be empty" in str(exc.value)

def test_validate_matrix_inconsistent_row_lengths_raises_value_error():
    with pytest.raises(ValueError) as exc:
        _validate_matrix([[1, 2], [3]], "I")
    assert "I must be rectangular; row 1 has inconsistent length" in str(exc.value)

def test_multiply_matrices_1x1():
    a = [[2]]
    b = [[3]]
    assert multiply_matrices(a, b) == [[6]]

def test_multiply_matrices_2x3_by_3x2():
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[7, 8], [9, 10], [11, 12]]
    expected = [[58, 64], [139, 154]]
    assert multiply_matrices(a, b) == expected

def test_multiply_matrices_row_by_column_returns_single_value_matrix():
    a = [[1, 2, 3]]
    b = [[4], [5], [6]]
    assert multiply_matrices(a, b) == [[32]]

def test_multiply_matrices_with_negatives_and_zeros():
    a = [[0, -1], [2, 3]]
    b = [[4, 0], [-5, 1]]
    # manual calculation:
    # row0*col0 = 0*4 + -1*-5 = 5
    # row0*col1 = 0*0 + -1*1 = -1
    # row1*col0 = 2*4 + 3*-5 = 8 -15 = -7
    # row1*col1 = 2*0 + 3*1 = 3
    assert multiply_matrices(a, b) == [[5, -1], [-7, 3]]

def test_multiply_matrices_dimension_mismatch_raises_value_error():
    a = [[1, 2], [3, 4]]
    b = [[5], [6], [7]]
    with pytest.raises(ValueError) as exc:
        multiply_matrices(a, b)
    assert "A's column count must equal B's row count" in str(exc.value)

def test_multiply_matrices_propagates_validate_errors():
    # B contains a non-int element -> should raise TypeError from _validate_matrix
    a = [[1, 2]]
    b = [[3, "x"]]
    with pytest.raises(TypeError) as exc:
        multiply_matrices(a, b)
    assert "B[0][1] must be an int" in str(exc.value)

def test_multiply_matrices_empty_matrix_raises_value_error():
    a = [[1, 2]]
    b = []
    with pytest.raises(ValueError) as exc:
        multiply_matrices(a, b)
    assert "B must not be empty" in str(exc.value)