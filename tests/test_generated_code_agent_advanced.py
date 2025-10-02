import pytest
from generated_code_agent_advanced import *

def test_validate_matrix_valid():
    assert _validate_matrix([[1, 2], [3, 4]], "Test Matrix") == (2, 2)

def test_validate_matrix_empty_matrix():
    with pytest.raises(ValueError, match="Test Matrix must be a non-empty list of lists."):
        _validate_matrix([], "Test Matrix")

def test_validate_matrix_non_list():
    with pytest.raises(ValueError, match="Test Matrix must be a non-empty list of lists."):
        _validate_matrix("not a list", "Test Matrix")

def test_validate_matrix_empty_row():
    with pytest.raises(ValueError, match="Test Matrix row 0 must be a non-empty list."):
        _validate_matrix([[]], "Test Matrix")

def test_validate_matrix_rows_different_length():
    with pytest.raises(ValueError, match="All rows in Test Matrix must have the same length."):
        _validate_matrix([[1, 2], [3]], "Test Matrix")

def test_validate_matrix_non_integer_element():
    with pytest.raises(ValueError, match="Element Test Matrix[0][1] must be an integer (bool not allowed)."):
        _validate_matrix([[1, True]], "Test Matrix")

def test_validate_matrix_bool_element():
    with pytest.raises(ValueError, match="Element Test Matrix[0][1] must be an integer (bool not allowed)."):
        _validate_matrix([[1, False]], "Test Matrix")

def test_multiply_matrices_valid():
    result = multiply_matrices([[1, 2]], [[3], [4]])
    assert result == [[11]]

def test_multiply_matrices_incompatible_dimensions():
    with pytest.raises(ValueError, match="Incompatible dimensions for multiplication: A is 1x2, B is 2x3."):
        multiply_matrices([[1, 2]], [[3, 4, 5]])

def test_multiply_matrices_invalid_matrix_a():
    with pytest.raises(ValueError, match="Matrix A must be a non-empty list of lists."):
        multiply_matrices([], [[1, 2], [3, 4]])

def test_multiply_matrices_invalid_matrix_b():
    with pytest.raises(ValueError, match="Matrix B must be a non-empty list of lists."):
        multiply_matrices([[1, 2], [3, 4]], [])

def test_multiply_matrices_invalid_element_in_a():
    with pytest.raises(ValueError, match="Element Matrix A[0][1] must be an integer (bool not allowed)."):
        multiply_matrices([[1, True]], [[3], [4]])

def test_multiply_matrices_invalid_element_in_b():
    with pytest.raises(ValueError, match="Element Matrix B[0][1] must be an integer (bool not allowed)."):
        multiply_matrices([[1, 2]], [[3, False], [4, 5]])