import pytest
from generated_code import _validate_matrix, multiply_matrices

def test_validate_matrix_valid_rectangular():
    mat = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    rows, cols = _validate_matrix(mat, "M")
    assert rows == 2
    assert cols == 3

def test_validate_matrix_minimum_size():
    mat = [[42]]
    rows, cols = _validate_matrix(mat, "Small")
    assert rows == 1
    assert cols == 1

def test_validate_matrix_non_list_outer_raises():
    with pytest.raises(ValueError):
        _validate_matrix(None, "N")
    with pytest.raises(ValueError):
        _validate_matrix(123, "N")

def test_validate_matrix_empty_outer_raises():
    with pytest.raises(ValueError):
        _validate_matrix([], "E")

def test_validate_matrix_first_row_not_list_or_empty_raises():
    with pytest.raises(ValueError):
        _validate_matrix([None], "F")
    with pytest.raises(ValueError):
        _validate_matrix([[]], "F")

def test_validate_matrix_row_not_list_raises():
    with pytest.raises(ValueError):
        _validate_matrix([[1, 2], "not a list"], "R")

def test_validate_matrix_inconsistent_row_lengths_raises():
    with pytest.raises(ValueError):
        _validate_matrix([[1, 2], [3]], "I")

def test_validate_matrix_non_int_value_raises():
    with pytest.raises(ValueError):
        _validate_matrix([[1, 2], [3, 4.5]], "V")
    with pytest.raises(ValueError):
        _validate_matrix([[1, 2], [3, "x"]], "V")

def test_validate_matrix_bool_is_instance_of_int():
    # bool is a subclass of int in Python; the function uses isinstance(value, int)
    # so True/False are accepted as ints here.
    rows, cols = _validate_matrix([[True, False]], "B")
    assert (rows, cols) == (1, 2)

def test_multiply_matrices_basic_case():
    a = [
        [1, 2, 3],
        [4, 5, 6],
    ]  # 2x3
    b = [
        [7, 8],
        [9, 10],
        [11, 12],
    ]  # 3x2
    product = multiply_matrices(a, b)
    # Expected:
    # [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [58, 64]
    # [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [139, 154]
    assert product == [
        [58, 64],
        [139, 154],
    ]

def test_multiply_matrices_1x1():
    assert multiply_matrices([[2]], [[3]]) == [[6]]

def test_multiply_row_by_column():
    a = [[1, 2, 3]]  # 1x3
    b = [[4], [5], [6]]  # 3x1
    assert multiply_matrices(a, b) == [[32]]

def test_multiply_with_negatives_and_zeros():
    a = [
        [0, -1, 2],
        [3, 4, 0],
    ]  # 2x3
    b = [
        [1, 0],
        [-2, 5],
        [3, -1],
    ]  # 3x2
    # compute expected manually
    # row0: 0*1 + -1*-2 + 2*3 = 0 + 2 + 6 = 8 ; 0*0 + -1*5 + 2*-1 = 0 -5 -2 = -7
    # row1: 3*1 + 4*-2 + 0*3 = 3 -8 +0 = -5 ; 3*0 + 4*5 + 0*-1 = 0 +20 +0 = 20
    assert multiply_matrices(a, b) == [[8, -7], [-5, 20]]

def test_multiply_incompatible_dimensions_raises():
    a = [
        [1, 2],
        [3, 4],
    ]  # 2x2
    b = [
        [5],
        [6],
        [7],
    ]  # 3x1
    with pytest.raises(ValueError):
        multiply_matrices(a, b)

def test_multiply_invalid_A_raises():
    a = []  # invalid
    b = [[1]]
    with pytest.raises(ValueError):
        multiply_matrices(a, b)

def test_multiply_invalid_B_raises():
    a = [[1]]
    b = []  # invalid
    with pytest.raises(ValueError):
        multiply_matrices(a, b)

def test_multiply_with_non_int_in_matrix_raises():
    a = [[1.0]]  # float not allowed
    b = [[2]]
    with pytest.raises(ValueError):
        multiply_matrices(a, b)
    a = [[1]]
    b = [[2.0]]
    with pytest.raises(ValueError):
        multiply_matrices(a, b)

def test_multiply_with_bool_values_treated_as_ints():
    # As in validation, bools are treated as ints; multiplication should reflect that
    a = [[True, False]]  # [1, 0]
    b = [[3], [4]]  # 2x1
    assert multiply_matrices(a, b) == [[3]]