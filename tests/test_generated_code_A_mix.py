import pytest
import solution


def test_multiply_square_matrices():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    expected = [[19, 22], [43, 50]]
    assert solution.multiply_matrices(a, b) == expected


def test_multiply_rectangular_matrices():
    a = [[1, 2, 3], [4, 5, 6]]  # 2x3
    b = [[7, 8], [9, 10], [11, 12]]  # 3x2
    expected = [[58, 64], [139, 154]]  # 2x2
    assert solution.multiply_matrices(a, b) == expected


def test_multiply_1x1_and_1xN_Nx1():
    # 1x1 * 1x1
    assert solution.multiply_matrices([[2]], [[3]]) == [[6]]
    # 1x3 * 3x1
    a = [[1, -2, 3]]
    b = [[4], [5], [6]]
    # dot = 1*4 + (-2)*5 + 3*6 = 4 -10 +18 = 12
    assert solution.multiply_matrices(a, b) == [[12]]
    # 3x1 * 1x3
    a2 = [[1], [2], [3]]
    b2 = [[7, 8, 9]]
    assert solution.multiply_matrices(a2, b2) == [[7, 8, 9], [14, 16, 18], [21, 24, 27]]


def test_negative_and_zero_values():
    a = [[0, -1], [-2, 3]]
    b = [[4, 0], [1, -1]]
    # manual calculation:
    # row0: [0*4 + -1*1 = -1, 0*0 + -1*-1 = 1]
    # row1: [-2*4 + 3*1 = -8+3=-5, -2*0 + 3*-1 = -3]
    assert solution.multiply_matrices(a, b) == [[-1, 1], [-5, -3]]


def test_inputs_are_not_mutated():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    a_copy = [row[:] for row in a]
    b_copy = [row[:] for row in b]
    _ = solution.multiply_matrices(a, b)
    assert a == a_copy and b == b_copy


def test_type_error_top_level_not_list_a():
    with pytest.raises(TypeError):
        solution.multiply_matrices("not a list", [[1]])


def test_type_error_top_level_not_list_b():
    with pytest.raises(TypeError):
        solution.multiply_matrices([[1]], "not a list")


def test_type_error_row_not_list():
    a = [[1, 2], "not a row"]
    b = [[1], [2]]
    with pytest.raises(TypeError):
        solution.multiply_matrices(a, b)


def test_type_error_element_not_int_float():
    a = [[1.0, 2], [3, 4]]
    b = [[1], [2]]
    with pytest.raises(TypeError):
        solution.multiply_matrices(a, b)


def test_type_error_element_bool():
    # bools should be rejected even though isinstance(True, int) is True
    a = [[1, True], [3, 4]]
    b = [[1], [2]]
    with pytest.raises(TypeError):
        solution.multiply_matrices(a, b)


def test_value_error_empty_matrix_a():
    with pytest.raises(ValueError):
        solution.multiply_matrices([], [[1]])


def test_value_error_empty_matrix_b():
    with pytest.raises(ValueError):
        solution.multiply_matrices([[1]], [])


def test_value_error_empty_row():
    a = [[1, 2], []]
    b = [[1], [2]]
    with pytest.raises(ValueError):
        solution.multiply_matrices(a, b)


def test_value_error_irregular_rows():
    a = [[1, 2], [3]]  # irregular row lengths
    b = [[1], [2]]
    with pytest.raises(ValueError):
        solution.multiply_matrices(a, b)


def test_value_error_incompatible_dimensions():
    # A is 2x3, B is 2x2 -> inner dimensions 3 != 2
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[7, 8], [9, 10]]
    with pytest.raises(ValueError):
        solution.multiply_matrices(a, b)