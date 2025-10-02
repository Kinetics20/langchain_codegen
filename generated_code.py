from typing import List, Tuple

Matrix = List[List[int]]


def _validate_matrix(matrix: Matrix, name: str) -> Tuple[int, int]:
    """
    Validate that a matrix is a non-empty rectangular list of lists of ints.

    Returns:
        A tuple of (number_of_rows, number_of_columns).

    Raises:
        TypeError: If the outer or inner containers are not lists or if an
            element is not an int.
        ValueError: If the matrix is empty, any row is empty, or rows have
            inconsistent lengths.
    """
    if not isinstance(matrix, list):
        raise TypeError(f"{name} must be a list of lists of ints")
    if not matrix:
        raise ValueError(f"{name} must not be empty")
    row_len: int = -1
    for i, row in enumerate(matrix):
        if not isinstance(row, list):
            raise TypeError(f"{name}[{i}] must be a list of ints")
        if row_len == -1:
            row_len = len(row)
            if row_len == 0:
                raise ValueError(f"{name} rows must not be empty")
        elif len(row) != row_len:
            raise ValueError(
                f"{name} must be rectangular; row {i} has inconsistent "
                "length"
            )
        for j, val in enumerate(row):
            if not isinstance(val, int):
                raise TypeError(f"{name}[{i}][{j}] must be an int")
    return len(matrix), row_len


def multiply_matrices(a: Matrix, b: Matrix) -> Matrix:
    """
    Multiply two integer matrices and return the product matrix.

    Both matrices must be non-empty, rectangular, and contain only ints.
    The number of columns of A must equal the number of rows of B.

    Args:
        a: Left multiplicand matrix.
        b: Right multiplicand matrix.

    Returns:
        The product matrix as a new list of lists of ints.

    Raises:
        TypeError: If input containers or elements have incorrect types.
        ValueError: If matrices are empty, non-rectangular, or
            dimensionally incompatible.
    """
    a_rows, a_cols = _validate_matrix(a, "A")
    b_rows, b_cols = _validate_matrix(b, "B")
    if a_cols != b_rows:
        raise ValueError(
            "A's column count must equal B's row count for multiplication"
        )
    # Transpose B to access columns as rows for efficient inner loops.
    b_t: List[List[int]] = [list(col) for col in zip(*b)]
    result: Matrix = [[0 for _ in range(b_cols)] for _ in range(a_rows)]
    for i in range(a_rows):
        row = a[i]
        for j in range(b_cols):
            col = b_t[j]
            s = 0
            for x, y in zip(row, col):
                s += x * y
            result[i][j] = s
    return result