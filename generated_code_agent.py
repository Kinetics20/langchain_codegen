from typing import List, Tuple


Matrix = List[List[int]]


def _matrix_dimensions(matrix: Matrix) -> Tuple[int, int]:
    """
    Return the dimensions (rows, cols) of the matrix.

    Args:
        matrix: A matrix represented as a list of lists of integers.

    Returns:
        A tuple (rows, cols).

    Raises:
        ValueError: If the matrix is empty or not rectangular.
    """
    if not matrix:
        raise ValueError("Matrix must not be empty.")
    row_count = len(matrix)
    col_count = len(matrix[0])
    if col_count == 0:
        raise ValueError("Matrix rows must contain at least one column.")
    for row in matrix:
        if len(row) != col_count:
            raise ValueError("All rows in the matrix must have the same length.")
    return row_count, col_count


def _validate_matrix_entries(matrix: Matrix) -> None:
    """
    Validate that all entries in the matrix are integers.

    Args:
        matrix: A matrix represented as a list of lists of integers.

    Raises:
        TypeError: If any entry is not an integer.
    """
    for row in matrix:
        for value in row:
            if not isinstance(value, int):
                raise TypeError("Matrix entries must be integers.")


def multiply_matrices(a: Matrix, b: Matrix) -> Matrix:
    """
    Multiply two matrices and return the resulting matrix.

    Both input matrices must be rectangular (all rows the same length),
    contain integer entries, and have compatible dimensions for
    multiplication (columns in 'a' == rows in 'b').

    Args:
        a: Left-hand matrix as a list of lists of integers.
        b: Right-hand matrix as a list of lists of integers.

    Returns:
        The product matrix as a list of lists of integers.

    Raises:
        ValueError: If matrices are empty, not rectangular, or have
                    incompatible dimensions.
        TypeError: If any matrix entry is not an integer.
    """
    # Validate shapes and entries
    a_rows, a_cols = _matrix_dimensions(a)
    b_rows, b_cols = _matrix_dimensions(b)
    _validate_matrix_entries(a)
    _validate_matrix_entries(b)

    if a_cols != b_rows:
        raise ValueError(
            f"Incompatible dimensions for multiplication: "
            f"a is {a_rows}x{a_cols}, b is {b_rows}x{b_cols}."
        )

    # Transpose b to improve cache locality during multiplication.
    b_t: Matrix = [[b[row][col] for row in range(b_rows)] for col in range(b_cols)]

    # Compute product
    result: Matrix = [
        [
            sum(a[i][k] * b_t[j][k] for k in range(a_cols))
            for j in range(b_cols)
        ]
        for i in range(a_rows)
    ]

    return result