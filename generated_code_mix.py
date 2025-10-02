from typing import List, Tuple

__all__ = ["_validate_matrix", "multiply_matrices"]

Matrix = List[List[int]]


def _validate_matrix(matrix: Matrix) -> Tuple[int, int]:
    """
    Validate that the matrix is a well-formed list of lists of ints.

    Returns the number of rows and columns.

    Raises ValueError if the matrix is malformed.
    """
    if not isinstance(matrix, list):
        raise ValueError("Matrix must be a list of lists of integers.")
    if not matrix:
        # An empty matrix: zero rows and zero columns.
        return 0, 0
    row_count = len(matrix)
    first_row = matrix[0]
    if not isinstance(first_row, list):
        raise ValueError("Matrix must be a list of lists of integers.")
    col_count = len(first_row)
    for idx, row in enumerate(matrix):
        if not isinstance(row, list):
            raise ValueError("Matrix must be a list of lists of integers.")
        if len(row) != col_count:
            raise ValueError("All rows in the matrix must have equal length.")
        for val in row:
            if not isinstance(val, int):
                raise ValueError("Matrix entries must be integers.")
    return row_count, col_count


def multiply_matrices(a: Matrix, b: Matrix) -> Matrix:
    """
    Multiply two matrices a and b and return the product.

    Both a and b must be lists of lists of integers where each row has
    the same length. The number of columns of a must equal the number
    of rows of b.

    Raises ValueError for malformed matrices or incompatible shapes.
    """
    a_rows, a_cols = _validate_matrix(a)
    b_rows, b_cols = _validate_matrix(b)

    if a_rows == 0 or b_rows == 0:
        # If either matrix has zero rows, determine shape validity.
        if a_cols != b_rows and not (a_rows == 0 and b_rows == 0):
            raise ValueError("Incompatible matrix dimensions for multiply.")
        # Return an empty matrix with appropriate shape.
        return [[0] * b_cols for _ in range(a_rows)]

    if a_cols != b_rows:
        raise ValueError("Number of columns of the first matrix must equal "
                         "number of rows of the second matrix.")

    # Transpose b to iterate over its columns efficiently.
    b_transposed: List[Tuple[int, ...]] = list(zip(*b))

    result: Matrix = [[0] * b_cols for _ in range(a_rows)]
    for i in range(a_rows):
        row_a = a[i]
        for j in range(b_cols):
            col_b = b_transposed[j]
            # Compute dot product of row_a and col_b.
            acc = 0
            for x, y in zip(row_a, col_b):
                acc += x * y
            result[i][j] = acc

    return result