from typing import List, Tuple

Matrix = List[List[int]]


def _matrix_dimensions(matrix: Matrix) -> Tuple[int, int]:
    """
    Return the number of rows and columns of the matrix.

    Args:
        matrix: A rectangular matrix represented as a list of lists.

    Returns:
        A tuple (rows, cols).

    Raises:
        ValueError: If the matrix is empty or not rectangular.
    """
    if not matrix:
        raise ValueError("Matrix must have at least one row.")
    row_count = len(matrix)
    first_row_len = len(matrix[0])
    if first_row_len == 0:
        raise ValueError("Matrix rows must have at least one column.")
    for row in matrix:
        if len(row) != first_row_len:
            raise ValueError("All rows in the matrix must be the same length.")
    return row_count, first_row_len


def _validate_matrix_entries(matrix: Matrix) -> None:
    """
    Ensure all entries in the matrix are integers.

    Args:
        matrix: A matrix to validate.

    Raises:
        ValueError: If any entry is not an int.
    """
    for row in matrix:
        for value in row:
            if not isinstance(value, int):
                raise ValueError("Matrix entries must be integers.")


def multiply_matrices(a: Matrix, b: Matrix) -> Matrix:
    """
    Multiply two matrices of integers and return the product.

    Both matrices must be non-empty, rectangular, and have compatible
    dimensions. If A is m x n and B is n x p, the result will be
    an m x p matrix.

    Args:
        a: Left matrix (m x n).
        b: Right matrix (n x p).

    Returns:
        The product matrix as a list of lists of integers.

    Raises:
        ValueError: If inputs are invalid or dimensions are incompatible.
    """
    _validate_matrix_entries(a)
    _validate_matrix_entries(b)
    m, n = _matrix_dimensions(a)
    n_b, p = _matrix_dimensions(b)
    if n != n_b:
        raise ValueError(
            "Incompatible dimensions: a is %dx%d, b is %dx%d." % (m, n, n_b, p)
        )
    # Initialize result matrix with zeros.
    result: Matrix = [[0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        row_a = a[i]
        row_res = result[i]
        for k in range(n):
            a_ik = row_a[k]
            row_b_k = b[k]
            for j in range(p):
                row_res[j] += a_ik * row_b_k[j]
    return result