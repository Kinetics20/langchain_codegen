from typing import List, Tuple

Matrix = List[List[int]]


def _validate_matrix(matrix: Matrix, name: str) -> Tuple[int, int]:
    """
    Validate that a matrix is well-formed and return its dimensions.

    A well-formed matrix is a non-empty list of non-empty rows where each
    row has the same number of integer elements.

    Args:
        matrix: The matrix to validate.
        name: Human-readable name used in error messages.

    Returns:
        A tuple (rows, cols) with the matrix dimensions.

    Raises:
        ValueError: If the matrix is malformed.
    """
    if not isinstance(matrix, list) or not matrix:
        raise ValueError(f"{name} must be a non-empty list of rows.")
    if any(not isinstance(row, list) or not row for row in matrix):
        raise ValueError(f"{name} must contain non-empty rows as lists.")
    row_length = len(matrix[0])
    for row in matrix:
        if len(row) != row_length:
            raise ValueError(f"All rows in {name} must have the same length.")
        for item in row:
            if not isinstance(item, int):
                raise ValueError(f"All elements of {name} must be integers.")
    return len(matrix), row_length


def multiply_matrices(matrix_a: Matrix, matrix_b: Matrix) -> Matrix:
    """
    Multiply two matrices of integers and return the product.

    The function validates that both matrices are well-formed and that the
    number of columns in the first matrix equals the number of rows in
    the second matrix.

    Args:
        matrix_a: Left-hand matrix as a list of integer rows.
        matrix_b: Right-hand matrix as a list of integer rows.

    Returns:
        The product matrix as a list of integer rows.

    Raises:
        ValueError: If matrices are malformed or have incompatible sizes.
    """
    a_rows, a_cols = _validate_matrix(matrix_a, "matrix_a")
    b_rows, b_cols = _validate_matrix(matrix_b, "matrix_b")
    if a_cols != b_rows:
        raise ValueError(
            "Incompatible dimensions: matrix_a columns must equal "
            "matrix_b rows."
        )

    # Transpose matrix_b for cache-friendly access to its columns.
    transposed_b: List[List[int]] = [
        [matrix_b[r][c] for r in range(b_rows)] for c in range(b_cols)
    ]

    result: Matrix = [
        [0 for _ in range(b_cols)] for _ in range(a_rows)
    ]

    for i in range(a_rows):
        row_a = matrix_a[i]
        for j in range(b_cols):
            col_b = transposed_b[j]
            # Compute dot product of row_a and col_b.
            total = 0
            for k in range(a_cols):
                total += row_a[k] * col_b[k]
            result[i][j] = total

    return result