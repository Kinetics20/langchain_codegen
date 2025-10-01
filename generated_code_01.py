from typing import List, Tuple

Matrix = List[List[int]]


def _validate_matrix(matrix: Matrix, name: str) -> Tuple[int, int]:
    """Validate that a matrix is a non-empty rectangular list of ints.

    Raises:
        ValueError: If the matrix is empty, has empty rows, or rows of differing lengths.
        TypeError: If the matrix is not a list, any row is not a list, or any element is not an int.

    Returns:
        A tuple (rows, cols) with the matrix dimensions.
    """
    if not isinstance(matrix, list):
        raise TypeError(f"{name} must be a list of lists of ints.")
    if not matrix:
        raise ValueError(f"{name} must be a non-empty list of rows.")
    if any(not isinstance(row, list) for row in matrix):
        raise TypeError(f"All rows of {name} must be lists.")
    cols = len(matrix[0])
    if cols == 0:
        raise ValueError(f"Rows of {name} must not be empty.")
    for row in matrix:
        if len(row) != cols:
            raise ValueError(f"All rows of {name} must have equal length.")
        for element in row:
            if not isinstance(element, int):
                raise TypeError(f"All elements of {name} must be ints.")
    return len(matrix), cols


def multiply_matrices(a: Matrix, b: Matrix) -> Matrix:
    """Multiply two matrices of integers and return the product.

    Both matrices must be well-formed (rectangular lists of ints) and
    have compatible dimensions: number of columns of `a` must equal
    number of rows of `b`.
    """
    a_rows, a_cols = _validate_matrix(a, "a")
    b_rows, b_cols = _validate_matrix(b, "b")
    if a_cols != b_rows:
        raise ValueError(
            "Incompatible dimensions: number of columns of 'a' must equal number of rows of 'b'."
        )
    # Transpose b so we can iterate columns as rows for efficient zip.
    b_transposed: Matrix = [list(col) for col in zip(*b)]
    product: Matrix = [
        [sum(x * y for x, y in zip(row_a, col_b)) for col_b in b_transposed]
        for row_a in a
    ]
    return product