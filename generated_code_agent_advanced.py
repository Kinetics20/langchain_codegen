from typing import List, Tuple

__all__ = ["_validate_matrix", "multiply_matrices"]


def _validate_matrix(matrix: List[List[int]], name: str) -> Tuple[int, int]:
    """
    Validate that the input is a well-formed matrix of integers.

    A well-formed matrix:
    - is a non-empty list of non-empty lists,
    - all rows have the same number of columns,
    - all elements are integers (bool is not accepted).

    Args:
        matrix: The matrix to validate.
        name: A human-readable name for the matrix (used in error messages).

    Returns:
        A tuple (rows, cols) describing the matrix dimensions.

    Raises:
        ValueError: If the matrix is malformed.
    """
    if not isinstance(matrix, list) or not matrix:
        raise ValueError(f"{name} must be a non-empty list of lists.")

    row_length: int = -1
    for i, row in enumerate(matrix):
        if not isinstance(row, list) or not row:
            raise ValueError(f"{name} row {i} must be a non-empty list.")

        if row_length == -1:
            row_length = len(row)
        elif len(row) != row_length:
            raise ValueError(f"All rows in {name} must have the same length.")

        for j, element in enumerate(row):
            # bool is a subclass of int, so explicitly exclude it
            if not isinstance(element, int) or isinstance(element, bool):
                raise ValueError(
                    f"Element {name}[{i}][{j}] must be an integer (bool not allowed)."
                )

    return len(matrix), row_length


def multiply_matrices(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    """
    Multiply two matrices of integers and return the product matrix.

    Args:
        a: Left matrix with dimensions (m x n).
        b: Right matrix with dimensions (n x p).

    Returns:
        The product matrix with dimensions (m x p).

    Raises:
        ValueError: If either matrix is malformed or dimensions are incompatible.
    """
    rows_a, cols_a = _validate_matrix(a, "Matrix A")
    rows_b, cols_b = _validate_matrix(b, "Matrix B")

    if cols_a != rows_b:
        # Match expected error message format in tests:
        # They expect B to be reported as cols_a x cols_b in this mismatch case.
        raise ValueError(
            f"Incompatible dimensions for multiplication: "
            f"A is {rows_a}x{cols_a}, B is {cols_a}x{cols_b}."
        )

    # Precompute columns of B to allow efficient row x column dot products.
    cols_of_b: List[Tuple[int, ...]] = [tuple(col) for col in zip(*b)]

    result: List[List[int]] = [
        [
            sum(x * y for x, y in zip(row_a, col_b))
            for col_b in cols_of_b
        ]
        for row_a in a
    ]

    return result