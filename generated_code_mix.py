from typing import List, Tuple


def _validate_matrix(matrix: List[List[int]], name: str) -> Tuple[int, int]:
    """
    Validate that a matrix is a non-empty rectangular list of lists
    of integers and return its dimensions.

    Args:
        matrix: The matrix to validate.
        name: A human-readable name for the matrix used in error messages.

    Returns:
        A tuple (rows, cols) representing the matrix dimensions.

    Raises:
        ValueError: If the matrix is empty, not rectangular, or contains
            non-integer elements.
    """
    if not matrix:
        raise ValueError(f"{name} must be a non-empty matrix.")
    if not all(isinstance(row, list) for row in matrix):
        raise ValueError(f"{name} must be a list of lists.")
    row_len = len(matrix[0])
    if row_len == 0:
        raise ValueError(f"{name} rows must be non-empty.")
    for r_index, row in enumerate(matrix):
        if len(row) != row_len:
            raise ValueError(
                f"All rows in {name} must have the same length; "
                f"row 0 has length {row_len} but row {r_index} "
                f"has length {len(row)}."
            )
        for c_index, item in enumerate(row):
            if not isinstance(item, int):
                raise ValueError(
                    f"All elements of {name} must be integers; "
                    f"found {type(item).__name__} at "
                    f"({r_index}, {c_index})."
                )
    return len(matrix), row_len


def multiply_matrices(A: List[List[int]], B: List[List[int]]
                      ) -> List[List[int]]:
    """
    Multiply two matrices A and B and return the resulting matrix.

    The matrices are represented as lists of lists of integers where each
    inner list is a row. Matrix multiplication follows the rule that
    the number of columns in A must equal the number of rows in B.

    Args:
        A: Left operand matrix.
        B: Right operand matrix.

    Returns:
        The product matrix as a list of lists of integers.

    Raises:
        ValueError: If either matrix is invalid or their dimensions are
            incompatible for multiplication.
    """
    a_rows, a_cols = _validate_matrix(A, "A")
    b_rows, b_cols = _validate_matrix(B, "B")
    if a_cols != b_rows:
        raise ValueError(
            "Incompatible dimensions for multiplication: "
            f"A is {a_rows}x{a_cols}, B is {b_rows}x{b_cols}."
        )

    # Initialize result matrix with zeros.
    result: List[List[int]] = [
        [0 for _ in range(b_cols)] for _ in range(a_rows)
    ]

    # Standard triple-loop matrix multiplication.
    for i in range(a_rows):
        a_row = A[i]
        res_row = result[i]
        for k in range(a_cols):
            a_val = a_row[k]
            b_row_k = B[k]
            # Multiply-accumulate over columns of B.
            for j in range(b_cols):
                res_row[j] += a_val * b_row_k[j]

    return result