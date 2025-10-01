from typing import List


def _validate_matrix(matrix: List[List[int]], name: str) -> None:
    """
    Validate that the given value is a well-formed integer matrix.

    A well-formed matrix is a non-empty list of non-empty lists where each
    row has the same length and every element is an int.

    Args:
        matrix: The matrix to validate.
        name: A human-readable name for the matrix used in error messages.

    Raises:
        TypeError: If the structure or element types are invalid.
        ValueError: If the matrix is empty or rows have inconsistent lengths.
    """
    if not isinstance(matrix, list):
        raise TypeError(f"{name} must be a list of lists of ints.")
    if len(matrix) == 0:
        raise ValueError(f"{name} must have at least one row.")
    row_length = None
    for i, row in enumerate(matrix):
        if not isinstance(row, list):
            raise TypeError(f"{name}[{i}] must be a list.")
        if len(row) == 0:
            raise ValueError(f"{name}[{i}] must have at least one column.")
        if row_length is None:
            row_length = len(row)
        elif len(row) != row_length:
            raise ValueError(
                f"All rows of {name} must have the same length."
            )
        for j, item in enumerate(row):
            if not isinstance(item, int):
                raise TypeError(
                    f"{name}[{i}][{j}] must be an int, got "
                    f"{type(item).__name__}."
                )


def multiply_matrices(a: List[List[int]],
                      b: List[List[int]]) -> List[List[int]]:
    """
    Multiply two matrices of integers and return the product matrix.

    The function validates that both matrices are well-formed and that the
    number of columns in the first matrix equals the number of rows in the
    second matrix.

    Args:
        a: Left matrix as a list of lists of ints (m x n).
        b: Right matrix as a list of lists of ints (n x p).

    Returns:
        The product matrix as a list of lists of ints (m x p).

    Raises:
        TypeError: If inputs are not well-formed integer matrices.
        ValueError: If the matrices have incompatible dimensions.
    """
    _validate_matrix(a, "a")
    _validate_matrix(b, "b")
    a_rows = len(a)
    a_cols = len(a[0])
    b_rows = len(b)
    b_cols = len(b[0])
    if a_cols != b_rows:
        raise ValueError(
            "Incompatible dimensions: a has %d columns but b has %d rows."
            % (a_cols, b_rows)
        )
    # Initialize result matrix with zeros (m x p).
    result: List[List[int]] = [
        [0 for _ in range(b_cols)] for _ in range(a_rows)
    ]
    # Compute multiplication.
    for i in range(a_rows):
        row_a = a[i]
        res_row = result[i]
        for k in range(a_cols):
            a_ik = row_a[k]
            row_b_k = b[k]
            # Multiply-add across columns of b.
            for j in range(b_cols):
                res_row[j] += a_ik * row_b_k[j]
    return result