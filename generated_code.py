def factorial(n: int) -> int:
    """
    Compute the factorial of a given number.

    Args:
    n: An integer for which factorial needs to be computed.

    Returns:
    The factorial of the input number.
    """
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

if __name__ == "__main__":
    num = 5
    print(f'The factorial of {num} is {factorial(num)}')