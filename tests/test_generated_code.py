import pytest
from generated_code import *

def test_merge_sort_empty():
    assert merge_sort([]) == []

def test_merge_sort_single_element():
    assert merge_sort([1]) == [1]
    assert merge_sort([-1]) == [-1]
    assert merge_sort([0]) == [0]

def test_merge_sort_two_elements_sorted():
    assert merge_sort([1, 2]) == [1, 2]
    assert merge_sort([-1, 0]) == [-1, 0]

def test_merge_sort_two_elements_unsorted():
    assert merge_sort([2, 1]) == [1, 2]
    assert merge_sort([0, -1]) == [-1, 0]

def test_merge_sort_multiple_elements():
    assert merge_sort([3, 1, 2]) == [1, 2, 3]
    assert merge_sort([5, 3, 8, 6, 2]) == [2, 3, 5, 6, 8]
    assert merge_sort([-1, -3, 2, 0, 1]) == [-3, -1, 0, 1, 2]

def test_merge_sort_duplicates():
    assert merge_sort([1, 2, 2, 1]) == [1, 1, 2, 2]
    assert merge_sort([3, 3, 3]) == [3, 3, 3]

def test_merge_sort_negative_numbers():
    assert merge_sort([-5, -1, -3, -2]) == [-5, -3, -2, -1]

def test_merge_sort_large_numbers():
    assert merge_sort([1000000, 500000, 2000000]) == [500000, 1000000, 2000000]

def test_merge_sort_large_list():
    assert merge_sort(list(range(1000, 0, -1))) == list(range(1, 1001))

def test_merge_sort_identical_elements():
    assert merge_sort([7] * 100) == [7] * 100