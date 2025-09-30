import pytest
from generated_code import merge_sort, merge

def test_merge_sort_empty():
    assert merge_sort([]) == []

def test_merge_sort_single_element():
    assert merge_sort([1]) == [1]
    assert merge_sort([-1]) == [-1]
    assert merge_sort([0]) == [0]

def test_merge_sort_two_elements():
    assert merge_sort([2, 1]) == [1, 2]
    assert merge_sort([-1, 1]) == [-1, 1]
    assert merge_sort([1, 1]) == [1, 1]

def test_merge_sort_multiple_elements():
    assert merge_sort([3, 2, 1]) == [1, 2, 3]
    assert merge_sort([1, 3, 2]) == [1, 2, 3]
    assert merge_sort([1, 2, 3]) == [1, 2, 3]
    assert merge_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]
    assert merge_sort([1, 2, 3, 2, 1]) == [1, 1, 2, 2, 3]

def test_merge_sort_with_negatives():
    assert merge_sort([-1, -3, -2]) == [-3, -2, -1]
    assert merge_sort([-1, 0, 1]) == [-1, 0, 1]

def test_merge_sort_with_duplicates():
    assert merge_sort([1, 2, 2, 1]) == [1, 1, 2, 2]
    assert merge_sort([3, 3, 2, 1, 1]) == [1, 1, 2, 3, 3]

def test_merge_sort_large_numbers():
    assert merge_sort([1000000, 500000, 0]) == [0, 500000, 1000000]
    assert merge_sort([-1000000, 0, 1000000]) == [-1000000, 0, 1000000]

def test_merge_sort_large_list():
    assert merge_sort(list(range(1000, 0, -1))) == list(range(1, 1001))