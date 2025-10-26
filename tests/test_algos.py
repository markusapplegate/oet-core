"""Tests for oet_core.algos module."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oet_core.algos import HashMap, binary_search


class TestBinarySearch:
    """Test cases for binary_search function."""

    def test_scalar_list_found(self):
        """Test binary search on a 1D list of scalars."""
        arr = [1, 2, 3, 4, 5]
        assert binary_search(arr, 3) == 2
        assert binary_search(arr, 1) == 0
        assert binary_search(arr, 5) == 4

    def test_scalar_list_not_found(self):
        """Test binary search when target doesn't exist."""
        arr = [1, 2, 3, 4, 5]
        assert binary_search(arr, 0) is None
        assert binary_search(arr, 6) is None
        assert binary_search(arr, 2.5) is None

    def test_pairs_x_only(self):
        """Test binary search on pairs with x-coordinate only."""
        pairs = [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]
        assert binary_search(pairs, 2) == 1
        assert binary_search(pairs, 1) == 0
        assert binary_search(pairs, 4) == 3

    def test_pairs_x_and_y(self):
        """Test binary search on pairs with both x and y coordinates."""
        pairs = [(1, 'a'), (2, 'b'), (2, 'c'), (3, 'd')]
        assert binary_search(pairs, 2, 'b') == 1
        assert binary_search(pairs, 2, 'c') == 2
        assert binary_search(pairs, 2, 'z') is None

    def test_duplicate_x_values(self):
        """Test with multiple entries having the same x value."""
        pairs = [(1, 'a'), (2, 'b'), (2, 'c'), (2, 'd'), (3, 'e')]
        # Without y_target, should return leftmost
        assert binary_search(pairs, 2) == 1
        # With y_target, should find specific match
        assert binary_search(pairs, 2, 'c') == 2
        assert binary_search(pairs, 2, 'd') == 3

    def test_empty_list(self):
        """Test binary search on empty list."""
        assert binary_search([], 1) is None

    def test_single_element(self):
        """Test binary search with single element."""
        assert binary_search([5], 5) == 0
        assert binary_search([5], 3) is None
        assert binary_search([(5, 'a')], 5) == 0
        assert binary_search([(5, 'a')], 5, 'a') == 0
        assert binary_search([(5, 'a')], 5, 'b') is None

    def test_string_values(self):
        """Test binary search with string x values."""
        arr = ['a', 'b', 'c', 'd']
        assert binary_search(arr, 'b') == 1
        assert binary_search(arr, 'z') is None

    def test_invalid_input(self):
        """Test binary search with invalid input."""
        try:
            binary_search("not a list", 1)
            assert False, "Should raise TypeError"
        except TypeError:
            pass


class TestHashMap:
    """Test cases for HashMap class."""

    def test_init(self):
        """Test HashMap initialization."""
        hm = HashMap()
        assert len(hm) == 0
        
        hm2 = HashMap(initial_capacity=16)
        assert len(hm2) == 0

    def test_init_invalid_capacity(self):
        """Test HashMap with invalid initial capacity."""
        try:
            HashMap(initial_capacity=0)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

        try:
            HashMap(initial_capacity=-5)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_put_and_get(self):
        """Test basic put and get operations."""
        hm = HashMap()
        hm.put('key1', 'value1')
        assert hm.get('key1') == 'value1'
        
        hm.put('key2', 123)
        assert hm.get('key2') == 123
        
        hm.put('key3', [1, 2, 3])
        assert hm.get('key3') == [1, 2, 3]

    def test_put_overwrites(self):
        """Test that put overwrites existing keys."""
        hm = HashMap()
        hm.put('key', 'old_value')
        assert hm.get('key') == 'old_value'
        
        hm.put('key', 'new_value')
        assert hm.get('key') == 'new_value'
        assert len(hm) == 1  # Size shouldn't increase

    def test_get_default(self):
        """Test get with default value."""
        hm = HashMap()
        assert hm.get('nonexistent') is None
        assert hm.get('nonexistent', 'default') == 'default'

    def test_delete(self):
        """Test delete operation."""
        hm = HashMap()
        hm.put('key1', 'value1')
        hm.put('key2', 'value2')
        
        assert hm.delete('key1') is True
        assert len(hm) == 1
        assert hm.get('key1') is None
        
        # Deleting non-existent key
        assert hm.delete('key1') is False
        assert hm.delete('key999') is False

    def test_contains(self):
        """Test contains operation."""
        hm = HashMap()
        hm.put('key1', 'value1')
        
        assert hm.contains('key1') is True
        assert hm.contains('key999') is False

    def test_keys(self):
        """Test keys iteration."""
        hm = HashMap()
        hm.put('a', 1)
        hm.put('b', 2)
        hm.put('c', 3)
        
        keys = list(hm.keys())
        assert len(keys) == 3
        assert 'a' in keys
        assert 'b' in keys
        assert 'c' in keys

    def test_values(self):
        """Test values iteration."""
        hm = HashMap()
        hm.put('a', 1)
        hm.put('b', 2)
        hm.put('c', 3)
        
        values = list(hm.values())
        assert len(values) == 3
        assert 1 in values
        assert 2 in values
        assert 3 in values

    def test_items(self):
        """Test items iteration."""
        hm = HashMap()
        hm.put('a', 1)
        hm.put('b', 2)
        
        items = list(hm.items())
        assert len(items) == 2
        assert ('a', 1) in items
        assert ('b', 2) in items

    def test_len(self):
        """Test length operation."""
        hm = HashMap()
        assert len(hm) == 0
        
        hm.put('key1', 'val1')
        assert len(hm) == 1
        
        hm.put('key2', 'val2')
        assert len(hm) == 2
        
        hm.delete('key1')
        assert len(hm) == 1

    def test_resize(self):
        """Test that HashMap resizes properly under load."""
        hm = HashMap(initial_capacity=4)
        
        # Add enough items to trigger resize (> 0.75 load factor)
        for i in range(10):
            hm.put(f'key{i}', f'value{i}')
        
        # Verify all items are still accessible after resize
        assert len(hm) == 10
        for i in range(10):
            assert hm.get(f'key{i}') == f'value{i}'

    def test_various_key_types(self):
        """Test HashMap with various hashable key types."""
        hm = HashMap()
        
        # String keys
        hm.put('string_key', 1)
        assert hm.get('string_key') == 1
        
        # Integer keys
        hm.put(42, 'int_value')
        assert hm.get(42) == 'int_value'
        
        # Tuple keys
        hm.put((1, 2), 'tuple_value')
        assert hm.get((1, 2)) == 'tuple_value'
        
        # Float keys
        hm.put(3.14, 'float_value')
        assert hm.get(3.14) == 'float_value'

    def test_collision_handling(self):
        """Test that HashMap handles hash collisions correctly."""
        hm = HashMap(initial_capacity=2)  # Small capacity to force collisions
        
        # Add multiple items that will likely collide
        hm.put('a', 1)
        hm.put('b', 2)
        hm.put('c', 3)
        hm.put('d', 4)
        
        # All should be retrievable
        assert hm.get('a') == 1
        assert hm.get('b') == 2
        assert hm.get('c') == 3
        assert hm.get('d') == 4


def run_tests():
    """Run all tests and report results."""
    test_classes = [TestBinarySearch, TestHashMap]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"PASS {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"FAIL {method_name}: {e}")
                failed_tests.append((test_class.__name__, method_name, str(e)))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Test Summary")
    print('='*60)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
        return False
    else:
        print(f"\nAll tests passed!")
        return True


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
