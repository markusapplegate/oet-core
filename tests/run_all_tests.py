#!/usr/bin/env python3
"""
Run all tests for the oet-core project.

This script discovers and runs all test files in the tests/ directory.
"""
import sys
from pathlib import Path

# Add parent directory to path to import tests
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_all_tests():
    """Discover and run all test modules."""
    tests_dir = Path(__file__).parent
    test_files = sorted(tests_dir.glob('test_*.py'))
    
    if not test_files:
        print("No test files found!")
        return False
    
    print("=" * 70)
    print("Running Test Suite for oet-core")
    print("=" * 70)
    
    all_passed = True
    results = []
    
    for test_file in test_files:
        module_name = test_file.stem
        print(f"\n{'#' * 70}")
        print(f"# {module_name.upper()}")
        print(f"{'#' * 70}")
        
        try:
            # Import the test module
            module = __import__(module_name)
            
            # Run the tests
            if hasattr(module, 'run_tests'):
                success = module.run_tests()
                results.append((module_name, success))
                if not success:
                    all_passed = False
            else:
                print(f"Warning: {module_name} does not have a run_tests() function")
                results.append((module_name, None))
        
        except Exception as e:
            print(f"Error running {module_name}: {e}")
            results.append((module_name, False))
            all_passed = False
    
    # Final summary
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print('=' * 70)
    
    for module_name, success in results:
        if success is True:
            status = "PASSED"
        elif success is False:
            status = "FAILED"
        else:
            status = "SKIPPED"
        print(f"{status}: {module_name}")
    
    print('=' * 70)
    
    if all_passed:
        print("\nALL TEST SUITES PASSED!\n")
        return True
    else:
        print("\nSOME TESTS FAILED\n")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
