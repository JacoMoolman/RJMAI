"""
Test runner script for RJMAI project
Run this script to execute all tests in the tests directory
"""
import unittest
import os
import sys

if __name__ == '__main__':
    # Add the parent directory to the path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    print("Starting test discovery...")
    
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    print(f"Found {test_suite.countTestCases()} test cases")
    
    # Run the tests with verbose output
    print("Running tests...")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print test summary
    print(f"\nTest Summary:")
    print(f"  Ran {result.testsRun} tests")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful())
