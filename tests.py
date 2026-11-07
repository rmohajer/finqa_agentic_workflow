#!/usr/bin/env python3
"""
Main test runner for FinQA project.
Runs all unit tests from the tests/ directory.
"""
import sys
import subprocess
import os
from pathlib import Path


def run_tests(test_file=None, verbose=True, coverage=False, markers=None):
    """
    Run pytest tests.
    
    Args:
        test_file: Specific test file to run (e.g., "tests/parsing_helper_tests.py")
        verbose: Whether to run in verbose mode
        coverage: Whether to run with coverage
        markers: Pytest markers to filter (e.g., "unit", "integration")
    
    Returns:
        Exit code from pytest
    """
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=.", "--cov-report=term-missing", "--cov-report=html"])
    
    if markers:
        cmd.extend(["-m", markers])
    
    if test_file:
        cmd.append(test_file)
    else:
        cmd.append("tests/")
    
    # Add common options
    cmd.extend([
        "--tb=short",
        "--strict-markers",
        "--strict-config",
    ])
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run from project root
    project_root = Path(__file__).parent
    result = subprocess.run(cmd, cwd=project_root)
    
    return result.returncode


def run_parsing_helper_tests():
    """Run parsing helper tests specifically."""
    return run_tests("tests/parsing_helper_tests.py", verbose=True)


def run_all_tests():
    """Run all tests in tests/ directory."""
    return run_tests()


def run_unit_tests():
    """Run only unit tests."""
    return run_tests(markers="unit")


def run_integration_tests():
    """Run only integration tests."""
    return run_tests(markers="integration")


def run_with_coverage(test_file=None):
    """Run tests with coverage report."""
    return run_tests(test_file=test_file, coverage=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FinQA Test Runner")
    parser.add_argument(
        "target",
        nargs="?",
        choices=["all", "parsing", "unit", "integration", "coverage"],
        default="parsing",
        help="What to test (default: parsing)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Verbose output"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet output (overrides -v)"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage report"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Specific test file to run"
    )
    
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    
    if args.file:
        exit_code = run_tests(test_file=args.file, verbose=verbose, coverage=args.coverage)
    elif args.target == "all":
        exit_code = run_all_tests()
    elif args.target == "parsing":
        exit_code = run_parsing_helper_tests()
    elif args.target == "unit":
        exit_code = run_unit_tests()
    elif args.target == "integration":
        exit_code = run_integration_tests()
    elif args.target == "coverage":
        exit_code = run_with_coverage()
    else:
        exit_code = run_parsing_helper_tests()
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    sys.exit(exit_code)