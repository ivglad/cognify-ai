#!/usr/bin/env python3
"""
Test runner script for RAGFlow backend tests.
"""
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run RAGFlow backend tests")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "e2e", "all", "fast", "slow"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--parallel", "-n",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--file",
        help="Run specific test file"
    )
    parser.add_argument(
        "--function",
        help="Run specific test function"
    )
    parser.add_argument(
        "--failed",
        action="store_true",
        help="Run only failed tests from last run"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile test performance"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.extend(["-v", "-s"])
    
    # Add parallel execution
    if args.parallel > 1:
        cmd.extend(["-n", str(args.parallel)])
    
    # Add coverage
    if args.coverage:
        cmd.extend([
            "--cov=app",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])
    
    # Add profiling
    if args.profile:
        cmd.extend(["--profile", "--profile-svg"])
    
    # Add failed tests only
    if args.failed:
        cmd.append("--lf")
    
    # Test type selection
    if args.type == "unit":
        cmd.extend(["-m", "unit"])
    elif args.type == "integration":
        cmd.extend(["-m", "integration"])
    elif args.type == "e2e":
        cmd.extend(["-m", "e2e"])
    elif args.type == "fast":
        cmd.extend(["-m", "not slow"])
    elif args.type == "slow":
        cmd.extend(["-m", "slow"])
    
    # Specific file or function
    if args.file:
        if args.function:
            cmd.append(f"{args.file}::{args.function}")
        else:
            cmd.append(args.file)
    elif args.function:
        cmd.extend(["-k", args.function])
    
    # Run the tests
    success = run_command(cmd, f"Running {args.type} tests")
    
    if success:
        print(f"\n🎉 All tests passed!")
        
        if args.coverage:
            print(f"\n📊 Coverage report generated:")
            print(f"  - HTML: htmlcov/index.html")
            print(f"  - XML: coverage.xml")
    else:
        print(f"\n💥 Some tests failed!")
        sys.exit(1)


def run_test_suites():
    """Run different test suites in sequence."""
    print("🚀 Running RAGFlow Test Suites")
    print("="*60)
    
    suites = [
        (["python", "-m", "pytest", "-m", "unit", "--tb=short"], "Unit Tests"),
        (["python", "-m", "pytest", "-m", "integration", "--tb=short"], "Integration Tests"),
        (["python", "-m", "pytest", "-m", "e2e", "--tb=short"], "End-to-End Tests"),
    ]
    
    results = []
    for cmd, description in suites:
        success = run_command(cmd, description)
        results.append((description, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUITE SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{description:<30} {status}")
        if not success:
            all_passed = False
    
    print(f"{'='*60}")
    if all_passed:
        print("🎉 ALL TEST SUITES PASSED!")
        return 0
    else:
        print("💥 SOME TEST SUITES FAILED!")
        return 1


def run_quick_tests():
    """Run quick tests for development."""
    cmd = [
        "python", "-m", "pytest",
        "-m", "not slow",
        "--tb=short",
        "-x",  # Stop on first failure
        "--ff"  # Run failures first
    ]
    
    return run_command(cmd, "Quick Development Tests")


def run_ci_tests():
    """Run tests suitable for CI/CD."""
    cmd = [
        "python", "-m", "pytest",
        "--tb=short",
        "--cov=app",
        "--cov-report=xml",
        "--cov-report=term",
        "--cov-fail-under=80",
        "--junitxml=test-results.xml"
    ]
    
    return run_command(cmd, "CI/CD Tests")


if __name__ == "__main__":
    # Check if specific commands are requested
    if len(sys.argv) > 1:
        if sys.argv[1] == "suites":
            sys.exit(run_test_suites())
        elif sys.argv[1] == "quick":
            success = run_quick_tests()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "ci":
            success = run_ci_tests()
            sys.exit(0 if success else 1)
    
    # Otherwise run main CLI
    main()