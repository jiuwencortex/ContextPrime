"""Test runner for skill discovery tests."""

import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parent

def run_tests():
    """Run the skill discovery tests."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(TESTS_DIR / "test_skill_discovery.py"), "-v"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
