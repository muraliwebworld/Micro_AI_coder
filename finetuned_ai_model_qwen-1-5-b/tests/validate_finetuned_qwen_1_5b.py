#!/usr/bin/env python3
"""
Validation test suite for fine-tuned Qwen2.5-Coder-1.5B React component generator.
Tests 20 diverse React patterns with 90%+ target pass rate.
"""

import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add inference module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "phase3_inference"))
from v3_inference_finetuned_1_5b import (
    generate_react_component,
    validate_react_code,
    check_ollama_connection,
)


class ValidationSuite:
    """Test suite for React component generation."""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def test_case(self, name: str, prompt: str) -> Tuple[bool, Dict]:
        """Run a single test case."""
        print(f"\n📝 Test: {name}")
        print(f"   Prompt: {prompt}")

        result = generate_react_component(prompt, temperature=0.3)

        if not result:
            print("   ❌ FAIL - Generation failed")
            self.failed += 1
            return False, {}

        # Check quality criteria
        checks = {
            "has_code": len(result["code"]) > 50,
            "valid_jsx": result["validation"]["valid_jsx"],
            "balanced_syntax": result["validation"]["balanced_braces"]
            and result["validation"]["balanced_brackets"],
            "has_imports": result["validation"]["has_import"],
            "has_exports": result["validation"]["has_export"],
            "has_return": result["validation"]["has_return"],
        }

        passed = sum(checks.values()) >= 5  # 5/6 required

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}")
        for check, value in checks.items():
            check_icon = "✓" if value else "✗"
            print(f"      {check_icon} {check}")

        if passed:
            self.passed += 1
        else:
            self.failed += 1

        return passed, checks

    def run_all_tests(self) -> Dict:
        """Run complete test suite."""
        print("\n" + "=" * 70)
        print("🧪 Qwen2.5-Coder-1.5B React Component Validation Test Suite")
        print("=" * 70)

        test_cases = [
            # Basic Components
            ("Simple Button", "Write a simple React button component"),
            ("Styled Button", "Create a styled React button with Tailwind CSS"),
            ("Click Handler", "Write a React button that logs a message on click"),
            # Forms
            ("Input Form", "Create a React form with text input and submit button"),
            ("Login Form", "Write a React login form with email and password fields"),
            ("Multi-input", "Generate a React form with multiple input types"),
            # State Management
            ("Counter", "Write a React counter component with increment/decrement"),
            ("Todo List", "Create a React component for a simple todo list"),
            ("Toggle", "Write a React toggle switch component"),
            # Lists and Data
            ("User Card", "Generate a React card component displaying user info"),
            ("User List", "Create a React component that displays a list of users"),
            ("Item Grid", "Write a React grid component for displaying items"),
            # Hooks and Advanced
            ("UseEffect", "Create a React component using useEffect hook"),
            ("UseContext", "Write a React component using Context API"),
            ("Custom Hook", "Generate a React custom hook for API calls"),
            # Styling
            ("CSS Modules", "Write a React component using CSS modules"),
            ("Styled Components", "Create a React component using styled-components"),
            ("Inline Styles", "Write a React component with inline CSS styling"),
            # Real-world Components
            ("Modal", "Create a React modal/dialog component"),
            ("Navbar", "Generate a React navigation bar component"),
        ]

        for name, prompt in test_cases:
            passed, checks = self.test_case(name, prompt)
            self.results.append(
                {
                    "name": name,
                    "prompt": prompt,
                    "passed": passed,
                    "checks": checks,
                }
            )

        return self.generate_report()

    def generate_report(self) -> Dict:
        """Generate test report."""
        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total,
            "passed": self.passed,
            "failed": self.failed,
            "success_rate": f"{success_rate:.1f}%",
            "target_rate": "≥90%",
            "target_met": success_rate >= 90,
            "results": self.results,
        }

        # Print summary
        print("\n" + "=" * 70)
        print("📊 Test Summary")
        print("=" * 70)
        print(f"Total Tests:   {total}")
        print(f"Passed:        {self.passed} ✅")
        print(f"Failed:        {self.failed} ❌")
        print(f"Success Rate:  {success_rate:.1f}%")
        print(f"Target:        ≥90%")

        if report["target_met"]:
            print(f"\n🎉 TARGET MET! Model is production-ready!")
        else:
            print(f"\n⚠️  Below target. Consider retraining or adjusting model.")

        print("=" * 70)

        return report


def main():
    """Main function."""
    # Check Ollama
    print("🔍 Checking Ollama connection...")
    if not check_ollama_connection():
        print("❌ Ollama is not running!")
        print("\nStart Ollama with: ollama serve")
        return 1

    print("✅ Ollama connected")

    # Run tests
    suite = ValidationSuite()
    report = suite.run_all_tests()

    # Save report
    output_file = Path("test_results_qwen_1_5b_finetuned.json")
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Report saved to: {output_file}")

    # Return exit code
    return 0 if report["target_met"] else 1


if __name__ == "__main__":
    sys.exit(main())
