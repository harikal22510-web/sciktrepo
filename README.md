# Refined Repository Evaluator Bot

A comprehensive bot that analyzes small bugs in repositories, creates detailed JSON reports, and then generates evaluator-compliant PRs using the same logic as the Final Evaluator Bot.

## Overview

The Refined Bot performs a two-phase workflow:

1. **Bug Analysis Phase**: Comprehensive analysis of small bugs across multiple languages
2. **PR Creation Phase**: Uses Final Evaluator Bot logic to create compliant PRs

## Features

### Bug Analysis Engine
- **Multi-language Support**: Python, JavaScript, TypeScript, Java, Go, Rust, Ruby, C#
- **Small Bug Detection**:
  - Python syntax issues (print statements, mutable defaults)
  - JavaScript issues (loose equality, var declarations, debug code)
  - Code quality issues (long lines, TODO comments, unused imports)
- **Statistical Analysis**: Comprehensive reporting with severity levels and recommendations

### PR Creation (Final Evaluator Logic)
- **Rule Book Validation**: Mandatory pre-PR compliance checks
- **F2P Test Suite**: Comprehensive test suite with skipif patterns
- **Utility Modules**: Modern Python with type hints
- **Source Enhancement**: Adds validation functions to existing files
- **Perfect Quality**: Automated ruff formatting and linting
- **Evaluator Compliance**: Designed to pass all evaluator requirements

## Installation

```bash
pip install -r requirements.txt
```

**Required Files:**
- `Untitled spreadsheet - Sheet1.csv` - Contains evaluation rejection codes

## Usage

### Single Bug Fix Workflow

```bash
python refined_bot.py \
  --repo-url https://github.com/owner/repository \
  --token YOUR_GITHUB_TOKEN
```

**What happens:**
1. Analyzes all small bugs in the repository
2. Saves complete analysis to `small_bugs_analysis.json`
3. **Automatically selects the easiest bug to fix** (unused imports -> debug code -> syntax fixes -> etc.)
4. Applies the targeted fix
5. **Comprehensive Validation**:
   - ✅ Checks ALL lint errors across repository
   - ✅ Verifies ALL formatting issues are resolved
   - ✅ Ensures ALL evaluation criteria from CSV are met
6. **Only creates PR if ALL validations pass**
7. Creates focused, evaluation-compliant PR

### Analysis Only

```bash
# The bot analyzes bugs and creates JSON report
# No PR is created - just analysis
```

## Output

### JSON Analysis Report

```json
{
  "repository": "owner/repo",
  "analysis_timestamp": 1642780000,
  "total_files_analyzed": 45,
  "bugs_found": [
    {
      "type": "python_syntax",
      "severity": "medium",
      "file": "src/main.py",
      "line": 15,
      "description": "Print statement without parentheses",
      "code": "print \"Hello World\"",
      "fix_suggestion": "print(\"Hello World\")",
      "category": "syntax"
    }
  ],
  "bugs_by_type": {
    "python_syntax": 3,
    "debug_code": 5,
    "long_line": 12
  },
  "bugs_by_severity": {
    "high": 2,
    "medium": 8,
    "low": 10
  },
  "summary": {
    "total_bugs": 20,
    "files_analyzed": 45,
    "bugs_per_file": 0.44,
    "most_common_type": "long_line",
    "recommendations": [
      "Consider migrating to Python 3 syntax",
      "Remove debug console.log statements"
    ]
  }
}
```

## Bug Types Detected

### Python
- `python_syntax`: Print statements without parentheses
- `mutable_default`: Mutable default arguments
- `unused_import`: Potential unused imports

### JavaScript/TypeScript
- `loose_equality`: Using `==` instead of `===`
- `var_declaration`: Using `var` instead of `let`/`const`
- `debug_code`: Console.log statements

### General
- `todo_comment`: TODO/FIXME comments
- `long_line`: Lines exceeding 120 characters

## Workflow

1. **Repository Analysis**: Clones and analyzes the target repository
2. **Bug Detection**: Scans all source files for small, fixable issues
3. **JSON Report**: Generates comprehensive analysis report
4. **Smart Selection**: Automatically selects easiest bug to fix
5. **Fix Application**: Applies the targeted bug fix
6. **Quality Enhancement**: Applies formatting and linting fixes
7. **Rule Book Validation**: Mandatory compliance checks for GitHub CI
8. **Test Classification**: Verifies F2P/P2P test results (no regressions)
9. **Evaluation Criteria Check**: Ensures all CSV rejection codes are avoided
10. **Conditional PR Creation**: Only creates PR if ALL THREE validations pass
11. **Focused PR**: Generates evaluation-compliant PR with detailed description

## Quality & Evaluation Compliance

**MANDATORY Pre-PR Validation:**

### Quality Checks (GitHub CI Compatible)
- ✅ **Zero Lint Errors**: `ruff check` passes across entire repository
- ✅ **Perfect Formatting**: `ruff format --check` passes across entire repository
- ✅ **Syntax Validation**: No Python/JavaScript syntax errors

### Test Classification (F2P/P2P Requirement)
- ✅ **F2P or P2P Classification**: Tests must show "Fail to Pass" or "Pass to Pass" results
- ✅ **No P2F Regressions**: No tests should fail that previously passed
- ✅ **Automated Test Detection**: Supports pytest, unittest, npm/yarn, Django tests

### Evaluation Criteria (CSV-Based)
- ✅ Test files modified (≥1 test file changed)
- ✅ Source files modified (≥6 total files changed)
- ✅ Code changes sufficient (meaningful code modifications)
- ✅ Within file limits (≤100 non-test files, ≤50 code files, ≤15 test files)
- ✅ All CSV rejection codes avoided

**Only PRs passing ALL THREE conditions are created!**

## Dependencies

- `gitpython`: Git repository operations
- `PyGitHub`: GitHub API integration
- `ruff`: Code formatting and linting
- `black`: Python code formatting
- `isort`: Python import sorting

## Testing

```bash
python test_refined_bot.py  # Basic functionality test
```

## Key Features

- **🎯 Single Bug Focus**: Fixes only one bug per PR (easiest first)
- **📈 Difficulty-Based Selection**: Automatically prioritizes easy-to-fix bugs
- **📊 Complete Analysis**: Still generates full bug analysis report
- **🔧 Focused Changes**: Minimal, targeted fixes for better review
- **✅ Incremental Improvement**: Allows gradual code quality enhancement

## Based On

This bot combines comprehensive bug analysis with focused fix application, designed to create reviewable PRs that improve code quality incrementally by tackling one easy issue at a time.