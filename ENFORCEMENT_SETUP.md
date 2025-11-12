# Automatic Consistency Enforcement Setup

This guide explains how to set up automatic enforcement of the development style guide patterns.

## Overview

The enforcement system includes:
1. **Pre-commit hooks** - Check code before commits
2. **Linting tools** - Static analysis for Python and JavaScript
3. **Custom consistency checker** - Validates patterns from style guide
4. **CI/CD integration** - Automated checks in deployment pipeline

## Quick Setup

### Step 1: Install Linting Tools

```bash
# Install Python linting tools
pip install flake8 black

# Install pre-commit framework (optional but recommended)
pip install pre-commit

# Install JavaScript linting (if using Node.js)
npm install -g eslint
# Or use npx without installing globally
```

### Step 2: Set Up Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Test the hooks
pre-commit run --all-files
```

### Step 3: Run Consistency Checker

```bash
# Run the custom consistency checker
python scripts/check_consistency.py

# Run with strict mode (warnings as errors)
python scripts/check_consistency.py --strict

# Get JSON output
python scripts/check_consistency.py --json
```

## Configuration Files

### `.flake8` - Python Linting
- Configures flake8 for Python code style
- Excludes unnecessary directories
- Sets line length to 120 characters

### `.eslintrc.json` - JavaScript Linting
- Configures ESLint for JavaScript code style
- Sets indentation to 4 spaces
- Allows console.log for debugging

### `.editorconfig` - Editor Configuration
- Ensures consistent formatting across editors
- Sets indentation, line endings, charset
- Works with VS Code, IntelliJ, and other editors

### `.pre-commit-config.yaml` - Pre-commit Hooks
- Defines hooks to run before commits
- Includes file checks, linting, and consistency checks
- Can be customized per project needs

## Integration with Existing Git Hooks

If you already have git hooks (like version incrementing), you can combine them:

### Option 1: Use Pre-commit Framework (Recommended)

The pre-commit framework manages multiple hooks automatically. Your existing version increment hook can be added:

```yaml
# Add to .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: version-increment
        name: Version Increment
        entry: python pre-commit-version.py
        language: system
        pass_filenames: false
        always_run: true
```

### Option 2: Manual Git Hook Integration

Edit `.git/hooks/pre-commit` to include consistency checks:

```bash
#!/bin/bash
# Run version increment
python pre-commit-version.py

# Run consistency check
python scripts/check_consistency.py --strict

# Exit with error if consistency check fails
if [ $? -ne 0 ]; then
    echo "❌ Consistency check failed. Please fix violations before committing."
    exit 1
fi
```

## Running Checks Manually

### Before Committing

```bash
# Run all pre-commit checks
pre-commit run --all-files

# Run only consistency checker
python scripts/check_consistency.py

# Run only Python linting
flake8 .

# Run only JavaScript linting (if ESLint is installed)
eslint static/js/*.js
```

### In Your IDE

#### VS Code

Install extensions:
- **Python** (Microsoft) - Includes linting
- **ESLint** (Microsoft) - JavaScript linting
- **EditorConfig** (EditorConfig) - Formatting

Add to `.vscode/settings.json`:
```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.flake8Path": "flake8",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  }
}
```

#### PyCharm/IntelliJ

1. Go to Settings → Editor → Code Style
2. Import `.editorconfig`
3. Enable inspections for Python and JavaScript
4. Configure flake8 as external tool

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/consistency-check.yml`:

```yaml
name: Consistency Check

on:
  pull_request:
    branches: [ main, develop ]
  push:
    branches: [ main, develop ]

jobs:
  consistency:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install flake8
      
      - name: Run consistency check
        run: |
          python scripts/check_consistency.py --strict
      
      - name: Run flake8
        run: |
          flake8 . --config=.flake8
```

### Azure DevOps

Add to your pipeline YAML:

```yaml
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.9'

- script: |
    pip install flake8
    python scripts/check_consistency.py --strict
  displayName: 'Consistency Check'

- script: |
    flake8 . --config=.flake8
  displayName: 'Python Linting'
```

## Customizing Checks

### Adding New Checks

Edit `scripts/check_consistency.py` to add new pattern checks:

```python
def check_custom_pattern(self, file_path: Path, content: str):
    """Check for custom pattern."""
    if 'custom_pattern' in content:
        # Check if pattern is followed correctly
        if not self._pattern_followed(content):
            self.violations.append({
                'file': str(file_path),
                'line': line_num,
                'type': 'custom_pattern',
                'message': 'Custom pattern violation'
            })
```

### Adjusting Linting Rules

**Python (flake8):**
- Edit `.flake8` to change rules
- Add per-file ignores for specific cases

**JavaScript (ESLint):**
- Edit `.eslintrc.json` to change rules
- Add exceptions for specific files

### Disabling Checks Temporarily

**Skip pre-commit hooks:**
```bash
git commit --no-verify -m "Emergency fix"
```

**Skip specific check:**
```python
# In code, add comment to disable flake8 for next line
result = long_line_that_violates_length  # noqa: E501
```

## Best Practices

1. **Run checks before committing** - Catch issues early
2. **Fix violations immediately** - Don't accumulate technical debt
3. **Use strict mode in CI/CD** - Ensure all code meets standards
4. **Review warnings** - They often indicate potential issues
5. **Update checks as patterns evolve** - Keep enforcement current

## Troubleshooting

### Pre-commit Hooks Not Running

```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install

# Check hook exists
ls -la .git/hooks/pre-commit
```

### Consistency Checker Errors

```bash
# Run with verbose output
python scripts/check_consistency.py --path . --strict

# Check Python version (requires 3.7+)
python --version
```

### Linting False Positives

1. Add file to ignore list in `.flake8` or `.eslintrc.json`
2. Use inline comments to disable specific rules
3. Update configuration to be less strict for specific cases

## Gradual Adoption

If you have existing code that doesn't meet standards:

1. **Start with warnings only** - Don't fail builds initially
2. **Fix incrementally** - Address violations as you touch files
3. **Use strict mode for new code** - Enforce standards on new features
4. **Create exceptions** - Document why certain patterns are allowed

## Next Steps

1. Install the tools: `pip install flake8 black pre-commit`
2. Set up pre-commit: `pre-commit install`
3. Run initial check: `python scripts/check_consistency.py`
4. Fix any violations found
5. Integrate into CI/CD pipeline
6. Update team on new process

---

**Remember**: The goal is consistency, not perfection. Use these tools to guide development, not to block progress.

