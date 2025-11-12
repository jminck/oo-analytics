# Quick Start: Automatic Consistency Enforcement

## 5-Minute Setup

### 1. Install Tools (One-time)

```bash
pip install flake8 pre-commit
```

### 2. Set Up Pre-commit Hooks

```bash
pre-commit install
```

### 3. Test It Works

```bash
# Run consistency check
python scripts/check_consistency.py

# Run all pre-commit checks
pre-commit run --all-files
```

That's it! Now every commit will automatically check for consistency violations.

## What Gets Checked?

### Automatic Checks (on every commit):
- ✅ API endpoints have try/except error handling
- ✅ API responses include 'success' field
- ✅ Tables have proper structure (table-responsive, table-dark)
- ✅ Sortable headers have required attributes
- ✅ Code follows style guide patterns

### Manual Checks (run when needed):
```bash
# Check consistency
python scripts/check_consistency.py

# Check Python style
flake8 .

# Check all files before committing
pre-commit run --all-files
```

## Common Workflows

### Before Committing
```bash
# Option 1: Let pre-commit handle it automatically
git add .
git commit -m "Your message"
# Pre-commit runs automatically

# Option 2: Run checks manually first
python scripts/check_consistency.py
git add .
git commit -m "Your message"
```

### Skip Checks (Emergency Only)
```bash
git commit --no-verify -m "Emergency fix"
```

### Fix Issues Found
1. Read the error message
2. Check `DEVELOPMENT_STYLE_GUIDE.md` for the pattern
3. Use `CODE_TEMPLATES.md` for correct examples
4. Fix and commit again

## Integration with Existing Hooks

If you have version increment hooks, they'll run together. The pre-commit framework manages multiple hooks automatically.

## Next Steps

- Read `ENFORCEMENT_SETUP.md` for detailed configuration
- Customize `.flake8` and `.eslintrc.json` for your preferences
- Add CI/CD integration (see `ENFORCEMENT_SETUP.md`)

## Troubleshooting

**Hooks not running?**
```bash
pre-commit uninstall
pre-commit install
```

**Too many violations?**
- Start with warnings only (remove `--strict`)
- Fix incrementally as you touch files
- Use `# noqa` comments for exceptions

**Need help?**
- Check `ENFORCEMENT_SETUP.md` for detailed docs
- Review `DEVELOPMENT_STYLE_GUIDE.md` for patterns
- See `CONSISTENCY_CHECKLIST.md` for manual checks

---

**Remember**: The goal is consistency, not blocking development. Use these tools to guide, not to prevent progress.

