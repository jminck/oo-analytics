# Git Hooks Setup for Automatic Version Incrementing

This guide explains how to set up Git hooks to automatically increment the application version when committing changes.

## Overview

The Git hooks will:
- **Automatically increment** the patch version (e.g., 1.0.0 → 1.0.1) on each commit
- **Only increment** when there are relevant changes (excludes version files themselves)
- **Add version files** to the commit automatically
- **Work on Windows, macOS, and Linux**

## Quick Setup

### Option 1: Automated Setup (Recommended)

Run the setup script:

```powershell
# Windows PowerShell
.\setup-git-hooks.ps1

# Or in Git Bash
bash setup-git-hooks.ps1
```

### Option 2: Manual Setup

1. **Copy the appropriate hook file:**
   ```bash
   # For PowerShell (Windows, recommended)
   copy .git\hooks\pre-commit.ps1 .git\hooks\pre-commit
   
   # For Batch (Windows, fallback)
   copy .git\hooks\pre-commit.bat .git\hooks\pre-commit
   
   # For Bash (Linux/macOS)
   cp .git/hooks/pre-commit .git/hooks/pre-commit
   ```

2. **Make the hook executable:**
   ```bash
   git update-index --chmod=+x .git/hooks/pre-commit
   ```

## How It Works

### Pre-commit Hook Process

1. **Check for staged changes** (excluding version files)
2. **If relevant changes found:**
   - Run `pre-commit-version.py` to increment patch version
   - Add updated `version.txt` and `version.json` to commit
   - Display success message
3. **If no relevant changes:**
   - Skip version increment
   - Display info message

### Version Increment Logic

- **Patch version** increments: `1.0.0` → `1.0.1` → `1.0.2`
- **Only increments** when there are changes to application code
- **Skips increment** for commits that only modify version files
- **Preserves** major and minor versions (manual control)

## File Structure

```
.git/hooks/
├── pre-commit          # Main hook (auto-selected)
├── pre-commit.ps1      # PowerShell version (Windows)
├── pre-commit.bat      # Batch version (Windows fallback)
└── pre-commit          # Bash version (Linux/macOS)

Root directory:
├── pre-commit-version.py    # Version increment script
├── setup-git-hooks.ps1      # Setup script
├── version.txt              # Current version
└── version.json             # Detailed version info
```

## Testing the Setup

### Test the Hook

1. **Make a change:**
   ```bash
   echo "# Test change" >> README.md
   ```

2. **Stage and commit:**
   ```bash
   git add README.md
   git commit -m "Test commit with version increment"
   ```

3. **Check the result:**
   ```bash
   cat version.txt  # Should show incremented version
   git log --oneline -1  # Should show both files in commit
   ```

### Expected Output

```
🔧 Detected staged changes. Incrementing version...
Updated version.txt to 1.0.1
Updated version and build timestamp in version.json to 1.0.1
✅ Version incremented and added to commit
[main abc1234] Test commit with version increment
 2 files changed, 2 insertions(+)
 create mode 100644 README.md
```

## Configuration Options

### Environment Variables

You can customize the behavior with environment variables:

```bash
# Disable automatic version incrementing
export DISABLE_AUTO_VERSION=true

# Set custom version increment strategy
export VERSION_INCREMENT_STRATEGY=patch  # patch, minor, major
```

### Manual Version Control

For major/minor version changes, use the manual scripts:

```bash
# Increment major version (1.0.0 → 2.0.0)
python version_increment.py --strategy major --save

# Increment minor version (1.0.0 → 1.1.0)
python version_increment.py --strategy minor --save

# Increment patch version (1.0.0 → 1.0.1)
python version_increment.py --strategy patch --save
```

## Troubleshooting

### Hook Not Running

**Problem:** Version not incrementing on commits

**Solutions:**
1. **Check hook exists:**
   ```bash
   ls -la .git/hooks/pre-commit
   ```

2. **Check hook permissions:**
   ```bash
   git update-index --chmod=+x .git/hooks/pre-commit
   ```

3. **Test hook manually:**
   ```bash
   .git/hooks/pre-commit
   ```

### Permission Errors

**Problem:** "Permission denied" errors

**Solutions:**
1. **Windows PowerShell:**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **Git Bash:**
   ```bash
   git update-index --chmod=+x .git/hooks/pre-commit
   ```

### Python Script Errors

**Problem:** `pre-commit-version.py` not found or failing

**Solutions:**
1. **Check file exists:**
   ```bash
   ls -la pre-commit-version.py
   ```

2. **Test script manually:**
   ```bash
   python pre-commit-version.py
   ```

3. **Check Python path:**
   ```bash
   which python
   which python3
   ```

## Disabling the Hook

### Temporary Disable

Skip the hook for a single commit:

```bash
git commit --no-verify -m "Skip version increment for this commit"
```

### Permanent Disable

Remove or rename the hook file:

```bash
# Remove the hook
rm .git/hooks/pre-commit

# Or rename it
mv .git/hooks/pre-commit .git/hooks/pre-commit.disabled
```

## Advanced Configuration

### Custom Hook Logic

You can modify the hook behavior by editing `.git/hooks/pre-commit`:

```bash
# Only increment on specific file types
if echo "$RELEVANT_FILES" | grep -q "\.(py|js|html)$"; then
    # Increment version
fi

# Skip increment for certain branches
if [ "$(git branch --show-current)" = "main" ]; then
    # Skip increment on main branch
    exit 0
fi
```

### Multiple Hooks

You can have multiple pre-commit hooks by creating additional files:

```bash
.git/hooks/
├── pre-commit          # Version increment
├── pre-commit-lint     # Code linting
└── pre-commit-test     # Run tests
```

## Integration with CI/CD

### GitHub Actions

The version files are automatically updated and committed, so your CI/CD pipeline will see the new version:

```yaml
- name: Checkout code
  uses: actions/checkout@v3
  with:
    fetch-depth: 0  # Fetch full history for version tracking

- name: Get version
  run: |
    VERSION=$(cat version.txt)
    echo "VERSION=$VERSION" >> $GITHUB_ENV
```

### Azure DevOps

```yaml
- script: |
    VERSION=$(cat version.txt)
    echo "##vso[task.setvariable variable=version]$VERSION"
  displayName: 'Get version from file'
```

## Best Practices

1. **Commit Messages:** Use descriptive commit messages since version increments are automatic
2. **Branch Strategy:** Consider disabling auto-increment on main/master branches
3. **Release Process:** Use manual version scripts for major/minor releases
4. **Testing:** Test the hook setup in a development branch first
5. **Backup:** Keep backups of version files before major changes

## Support

If you encounter issues:

1. **Check the logs:** Look for error messages in the commit output
2. **Test manually:** Run the hook and version script manually
3. **Verify setup:** Ensure all required files are present and executable
4. **Check permissions:** Verify file and directory permissions

The Git hooks provide a seamless way to maintain version consistency across your development workflow while giving you full control over when and how versions are incremented.
