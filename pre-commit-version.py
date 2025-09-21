#!/usr/bin/env python3
"""
Pre-commit hook to automatically increment version.
This script can be used as a git pre-commit hook or run manually.
"""

import os
import sys
import subprocess
from pathlib import Path

def increment_version():
    """Increment version based on git status."""
    try:
        # Check if there are changes to version-sensitive files
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-only'],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode == 0:
            changed_files = result.stdout.strip().split('\n')
            
            # Check if any important files changed
            important_files = ['app.py', 'models.py', 'analytics.py', 'charts.py', 'requirements.txt']
            has_important_changes = any(f in changed_files for f in important_files)
            
            if has_important_changes:
                # Increment patch version
                from version_increment import VersionIncrementer
                incrementer = VersionIncrementer()
                new_version = incrementer.increment_based_on_strategy('patch')
                incrementer.save_version(new_version)
                
                print(f"✓ Version incremented to {new_version}")
                return True
        
        return False
        
    except Exception as e:
        print(f"✗ Error incrementing version: {e}")
        return False

def main():
    """Main function."""
    if increment_version():
        print("✓ Pre-commit version increment completed")
        return 0
    else:
        print("ℹ No version increment needed")
        return 0

if __name__ == '__main__':
    sys.exit(main())
