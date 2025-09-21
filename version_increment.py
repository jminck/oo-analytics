#!/usr/bin/env python3
"""
Automatic version incrementing system for Portfolio Strategy Analytics.
Supports multiple increment strategies: git tags, semantic versioning, and build numbers.
"""

import os
import sys
import json
import subprocess
import datetime
from pathlib import Path
from typing import Optional, Tuple

class VersionIncrementer:
    """Handles automatic version incrementing based on different strategies."""
    
    def __init__(self):
        self.version_file = 'version.json'
        self.version_txt = 'version.txt'
    
    def get_current_version(self) -> str:
        """Get current version from various sources."""
        # Check environment variable first
        env_version = os.getenv('APP_VERSION')
        if env_version:
            return env_version
        
        # Check version.txt
        if os.path.exists(self.version_txt):
            try:
                with open(self.version_txt, 'r') as f:
                    return f.read().strip()
            except IOError:
                pass
        
        # Check version.json
        if os.path.exists(self.version_file):
            try:
                with open(self.version_file, 'r') as f:
                    data = json.load(f)
                    return data.get('version', '1.0.0')
            except (json.JSONDecodeError, IOError):
                pass
        
        return '1.0.0'
    
    def parse_semantic_version(self, version: str) -> Tuple[int, int, int]:
        """Parse semantic version string into major, minor, patch."""
        try:
            # Remove 'v' prefix if present
            version = version.lstrip('v')
            parts = version.split('.')
            
            major = int(parts[0]) if len(parts) > 0 else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            
            return major, minor, patch
        except (ValueError, IndexError):
            return 1, 0, 0
    
    def format_semantic_version(self, major: int, minor: int, patch: int) -> str:
        """Format semantic version from components."""
        return f"{major}.{minor}.{patch}"
    
    def increment_patch(self, version: str) -> str:
        """Increment patch version (1.0.0 -> 1.0.1)."""
        major, minor, patch = self.parse_semantic_version(version)
        return self.format_semantic_version(major, minor, patch + 1)
    
    def increment_minor(self, version: str) -> str:
        """Increment minor version (1.0.0 -> 1.1.0)."""
        major, minor, patch = self.parse_semantic_version(version)
        return self.format_semantic_version(major, minor + 1, 0)
    
    def increment_major(self, version: str) -> str:
        """Increment major version (1.0.0 -> 2.0.0)."""
        major, minor, patch = self.parse_semantic_version(version)
        return self.format_semantic_version(major + 1, 0, 0)
    
    def get_latest_git_tag(self) -> Optional[str]:
        """Get the latest git tag."""
        try:
            result = subprocess.run(
                ['git', 'describe', '--tags', '--abbrev=0'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        return None
    
    def get_git_tag_count(self) -> int:
        """Get number of commits since last tag."""
        try:
            result = subprocess.run(
                ['git', 'rev-list', '--count', 'HEAD'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError, ValueError):
            pass
        return 0
    
    def increment_based_on_git_tags(self) -> str:
        """Increment version based on git tags and commits."""
        latest_tag = self.get_latest_git_tag()
        
        if latest_tag:
            # Use git tag as base version
            base_version = latest_tag.lstrip('v')
            commit_count = self.get_git_tag_count()
            
            if commit_count > 0:
                # Increment patch for each commit since last tag
                major, minor, patch = self.parse_semantic_version(base_version)
                return self.format_semantic_version(major, minor, patch + commit_count)
            else:
                return base_version
        else:
            # No tags found, start from 1.0.0
            commit_count = self.get_git_tag_count()
            return self.format_semantic_version(1, 0, commit_count)
    
    def increment_based_on_strategy(self, strategy: str = 'auto') -> str:
        """Increment version based on specified strategy."""
        current_version = self.get_current_version()
        
        if strategy == 'major':
            return self.increment_major(current_version)
        elif strategy == 'minor':
            return self.increment_minor(current_version)
        elif strategy == 'patch':
            return self.increment_patch(current_version)
        elif strategy == 'git':
            return self.increment_based_on_git_tags()
        elif strategy == 'auto':
            # Auto-detect based on git status
            if self._has_git_changes():
                return self.increment_patch(current_version)
            else:
                return current_version
        else:
            return current_version
    
    def _has_git_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return len(result.stdout.strip()) > 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        return False
    
    def save_version(self, version: str):
        """Save version to files."""
        # Save to version.txt
        try:
            with open(self.version_txt, 'w') as f:
                f.write(version)
            print(f"✓ Version saved to {self.version_txt}: {version}")
        except IOError as e:
            print(f"✗ Error saving to {self.version_txt}: {e}")
        
        # Update version.json if it exists
        if os.path.exists(self.version_file):
            try:
                with open(self.version_file, 'r') as f:
                    data = json.load(f)
                data['version'] = version
                data['version_updated'] = datetime.datetime.now().isoformat()
                
                with open(self.version_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"✓ Version updated in {self.version_file}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"✗ Error updating {self.version_file}: {e}")

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Increment version for Portfolio Strategy Analytics')
    parser.add_argument('--strategy', choices=['major', 'minor', 'patch', 'git', 'auto'], 
                       default='auto', help='Version increment strategy')
    parser.add_argument('--save', action='store_true', help='Save the new version to files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without saving')
    
    args = parser.parse_args()
    
    incrementer = VersionIncrementer()
    current_version = incrementer.get_current_version()
    new_version = incrementer.increment_based_on_strategy(args.strategy)
    
    print(f"Current version: {current_version}")
    print(f"New version: {new_version}")
    
    if new_version != current_version:
        if args.dry_run:
            print("✓ Dry run: Version would be updated")
        elif args.save:
            incrementer.save_version(new_version)
            print("✓ Version saved successfully")
        else:
            print("ℹ Use --save to save the new version")
    else:
        print("ℹ Version unchanged")

if __name__ == '__main__':
    main()
