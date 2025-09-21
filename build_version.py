#!/usr/bin/env python3
"""
Build script to generate version information for Portfolio Strategy Analytics.
This script can be run manually or integrated into CI/CD pipelines.
"""

import os
import sys
import json
import subprocess
import datetime
from pathlib import Path

def get_git_info():
    """Get git repository information."""
    git_info = {
        'commit_hash': 'unknown',
        'commit_short': 'unknown',
        'branch': 'unknown',
        'is_dirty': False,
        'remote_url': 'unknown'
    }
    
    try:
        # Get commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            git_info['commit_hash'] = result.stdout.strip()
            git_info['commit_short'] = result.stdout.strip()[:8]
        
        # Get current branch
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            git_info['branch'] = result.stdout.strip()
        
        # Check if working directory is dirty
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            git_info['is_dirty'] = len(result.stdout.strip()) > 0
        
        # Get remote URL
        result = subprocess.run(
            ['git', 'config', '--get', 'remote.origin.url'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            git_info['remote_url'] = result.stdout.strip()
            
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        print("Warning: Git information not available")
    
    return git_info

def get_python_version():
    """Get Python version information."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

def get_environment():
    """Determine environment based on environment variables."""
    env = os.getenv('ENVIRONMENT', 'development')
    
    # Auto-detect environment based on common patterns
    if os.getenv('WEBSITES_PORT'):  # Azure App Service
        env = 'production'
    elif os.getenv('STAGING'):
        env = 'staging'
    elif os.getenv('FLASK_ENV') == 'production':
        env = 'production'
    elif os.getenv('FLASK_ENV') == 'development':
        env = 'development'
    
    return env

def get_version_number():
    """Get version number from various sources."""
    # Check for version in environment variable
    env_version = os.getenv('APP_VERSION')
    if env_version:
        return env_version
    
    # Check for version in version.txt file
    if os.path.exists('version.txt'):
        try:
            with open('version.txt', 'r') as f:
                return f.read().strip()
        except IOError:
            pass
    
    # Check for version in package.json (if exists)
    if os.path.exists('package.json'):
        try:
            with open('package.json', 'r') as f:
                package_data = json.load(f)
                if 'version' in package_data:
                    return package_data['version']
        except (json.JSONDecodeError, IOError):
            pass
    
    # Default version
    return '1.0.0'

def generate_version_info():
    """Generate complete version information."""
    build_time = datetime.datetime.now()
    
    version_info = {
        'version': get_version_number(),
        'build_time': build_time.isoformat(),
        'build_date': build_time.strftime('%Y-%m-%d'),
        'build_time_short': build_time.strftime('%H:%M:%S'),
        'build_timestamp': int(build_time.timestamp()),
        'git': get_git_info(),
        'environment': get_environment(),
        'python_version': get_python_version(),
        'build_host': os.getenv('HOSTNAME', 'unknown'),
        'build_user': os.getenv('USER', os.getenv('USERNAME', 'unknown'))
    }
    
    return version_info

def save_version_file(version_info):
    """Save version information to version.json."""
    try:
        with open('version.json', 'w') as f:
            json.dump(version_info, f, indent=2)
        print(f"✓ Version file saved: version.json")
        return True
    except IOError as e:
        print(f"✗ Error saving version file: {e}")
        return False

def save_version_txt(version_info):
    """Save version number to version.txt."""
    try:
        with open('version.txt', 'w') as f:
            f.write(version_info['version'])
        print(f"✓ Version text file saved: version.txt")
        return True
    except IOError as e:
        print(f"✗ Error saving version.txt: {e}")
        return False

def print_version_summary(version_info):
    """Print a summary of the version information."""
    print("\n" + "="*60)
    print("PORTFOLIO STRATEGY ANALYTICS - BUILD VERSION")
    print("="*60)
    print(f"Version:        {version_info['version']}")
    print(f"Build Date:     {version_info['build_date']}")
    print(f"Build Time:     {version_info['build_time_short']}")
    print(f"Environment:    {version_info['environment']}")
    print(f"Python:         {version_info['python_version']}")
    print(f"Git Branch:     {version_info['git']['branch']}")
    print(f"Git Commit:     {version_info['git']['commit_short']}")
    print(f"Git Status:     {'Dirty' if version_info['git']['is_dirty'] else 'Clean'}")
    print(f"Build Host:     {version_info['build_host']}")
    print(f"Build User:     {version_info['build_user']}")
    print("="*60)

def main():
    """Main build script function."""
    print("Building version information for Portfolio Strategy Analytics...")
    
    # Generate version information
    version_info = generate_version_info()
    
    # Save version files
    success = True
    success &= save_version_file(version_info)
    success &= save_version_txt(version_info)
    
    # Print summary
    print_version_summary(version_info)
    
    if success:
        print("\n✓ Build version generation completed successfully!")
        return 0
    else:
        print("\n✗ Build version generation completed with errors!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
