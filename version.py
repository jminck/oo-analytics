"""
Version management for Portfolio Strategy Analytics.
Automatically generates build information and version details.
"""

import os
import json
import subprocess
import datetime
from typing import Dict, Optional

class VersionManager:
    """Manages application version and build information."""
    
    def __init__(self):
        self.version_file = 'version.json'
        self.version_info = self._load_or_generate_version()
    
    def _load_or_generate_version(self) -> Dict:
        """Load existing version info or generate new version."""
        if os.path.exists(self.version_file):
            try:
                with open(self.version_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # If file is corrupted, regenerate
                pass
        
        return self._generate_new_version()
    
    def _generate_new_version(self) -> Dict:
        """Generate new version information."""
        # Get git information if available
        git_info = self._get_git_info()
        
        # Generate build timestamp
        build_time = datetime.datetime.now().isoformat()
        
        # Create version info
        version_info = {
            'version': self._get_version_number(),
            'build_time': build_time,
            'build_date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'build_time_short': datetime.datetime.now().strftime('%H:%M:%S'),
            'git': git_info,
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'python_version': self._get_python_version()
        }
        
        # Save to file
        self._save_version(version_info)
        
        return version_info
    
    def _get_git_info(self) -> Dict:
        """Get git repository information."""
        git_info = {
            'commit_hash': 'unknown',
            'commit_short': 'unknown',
            'branch': 'unknown',
            'is_dirty': False
        }
        
        try:
            # Get commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                git_info['commit_hash'] = result.stdout.strip()
                git_info['commit_short'] = result.stdout.strip()[:8]
            
            # Get current branch
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
            
            # Check if working directory is dirty
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                git_info['is_dirty'] = len(result.stdout.strip()) > 0
                
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            # Git not available or error occurred
            pass
        
        return git_info
    
    def _get_version_number(self) -> str:
        """Get version number from environment or default."""
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
        
        # Check for auto-increment setting
        auto_increment = os.getenv('AUTO_INCREMENT_VERSION', 'false').lower() == 'true'
        if auto_increment:
            return self._get_auto_incremented_version()
        
        # Default version
        return '1.0.0'
    
    def _get_auto_incremented_version(self) -> str:
        """Get auto-incremented version based on git tags."""
        try:
            # Get latest git tag
            result = subprocess.run(
                ['git', 'describe', '--tags', '--abbrev=0'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                latest_tag = result.stdout.strip().lstrip('v')
                # Get commit count since last tag
                result = subprocess.run(
                    ['git', 'rev-list', '--count', 'HEAD'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    commit_count = int(result.stdout.strip())
                    if commit_count > 0:
                        # Increment patch version for each commit
                        parts = latest_tag.split('.')
                        if len(parts) >= 3:
                            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
                            return f"{major}.{minor}.{patch + commit_count}"
                        else:
                            return f"{latest_tag}.{commit_count}"
                    else:
                        return latest_tag
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError, ValueError):
            pass
        
        # Fallback to default
        return '1.0.0'
    
    def _get_python_version(self) -> str:
        """Get Python version information."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _save_version(self, version_info: Dict):
        """Save version information to file."""
        try:
            with open(self.version_file, 'w') as f:
                json.dump(version_info, f, indent=2)
        except IOError:
            # If we can't save, continue without saving
            pass
    
    def get_version_string(self) -> str:
        """Get formatted version string for display."""
        version = self.version_info['version']
        build_date = self.version_info['build_date']
        commit_short = self.version_info['git']['commit_short']
        
        if commit_short != 'unknown':
            return f"v{version} ({build_date} - {commit_short})"
        else:
            return f"v{version} ({build_date})"
    
    def get_full_version_info(self) -> Dict:
        """Get complete version information."""
        return self.version_info.copy()
    
    def get_build_badge(self) -> str:
        """Get build status badge for display."""
        env = self.version_info['environment']
        is_dirty = self.version_info['git']['is_dirty']
        
        if env == 'production':
            if is_dirty:
                return '<span class="badge badge-warning">PROD*</span>'
            else:
                return '<span class="badge badge-success">PROD</span>'
        elif env == 'staging':
            return '<span class="badge badge-info">STAGING</span>'
        else:
            if is_dirty:
                return '<span class="badge badge-danger">DEV*</span>'
            else:
                return '<span class="badge badge-secondary">DEV</span>'

# Global version manager instance
version_manager = VersionManager()

def get_version_string() -> str:
    """Get formatted version string."""
    return version_manager.get_version_string()

def get_version_info() -> Dict:
    """Get complete version information."""
    return version_manager.get_full_version_info()

def get_build_badge() -> str:
    """Get build status badge."""
    return version_manager.get_build_badge()
