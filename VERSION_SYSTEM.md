# Version System for Portfolio Strategy Analytics

This document describes the automated version system implemented for the Portfolio Strategy Analytics application.

## Overview

The version system automatically tracks and displays build information, including:
- Application version number
- Build timestamp and date
- Git commit information
- Environment status
- Python version
- Build host details

## Files

### Core Files
- `version.py` - Main version management module
- `build_version.py` - Build script for generating version information
- `version.json` - Generated version information (auto-created)
- `version.txt` - Simple version number file (auto-created)

### Integration
- `app.py` - Flask app integration with version endpoints
- `templates/dashboard.html` - UI display of version information

## Usage

### Automatic Version Generation

The version system automatically generates version information when the application starts. It will:

1. **Check for existing version.json** - If found, use it
2. **Generate new version info** - If not found or corrupted
3. **Save to version.json** - For future use

### Manual Version Generation

Run the build script to manually generate version information:

```bash
python build_version.py
```

This will:
- Generate complete version information
- Save to `version.json` and `version.txt`
- Display a summary of the build information

### Setting Version Numbers

You can set the version number in several ways:

#### 1. Environment Variable
```bash
export APP_VERSION="2.1.0"
python app.py
```

#### 2. Version File
Create a `version.txt` file:
```
2.1.0
```

#### 3. Package.json (if exists)
```json
{
  "version": "2.1.0"
}
```

#### 4. Default
If none of the above are found, defaults to `1.0.0`

## Display

### In the UI

The version information is displayed in the top navigation bar next to "Portfolio Strategy Analytics":

- **Version String**: `v1.0.0 (2024-01-15 - a1b2c3d4)`
- **Environment Badge**: Color-coded badge showing environment status
  - 🟢 **PROD** - Production environment
  - 🟡 **PROD*** - Production with uncommitted changes
  - 🔵 **STAGING** - Staging environment
  - ⚫ **DEV** - Development environment
  - 🔴 **DEV*** - Development with uncommitted changes

### Version Modal

Click on the version information to open a detailed modal showing:
- Complete version information
- Git details (branch, commit, status)
- Build environment details
- Python version
- Build host information

### API Endpoint

Access version information via API:

```bash
curl http://localhost:5000/api/version
```

Response:
```json
{
  "success": true,
  "version": {
    "version": "1.0.0",
    "build_time": "2024-01-15T10:30:45.123456",
    "build_date": "2024-01-15",
    "build_time_short": "10:30:45",
    "git": {
      "commit_hash": "a1b2c3d4e5f6...",
      "commit_short": "a1b2c3d4",
      "branch": "main",
      "is_dirty": false,
      "remote_url": "https://github.com/user/repo.git"
    },
    "environment": "production",
    "python_version": "3.9.7"
  }
}
```

## Environment Detection

The system automatically detects the environment:

- **Production**: `WEBSITES_PORT` environment variable (Azure App Service)
- **Staging**: `STAGING` environment variable
- **Development**: `FLASK_ENV=development` or default

## Git Integration

The version system integrates with Git to provide:
- Current branch name
- Latest commit hash (full and short)
- Working directory status (clean/dirty)
- Remote repository URL

**Note**: Git integration is optional. If Git is not available, the system will still work with "unknown" values.

## CI/CD Integration

### GitHub Actions

Add to your workflow:

```yaml
- name: Set version
  run: echo "APP_VERSION=${{ github.ref_name }}" >> $GITHUB_ENV

- name: Generate version info
  run: python build_version.py
```

### Azure DevOps

Add to your pipeline:

```yaml
- script: |
    echo "APP_VERSION=$(Build.SourceBranchName)" >> $(Build.SourcesDirectory)/.env
    python build_version.py
  displayName: 'Generate version information'
```

### Docker

In your Dockerfile:

```dockerfile
# Set version from build arg
ARG APP_VERSION=1.0.0
ENV APP_VERSION=${APP_VERSION}

# Generate version info
RUN python build_version.py
```

## Customization

### Custom Version Format

Modify the `get_version_string()` method in `version.py`:

```python
def get_version_string(self) -> str:
    version = self.version_info['version']
    build_date = self.version_info['build_date']
    commit_short = self.version_info['git']['commit_short']
    
    # Custom format
    return f"v{version}-{commit_short}-{build_date}"
```

### Custom Badge Colors

Modify the `get_build_badge()` method in `version.py`:

```python
def get_build_badge(self) -> str:
    env = self.version_info['environment']
    is_dirty = self.version_info['git']['is_dirty']
    
    if env == 'production':
        return '<span class="badge badge-success">LIVE</span>'
    # ... other customizations
```

## Troubleshooting

### Version Not Updating

1. **Check file permissions** - Ensure the app can write to `version.json`
2. **Clear version.json** - Delete the file to force regeneration
3. **Check Git status** - Ensure Git repository is accessible

### Git Information Missing

1. **Check Git installation** - Ensure Git is installed and accessible
2. **Check repository** - Ensure you're in a Git repository
3. **Check permissions** - Ensure the app can read Git information

### Environment Detection Issues

1. **Check environment variables** - Verify `ENVIRONMENT` or other detection variables
2. **Manual override** - Set `ENVIRONMENT` environment variable explicitly

## Security Considerations

- Version information is publicly accessible via the API endpoint
- Git information may reveal repository details
- Build host information may reveal infrastructure details
- Consider what information to expose in production environments

## Future Enhancements

Potential improvements:
- Integration with package managers (pip, npm)
- Docker image information
- Health check integration
- Version comparison and update notifications
- Automated version bumping based on Git tags
