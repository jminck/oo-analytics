# Setup script for Git hooks to automatically increment version
# Run this script to configure version incrementing on commits

Write-Host "🔧 Setting up Git hooks for automatic version incrementing..." -ForegroundColor Yellow

# Check if we're in a git repository
if (-not (Test-Path ".git")) {
    Write-Host "❌ Error: Not in a Git repository. Please run this from the root of your Git repo." -ForegroundColor Red
    exit 1
}

# Create hooks directory if it doesn't exist
$HooksDir = ".git\hooks"
if (-not (Test-Path $HooksDir)) {
    New-Item -ItemType Directory -Path $HooksDir -Force | Out-Null
}

# Determine which hook to use based on system
$HookFile = ""
if ($PSVersionTable.PSVersion.Major -ge 5) {
    # Use PowerShell version for modern Windows
    $HookFile = "pre-commit.ps1"
    Write-Host "📝 Using PowerShell hook (recommended for Windows)" -ForegroundColor Green
} else {
    # Use batch version for older systems
    $HookFile = "pre-commit.bat"
    Write-Host "📝 Using batch hook (fallback for older systems)" -ForegroundColor Yellow
}

# Copy the appropriate hook file
$SourceHook = ".git\hooks\$HookFile"
$TargetHook = ".git\hooks\pre-commit"

if (Test-Path $SourceHook) {
    Copy-Item $SourceHook $TargetHook -Force
    Write-Host "✅ Copied $HookFile to pre-commit hook" -ForegroundColor Green
} else {
    Write-Host "❌ Error: Hook file $SourceHook not found" -ForegroundColor Red
    exit 1
}

# Make the hook executable (Unix-style, works in Git Bash)
try {
    & git update-index --chmod=+x .git/hooks/pre-commit 2>$null
    Write-Host "✅ Made pre-commit hook executable" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Warning: Could not make hook executable (this is normal on Windows)" -ForegroundColor Yellow
}

# Test if the pre-commit-version.py script exists
if (Test-Path "pre-commit-version.py") {
    Write-Host "✅ Found pre-commit-version.py script" -ForegroundColor Green
} else {
    Write-Host "❌ Error: pre-commit-version.py script not found" -ForegroundColor Red
    Write-Host "   Please ensure the pre-commit-version.py file exists in the repository root" -ForegroundColor Yellow
    exit 1
}

# Test if version.txt exists
if (Test-Path "version.txt") {
    Write-Host "✅ Found version.txt file" -ForegroundColor Green
} else {
    Write-Host "⚠️  Warning: version.txt not found, creating initial version..." -ForegroundColor Yellow
    "1.0.0" | Out-File -FilePath "version.txt" -Encoding UTF8
    Write-Host "✅ Created initial version.txt with version 1.0.0" -ForegroundColor Green
}

Write-Host ""
Write-Host "🎉 Git hooks setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "How it works:" -ForegroundColor Cyan
Write-Host "• Every time you commit, the hook will check for relevant changes" -ForegroundColor White
Write-Host "• If changes are detected (excluding version files), the patch version will increment" -ForegroundColor White
Write-Host "• The updated version files will be automatically added to your commit" -ForegroundColor White
Write-Host ""
Write-Host "To test the hook:" -ForegroundColor Cyan
Write-Host "1. Make a change to any file (except version.txt or version.json)" -ForegroundColor White
Write-Host "2. Run: git add ." -ForegroundColor White
Write-Host "3. Run: git commit -m 'Test commit'" -ForegroundColor White
Write-Host "4. Check that version.txt was updated" -ForegroundColor White
Write-Host ""
Write-Host "To disable the hook temporarily:" -ForegroundColor Cyan
Write-Host "• Use: git commit --no-verify -m 'Skip version increment'" -ForegroundColor White