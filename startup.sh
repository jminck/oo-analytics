#!/bin/bash
set -e

echo "=== Starting Azure App Service startup script ==="

# Change to the correct working directory
cd /home/site/wwwroot

echo "Current working directory: $(pwd)"
echo "Files in current directory: $(ls -la)"

# Install dependencies first
#echo "Installing Python dependencies..."
#pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p instance
mkdir -p data/guest
mkdir -p data/users

# Set proper permissions
chmod 755 instance
chmod 755 data

# Start the application
echo "Starting application with gunicorn..."
exec gunicorn --bind 0.0.0.0:8000 --workers 1 app:application
