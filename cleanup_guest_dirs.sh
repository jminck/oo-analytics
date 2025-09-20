#!/bin/bash
# Azure WebJob Entry Point: Cleanup Old Guest Directories
# This script runs the Python cleanup script in the Linux container

echo "Starting guest directory cleanup WebJob at $(date)"

# Change to the application directory
cd /home/site/wwwroot

# Run the Python cleanup script
python3 cleanup_guest_dirs.py

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "WebJob completed successfully at $(date)"
else
    echo "WebJob failed with exit code $EXIT_CODE at $(date)"
fi

exit $EXIT_CODE
