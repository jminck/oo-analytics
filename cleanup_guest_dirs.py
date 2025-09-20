#!/usr/bin/env python3
"""
Azure WebJob Script: Cleanup Old Guest Directories

This script deletes all subfolders under /home/site/wwwroot/data/guest/ 
that are older than 24 hours, regardless of whether they contain files.

Usage: python cleanup_guest_dirs.py
"""

import os
import time
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging to stdout only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def cleanup_old_guest_directories():
    """
    Delete all subfolders under /home/site/wwwroot/data/guest/ 
    that are older than 24 hours.
    """
    base_path = Path('/home/site/wwwroot/data/guest')
    
    if not base_path.exists():
        logging.warning(f"Base path does not exist: {base_path}")
        return 0
    
    # Calculate cutoff time (24 hours ago)
    cutoff_time = time.time() - (24 * 60 * 60)  # 24 hours in seconds
    
    deleted_count = 0
    total_size_freed = 0
    
    try:
        # Get all subdirectories
        subdirs = [d for d in base_path.iterdir() if d.is_dir()]
        
        logging.info(f"Found {len(subdirs)} subdirectories to check")
        
        for subdir in subdirs:
            try:
                # Get directory modification time
                dir_mtime = subdir.stat().st_mtime
                
                # Check if directory is older than 24 hours
                if dir_mtime < cutoff_time:
                    # Calculate size before deletion
                    dir_size = get_directory_size(subdir)
                    
                    # Delete the directory and all its contents
                    shutil.rmtree(subdir)
                    
                    deleted_count += 1
                    total_size_freed += dir_size
                    
                    logging.info(f"Deleted: {subdir.name} (age: {get_age_string(dir_mtime)}, size: {format_size(dir_size)})")
                else:
                    age = get_age_string(dir_mtime)
                    logging.debug(f"Keeping: {subdir.name} (age: {age})")
                    
            except PermissionError as e:
                logging.error(f"Permission denied deleting {subdir.name}: {e}")
            except OSError as e:
                logging.error(f"OS error deleting {subdir.name}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error deleting {subdir.name}: {e}")
    
    except Exception as e:
        logging.error(f"Error accessing base directory {base_path}: {e}")
        return 0
    
    # Log summary
    logging.info(f"Cleanup completed: {deleted_count} directories deleted, {format_size(total_size_freed)} freed")
    
    return deleted_count

def get_directory_size(path):
    """Calculate total size of directory in bytes."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except Exception:
        pass
    return total_size

def get_age_string(timestamp):
    """Convert timestamp to human-readable age string."""
    age_seconds = time.time() - timestamp
    
    if age_seconds < 60:
        return f"{int(age_seconds)}s"
    elif age_seconds < 3600:
        return f"{int(age_seconds/60)}m"
    elif age_seconds < 86400:
        return f"{int(age_seconds/3600)}h"
    else:
        return f"{int(age_seconds/86400)}d"

def format_size(size_bytes):
    """Convert bytes to human-readable size string."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def main():
    """Main entry point for the WebJob."""
    logging.info("Starting guest directory cleanup WebJob")
    logging.info(f"Current time: {datetime.now().isoformat()}")
    
    try:
        deleted_count = cleanup_old_guest_directories()
        
        if deleted_count > 0:
            logging.info(f"WebJob completed successfully: {deleted_count} directories cleaned up")
        else:
            logging.info("WebJob completed: No directories needed cleanup")
            
    except Exception as e:
        logging.error(f"WebJob failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
