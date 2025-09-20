"""
Admin module for site administration.
"""
import os
import shutil
from datetime import datetime, timedelta
from flask import Blueprint, render_template, jsonify, request, current_app
from flask_login import login_required, current_user
from auth import admin_required, db, User
from pathlib import Path

# Create admin blueprint
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/')
@login_required
@admin_required
def dashboard():
    """Admin dashboard."""
    return render_template('admin/dashboard.html')

@admin_bp.route('/users')
@login_required
@admin_required
def users():
    """List all users with file statistics."""
    users = User.query.order_by(User.created_at.desc()).all()
    user_data = []
    
    for user in users:
        # Get user's file statistics
        user_stats = get_user_file_stats(user)
        user_data.append({
            'id': user.id,
            'email': user.email,
            'name': user.name,
            'provider': user.provider,
            'created_at': user.created_at.isoformat(),
            'last_login': user.last_login.isoformat() if user.last_login else None,
            'is_admin': user.is_admin(),
            'file_stats': user_stats
        })
    
    return jsonify({
        'success': True,
        'data': user_data
    })

@admin_bp.route('/system-stats')
@login_required
@admin_required
def system_stats():
    """Get system statistics."""
    try:
        # User stats
        total_users = User.query.count()
        recent_users = User.query.filter(
            User.created_at >= datetime.utcnow() - timedelta(days=7)
        ).count()
        
        # File system stats
        guest_data_dir = current_app.config.get('GUEST_DATA_DIR', '')
        user_data_dir = current_app.config.get('USER_DATA_DIR', '')
        
        guest_stats = get_directory_stats(guest_data_dir)
        user_stats = get_directory_stats(user_data_dir)
        
        # Recycled folder stats
        recycled_stats = get_recycled_folder_stats()
        
        # Database stats
        db_size = get_database_size()
        
        return jsonify({
            'success': True,
            'data': {
                'users': {
                    'total': total_users,
                    'recent_7_days': recent_users
                },
                'storage': {
                    'guest_data': guest_stats,
                    'user_data': user_stats,
                    'recycled_data': recycled_stats,
                    'database_size': db_size
                },
                'admin_emails': current_app.config.get('ADMIN_EMAILS', [])
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@admin_bp.route('/cleanup-guest-data', methods=['POST'])
@login_required
@admin_required
def cleanup_guest_data():
    """Clean up old guest data directories."""
    try:
        guest_data_dir = current_app.config.get('GUEST_DATA_DIR', '')
        if not guest_data_dir or not os.path.exists(guest_data_dir):
            return jsonify({'success': False, 'error': 'Guest data directory not found'}), 404
        
        # Get cutoff time (24 hours ago)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        deleted_count = 0
        total_size_freed = 0
        
        for item in Path(guest_data_dir).iterdir():
            if item.is_dir():
                # Check if directory is older than 24 hours
                dir_mtime = datetime.fromtimestamp(item.stat().st_mtime)
                if dir_mtime < cutoff_time:
                    # Calculate size before deletion
                    dir_size = get_directory_size(item)
                    
                    # Delete the directory
                    shutil.rmtree(item)
                    
                    deleted_count += 1
                    total_size_freed += dir_size
        
        return jsonify({
            'success': True,
            'data': {
                'deleted_directories': deleted_count,
                'size_freed': total_size_freed,
                'message': f'Cleaned up {deleted_count} directories, freed {format_size(total_size_freed)}'
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@admin_bp.route('/logs')
@login_required
@admin_required
def logs():
    """View application logs (if available)."""
    # This would typically read from log files
    # For now, return a placeholder
    return jsonify({
        'success': True,
        'data': {
            'message': 'Log viewing not implemented yet',
            'logs': []
        }
    })

def get_directory_stats(directory_path):
    """Get statistics for a directory."""
    if not directory_path or not os.path.exists(directory_path):
        return {'folders': 0, 'files': 0, 'size': 0}
    
    folder_count = 0
    file_count = 0
    total_size = 0
    
    try:
        for item in Path(directory_path).rglob('*'):
            if item.is_file():
                file_count += 1
                try:
                    total_size += item.stat().st_size
                except (OSError, FileNotFoundError):
                    pass
            elif item.is_dir():
                folder_count += 1
    except Exception:
        pass
    
    return {
        'folders': folder_count,
        'files': file_count,
        'size': total_size
    }

def get_directory_stats_excluding_recycle(directory_path):
    """Get statistics for a directory, excluding the 'recycle' subfolder."""
    if not directory_path or not os.path.exists(directory_path):
        return {'folders': 0, 'files': 0, 'size': 0}
    
    folder_count = 0
    file_count = 0
    total_size = 0
    
    try:
        for item in Path(directory_path).rglob('*'):
            # Skip the recycle folder and its contents
            if 'recycle' in item.parts:
                continue
                
            if item.is_file():
                file_count += 1
                try:
                    total_size += item.stat().st_size
                except (OSError, FileNotFoundError):
                    pass
            elif item.is_dir():
                folder_count += 1
    except Exception:
        pass
    
    return {
        'folders': folder_count,
        'files': file_count,
        'size': total_size
    }

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

def get_database_size():
    """Get database file size."""
    try:
        db_path = current_app.config.get('SQLALCHEMY_DATABASE_URI', '').replace('sqlite:///', '')
        if os.path.exists(db_path):
            return os.path.getsize(db_path)
    except Exception:
        pass
    return 0

def get_user_file_stats(user):
    """Get file statistics for a specific user."""
    try:
        user_data_dir = current_app.config.get('USER_DATA_DIR', '')
        user_folder = os.path.join(user_data_dir, str(user.id))
        
        if not os.path.exists(user_folder):
            return {
                'files': 0,
                'size': 0,
                'folders': 0,
                'recycled_size': 0
            }
        
        # Get main directory stats (excluding recycle folder)
        main_stats = get_directory_stats_excluding_recycle(user_folder)
        
        # Get recycled folder stats (correct folder name: 'recycle')
        recycled_folder = os.path.join(user_folder, 'recycle')
        recycled_stats = get_directory_stats(recycled_folder) if os.path.exists(recycled_folder) else {'size': 0}
        
        # Add recycled size to main stats
        main_stats['recycled_size'] = recycled_stats['size']
        
        return main_stats
    except Exception:
        return {
            'files': 0,
            'size': 0,
            'folders': 0,
            'recycled_size': 0
        }

def get_recycled_folder_stats():
    """Get statistics for recycled folders."""
    try:
        # Look for recycled folders in both guest and user data directories
        guest_data_dir = current_app.config.get('GUEST_DATA_DIR', '')
        user_data_dir = current_app.config.get('USER_DATA_DIR', '')
        
        total_files = 0
        total_size = 0
        total_folders = 0
        
        # Check guest recycled folder
        guest_recycled = os.path.join(guest_data_dir, 'recycle')
        if os.path.exists(guest_recycled):
            guest_stats = get_directory_stats(guest_recycled)
            total_files += guest_stats['files']
            total_size += guest_stats['size']
            total_folders += guest_stats['folders']
        
        # Check user recycled folders
        if os.path.exists(user_data_dir):
            for user_folder in os.listdir(user_data_dir):
                user_recycled = os.path.join(user_data_dir, user_folder, 'recycle')
                if os.path.exists(user_recycled):
                    user_stats = get_directory_stats(user_recycled)
                    total_files += user_stats['files']
                    total_size += user_stats['size']
                    total_folders += user_stats['folders']
        
        return {
            'files': total_files,
            'size': total_size,
            'folders': total_folders
        }
    except Exception:
        return {
            'files': 0,
            'size': 0,
            'folders': 0
        }

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
