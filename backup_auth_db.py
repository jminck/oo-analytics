"""
Backup and restore utility for portfolio_auth.db
"""
import os
import shutil
import sqlite3
from datetime import datetime

def backup_auth_db():
    """Backup the authentication database before deployment."""
    if os.environ.get('WEBSITES_PORT'):
        # Azure App Service
        db_path = '/home/site/wwwroot/instance/portfolio_auth.db'
        backup_dir = '/home/site/wwwroot/backups'
    else:
        # Local development
        db_path = os.path.join('instance', 'portfolio_auth.db')
        backup_dir = 'backups'
    
    if not os.path.exists(db_path):
        print("No auth database found to backup")
        return None
    
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    
    # Create timestamped backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f'portfolio_auth_backup_{timestamp}.db')
    
    # Copy database
    shutil.copy2(db_path, backup_path)
    print(f"Auth database backed up to: {backup_path}")
    
    return backup_path

def restore_auth_db(backup_path=None):
    """Restore the authentication database from backup."""
    if os.environ.get('WEBSITES_PORT'):
        # Azure App Service
        db_path = '/home/site/wwwroot/instance/portfolio_auth.db'
        backup_dir = '/home/site/wwwroot/backups'
    else:
        # Local development
        db_path = os.path.join('instance', 'portfolio_auth.db')
        backup_dir = 'backups'
    
    if not backup_path:
        # Find the most recent backup
        if not os.path.exists(backup_dir):
            print("No backup directory found")
            return False
        
        backups = [f for f in os.listdir(backup_dir) if f.startswith('portfolio_auth_backup_') and f.endswith('.db')]
        if not backups:
            print("No backup files found")
            return False
        
        backup_path = os.path.join(backup_dir, sorted(backups)[-1])
    
    if not os.path.exists(backup_path):
        print(f"Backup file not found: {backup_path}")
        return False
    
    # Ensure target directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Restore database
    shutil.copy2(backup_path, db_path)
    print(f"Auth database restored from: {backup_path}")
    
    return True

def verify_auth_db():
    """Verify the authentication database is working."""
    if os.environ.get('WEBSITES_PORT'):
        db_path = '/home/site/wwwroot/instance/portfolio_auth.db'
    else:
        db_path = os.path.join('instance', 'portfolio_auth.db')
    
    if not os.path.exists(db_path):
        print("Auth database does not exist")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        
        print(f"Auth database verified. Tables: {[t[0] for t in tables]}")
        return True
    except Exception as e:
        print(f"Auth database verification failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "backup":
            backup_auth_db()
        elif command == "restore":
            backup_path = sys.argv[2] if len(sys.argv) > 2 else None
            restore_auth_db(backup_path)
        elif command == "verify":
            verify_auth_db()
        else:
            print("Usage: python backup_auth_db.py [backup|restore|verify] [backup_path]")
    else:
        print("Usage: python backup_auth_db.py [backup|restore|verify] [backup_path]")
