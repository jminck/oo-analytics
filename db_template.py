"""
Database template utility for initializing SQLite databases from templates.
Ensures existing databases are preserved during deployments.
"""

import os
import sqlite3
import shutil
from pathlib import Path

class DatabaseTemplate:
    """Manages database templates and initialization."""
    
    def __init__(self, db_path: str, template_path: str = None):
        self.db_path = db_path
        self.template_path = template_path or self._get_default_template_path()
        self.db_dir = os.path.dirname(db_path)
    
    def _get_default_template_path(self) -> str:
        """Get the default template path based on environment."""
        if os.environ.get('WEBSITES_PORT'):
            # Azure App Service
            return '/home/site/wwwroot/db_templates/portfolio_auth_template.db'
        else:
            # Local development
            return os.path.join('db_templates', 'portfolio_auth_template.db')
    
    def ensure_database_exists(self):
        """Ensure database exists, creating from template if needed."""
        # Ensure directory exists
        os.makedirs(self.db_dir, exist_ok=True)
        
        if os.path.exists(self.db_path):
            print(f"✅ Database already exists: {self.db_path}")
            return True
        
        if not os.path.exists(self.template_path):
            print(f"⚠️ Template not found: {self.template_path}")
            print("Creating empty database...")
            self._create_empty_database()
            return True
        
        print(f"📋 Creating database from template: {self.template_path}")
        try:
            shutil.copy2(self.template_path, self.db_path)
            print(f"✅ Database created from template: {self.db_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to create database from template: {e}")
            print("Creating empty database...")
            self._create_empty_database()
            return False
    
    def _create_empty_database(self):
        """Create an empty database with basic schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create basic tables (you can expand this based on your needs)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    avatar_url VARCHAR(200),
                    provider VARCHAR(50) NOT NULL,
                    provider_id VARCHAR(100) NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_login DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS custom_blackout_list (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    name VARCHAR(100) NOT NULL,
                    dates TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            print(f"✅ Empty database created: {self.db_path}")
            
        except Exception as e:
            print(f"❌ Failed to create empty database: {e}")
            raise
    
    def create_template_from_existing(self, source_db_path: str = None):
        """Create a template from an existing database."""
        source = source_db_path or self.db_path
        
        if not os.path.exists(source):
            print(f"❌ Source database not found: {source}")
            return False
        
        # Ensure template directory exists
        template_dir = os.path.dirname(self.template_path)
        os.makedirs(template_dir, exist_ok=True)
        
        try:
            shutil.copy2(source, self.template_path)
            print(f"✅ Template created: {self.template_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to create template: {e}")
            return False
    
    def backup_database(self, backup_dir: str = None):
        """Create a backup of the current database."""
        if not os.path.exists(self.db_path):
            print(f"❌ Database not found: {self.db_path}")
            return None
        
        if not backup_dir:
            backup_dir = os.path.join(self.db_dir, 'backups')
        
        os.makedirs(backup_dir, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(backup_dir, f'portfolio_auth_backup_{timestamp}.db')
        
        try:
            shutil.copy2(self.db_path, backup_path)
            print(f"✅ Database backed up: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"❌ Failed to backup database: {e}")
            return None
    
    def verify_database(self):
        """Verify the database is working correctly."""
        if not os.path.exists(self.db_path):
            print(f"❌ Database not found: {self.db_path}")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Check user count
            if 'user' in tables:
                cursor.execute("SELECT COUNT(*) FROM user")
                user_count = cursor.fetchone()[0]
                print(f"✅ Database verified. Tables: {tables}, Users: {user_count}")
            else:
                print(f"✅ Database verified. Tables: {tables}")
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Database verification failed: {e}")
            return False

# Convenience functions
def ensure_auth_database():
    """Ensure the authentication database exists."""
    if os.environ.get('WEBSITES_PORT'):
        # Azure App Service
        db_path = '/home/site/wwwroot/instance/portfolio_auth.db'
    else:
        # Local development
        db_path = os.path.join('instance', 'portfolio_auth.db')
    
    template = DatabaseTemplate(db_path)
    return template.ensure_database_exists()

def create_auth_template():
    """Create a template from the current auth database."""
    if os.environ.get('WEBSITES_PORT'):
        # Azure App Service
        db_path = '/home/site/wwwroot/instance/portfolio_auth.db'
    else:
        # Local development
        db_path = os.path.join('instance', 'portfolio_auth.db')
    
    template = DatabaseTemplate(db_path)
    return template.create_template_from_existing()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "init":
            ensure_auth_database()
        elif command == "template":
            create_auth_template()
        elif command == "verify":
            if os.environ.get('WEBSITES_PORT'):
                db_path = '/home/site/wwwroot/instance/portfolio_auth.db'
            else:
                db_path = os.path.join('instance', 'portfolio_auth.db')
            template = DatabaseTemplate(db_path)
            template.verify_database()
        else:
            print("Usage: python db_template.py [init|template|verify]")
    else:
        print("Usage: python db_template.py [init|template|verify]")
