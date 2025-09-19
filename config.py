"""
Configuration file for Flask application with OAuth support.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Base directory - define first so it can be used by other settings
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production'
    
    # Database Configuration
    # For production, use Azure SQL Database or PostgreSQL
    # For development, use SQLite
    if os.environ.get('DATABASE_URL'):
        SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    else:
        # Use Azure-compatible path for SQLite
        if os.environ.get('WEBSITES_PORT'):
            # Azure App Service - use persistent directory
            db_path = '/home/site/wwwroot/instance/portfolio_auth.db'
            # Ensure the directory exists and is writable
            os.makedirs('/home/site/wwwroot/instance', exist_ok=True)
        else:
            # Local development
            db_path = os.path.join(BASE_DIR, 'instance', 'portfolio_auth.db')
        SQLALCHEMY_DATABASE_URI = f'sqlite:///{db_path}'
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # OAuth Configuration
    DISCORD_CLIENT_ID = os.environ.get('DISCORD_CLIENT_ID')
    DISCORD_CLIENT_SECRET = os.environ.get('DISCORD_CLIENT_SECRET')
    
    GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
    GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')
    
    # OAuth URLs
    GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid_configuration"
    
    # OAuth Redirect URIs (for production deployment)
    GOOGLE_REDIRECT_URI = os.environ.get('GOOGLE_REDIRECT_URI')
    DISCORD_REDIRECT_URI = os.environ.get('DISCORD_REDIRECT_URI')
    
    # Debug logging for environment variables
    print(f"DEBUG CONFIG: GOOGLE_REDIRECT_URI = {GOOGLE_REDIRECT_URI}")
    print(f"DEBUG CONFIG: DISCORD_REDIRECT_URI = {DISCORD_REDIRECT_URI}")
    
    # File Storage
    if os.environ.get('WEBSITE_SITE_NAME'):
        # Azure App Service - use persistent paths
        DATA_BASE_DIR = '/home/site/wwwroot/data'
        GUEST_DATA_DIR = '/home/site/wwwroot/data/guest'
        USER_DATA_DIR = '/home/site/wwwroot/data/users'
    else:
        # Local development
        DATA_BASE_DIR = os.path.join(BASE_DIR, 'data')
        GUEST_DATA_DIR = os.path.join(BASE_DIR, 'data', 'guest')
        USER_DATA_DIR = os.path.join(BASE_DIR, 'data', 'users')