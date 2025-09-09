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
        # Use absolute path for SQLite in production
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
    
    # File Storage
    DATA_BASE_DIR = os.path.join(BASE_DIR, 'data')
    GUEST_DATA_DIR = os.path.join(BASE_DIR, 'data', 'guest')
    USER_DATA_DIR = os.path.join(BASE_DIR, 'data', 'users')