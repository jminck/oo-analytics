"""
Authentication module with OAuth support for Discord and Google.
"""
import os
import json
import requests
from urllib.parse import urlencode
from functools import wraps
from flask import Blueprint, request, redirect, url_for, session, jsonify, current_app
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from authlib.integrations.flask_client import OAuth
from datetime import datetime
import secrets

# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()
oauth = OAuth()

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    avatar_url = db.Column(db.String(200))
    provider = db.Column(db.String(50), nullable=False)  # 'discord' or 'google'
    provider_id = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<User {self.email}>'
    
    def get_data_folder(self):
        """Get user-specific data folder path."""
        return os.path.join(current_app.config['USER_DATA_DIR'], str(self.id))
    
    def ensure_data_folder(self):
        """Ensure user's data folder exists."""
        folder = self.get_data_folder()
        os.makedirs(folder, exist_ok=True)
        return folder
    
    def is_admin(self):
        """Check if user is an admin based on email."""
        if not self.email:
            return False
        admin_emails = current_app.config.get('ADMIN_EMAILS', [])
        return self.email.lower() in admin_emails

# Create authentication blueprint
auth_bp = Blueprint('auth', __name__)

def admin_required(f):
    """Decorator to require admin access."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('auth.login'))
        if not current_user.is_admin():
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def init_auth(app):
    """Initialize authentication for the Flask app."""
    # Configure Flask-Login
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    # Initialize database
    db.init_app(app)
    
    # Initialize OAuth
    oauth.init_app(app)
    
    # Register OAuth providers
    discord_client_id = app.config.get('DISCORD_CLIENT_ID')
    discord_client_secret = app.config.get('DISCORD_CLIENT_SECRET')
    
    print(f"Discord Client ID: {discord_client_id[:10] + '...' if discord_client_id else 'None'}")
    print(f"Discord Client Secret: {'Set' if discord_client_secret else 'Not set'}")
    
    if discord_client_id and discord_client_secret:
        oauth.register(
            name='discord',
            client_id=discord_client_id,
            client_secret=discord_client_secret,
            authorize_url='https://discord.com/api/oauth2/authorize',
            access_token_url='https://discord.com/api/oauth2/token',
            client_kwargs={
                'scope': 'identify email'
            }
        )
        print("Discord OAuth client registered successfully")
    else:
        print("WARNING: Discord OAuth credentials not found - Discord login will not work")
    
    # Register Google OAuth
    google_client_id = app.config.get('GOOGLE_CLIENT_ID')
    google_client_secret = app.config.get('GOOGLE_CLIENT_SECRET')
    
    print(f"Google Client ID: {google_client_id[:20] + '...' if google_client_id else 'None'}")
    print(f"Google Client Secret: {'Set' if google_client_secret else 'Not set'}")
    
    if google_client_id and google_client_secret:
        oauth.register(
            name='google',
            client_id=google_client_id,
            client_secret=google_client_secret,
            authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
            access_token_url='https://oauth2.googleapis.com/token',
            userinfo_endpoint='https://www.googleapis.com/oauth2/v3/userinfo',
            jwks_uri='https://www.googleapis.com/oauth2/v3/certs',
            client_kwargs={
                'scope': 'openid email profile'
            }
        )
        print("Google OAuth client registered successfully")
    else:
        print("WARNING: Google OAuth credentials not found - Google login will not work")
    
    # Ensure database exists from template before creating tables
    from db_template import ensure_auth_database
    ensure_auth_database()
    
    # Create tables
    with app.app_context():
        db.create_all()
        
        # Ensure data directories exist
        os.makedirs(app.config['GUEST_DATA_DIR'], exist_ok=True)
        os.makedirs(app.config['USER_DATA_DIR'], exist_ok=True)

def guest_mode_required(f):
    """Decorator to ensure guest mode session exists."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated and 'guest_id' not in session:
            # Create guest session
            session['guest_id'] = secrets.token_urlsafe(16)
            session['is_guest'] = True
            
            # Ensure guest data folder exists
            guest_folder = get_guest_data_folder()
            os.makedirs(guest_folder, exist_ok=True)
            
        return f(*args, **kwargs)
    return decorated_function

def get_current_data_folder():
    """Get the current user's data folder (user or guest)."""
    if current_user.is_authenticated:
        return current_user.get_data_folder()
    else:
        # For guest users, ensure we have a persistent guest_id
        guest_id = session.get('guest_id')
        if not guest_id:
            # Create a new guest session
            guest_id = secrets.token_urlsafe(16)
            session['guest_id'] = guest_id
            session['is_guest'] = True
            print(f"Created new guest session: {guest_id}")
        else:
            print(f"Using existing guest session: {guest_id}")
        
        return get_guest_data_folder()

def get_guest_data_folder():
    """Get guest data folder path."""
    guest_id = session.get('guest_id', 'default')
    return os.path.join(current_app.config['GUEST_DATA_DIR'], guest_id)

# Authentication Routes
@auth_bp.route('/login')
def login():
    """Show login page with OAuth options."""
    return jsonify({
        'login_urls': {
            'discord': url_for('auth.discord_login'),
            'google': url_for('auth.google_login'),
            'guest': url_for('auth.guest_login')
        },
        'current_user': {
            'is_authenticated': current_user.is_authenticated,
            'is_guest': session.get('is_guest', False),
            'name': current_user.name if current_user.is_authenticated else session.get('guest_id', 'Guest')
        }
    })

@auth_bp.route('/login/discord')
def discord_login():
    """Initiate Discord OAuth login."""
    # Check if user is already logged in with Discord
    if current_user.is_authenticated and current_user.provider == 'discord':
        # User is already logged in with Discord, redirect to dashboard
        return redirect(url_for('dashboard'))
    
    # Check if there's a stored Discord token in session that's still valid
    discord_token = session.get('discord_token')
    if discord_token:
        try:
            # Try to validate the existing token
            headers = {
                'Authorization': f"Bearer {discord_token}"
            }
            resp = requests.get('https://discord.com/api/users/@me', headers=headers)
            if resp.status_code == 200:
                # Token is still valid, get user info and log them in
                user_info = resp.json()
                user = User.query.filter_by(provider='discord', provider_id=str(user_info['id'])).first()
                
                if user:
                    # Update last login and profile info
                    user.last_login = datetime.utcnow()
                    user.name = user_info.get('username', user.name)
                    user.avatar_url = f"https://cdn.discordapp.com/avatars/{user_info['id']}/{user_info.get('avatar', '')}.png" if user_info.get('avatar') else user.avatar_url
                    db.session.commit()
                    
                    # Clear guest session
                    session.pop('guest_id', None)
                    session.pop('is_guest', None)
                    
                    # Log in user
                    login_user(user, remember=True)
                    
                    return redirect(url_for('dashboard'))
        except Exception as e:
            # Token validation failed, clear it and continue with OAuth flow
            session.pop('discord_token', None)
            current_app.logger.debug(f"Discord token validation failed: {str(e)}")
    
    # No valid session found, proceed with OAuth flow
    # Use configured redirect URI if available, otherwise generate dynamically
    print(f"DEBUG: Checking for DISCORD_REDIRECT_URI in config...")
    print(f"DEBUG: Available config keys: {list(current_app.config.keys())}")
    print(f"DEBUG: DISCORD_REDIRECT_URI value: {current_app.config.get('DISCORD_REDIRECT_URI')}")
    
    if current_app.config.get('DISCORD_REDIRECT_URI'):
        redirect_uri = current_app.config['DISCORD_REDIRECT_URI']
        print(f"Using configured Discord redirect URI: {redirect_uri}")
    else:
        redirect_uri = url_for('auth.discord_callback', _external=True)
        # Force localhost usage for consistency with Discord app settings
        redirect_uri = redirect_uri.replace('127.0.0.1', 'localhost')
        print(f"Using dynamic Discord redirect URI: {redirect_uri}")
        print(f"DEBUG: Generated redirect URI from url_for: {url_for('auth.discord_callback', _external=True)}")
    return oauth.discord.authorize_redirect(redirect_uri)

@auth_bp.route('/check-discord-auth')
def check_discord_auth():
    """Check if user has valid Discord authentication without redirecting."""
    if current_user.is_authenticated and current_user.provider == 'discord':
        return jsonify({
            'authenticated': True,
            'user': {
                'name': current_user.name,
                'email': current_user.email,
                'avatar_url': current_user.avatar_url
            }
        })
    
    # Check for stored Discord token
    discord_token = session.get('discord_token')
    if discord_token:
        try:
            headers = {
                'Authorization': f"Bearer {discord_token}"
            }
            resp = requests.get('https://discord.com/api/users/@me', headers=headers)
            if resp.status_code == 200:
                user_info = resp.json()
                user = User.query.filter_by(provider='discord', provider_id=str(user_info['id'])).first()
                
                if user:
                    return jsonify({
                        'authenticated': True,
                        'user': {
                            'name': user.name,
                            'email': user.email,
                            'avatar_url': user.avatar_url
                        }
                    })
        except Exception:
            session.pop('discord_token', None)
    
    return jsonify({'authenticated': False})

@auth_bp.route('/login/google')
def google_login():
    """Initiate Google OAuth login."""
    try:
        print("Google login initiated")
        # Use configured redirect URI if available, otherwise generate dynamically
        if current_app.config.get('GOOGLE_REDIRECT_URI'):
            redirect_uri = current_app.config['GOOGLE_REDIRECT_URI']
            print(f"Using configured redirect URI: {redirect_uri}")
        else:
            redirect_uri = url_for('auth.google_callback', _external=True)
            print(f"Using dynamic redirect URI: {redirect_uri}")
        return oauth.google.authorize_redirect(redirect_uri)
    except Exception as e:
        print(f"Google login error: {str(e)}")
        return redirect(url_for('auth.login') + f'?error=google_login_failed&details={str(e)}')

@auth_bp.route('/login/guest')
def guest_login():
    """Enter guest mode."""
    session['guest_id'] = secrets.token_urlsafe(16)
    session['is_guest'] = True
    
    # Ensure guest data folder exists
    guest_folder = get_guest_data_folder()
    os.makedirs(guest_folder, exist_ok=True)
    
    return redirect(url_for('dashboard'))

@auth_bp.route('/callback/discord')
def discord_callback():
    """Handle Discord OAuth callback."""
    # Check if there's an error in the callback
    if 'error' in request.args:
        error = request.args.get('error')
        error_description = request.args.get('error_description', 'No description')
        return f"Discord OAuth error: {error} - {error_description}", 400
    
    try:
        token = oauth.discord.authorize_access_token()
        
        # Manually fetch user info from Discord API
        headers = {
            'Authorization': f"Bearer {token['access_token']}"
        }
        
        resp = requests.get('https://discord.com/api/users/@me', headers=headers)
        resp.raise_for_status()
        user_info = resp.json()
        
        # Find or create user
        user = User.query.filter_by(provider='discord', provider_id=str(user_info['id'])).first()
        
        if not user:
            user = User(
                email=user_info.get('email', f"discord_{user_info['id']}@discord.local"),
                name=user_info.get('username', 'Discord User'),
                avatar_url=f"https://cdn.discordapp.com/avatars/{user_info['id']}/{user_info.get('avatar', '')}.png" if user_info.get('avatar') else None,
                provider='discord',
                provider_id=str(user_info['id'])
            )
            db.session.add(user)
        else:
            # Update last login
            user.last_login = datetime.utcnow()
            # Update profile info
            user.name = user_info.get('username', user.name)
            user.avatar_url = f"https://cdn.discordapp.com/avatars/{user_info['id']}/{user_info.get('avatar', '')}.png" if user_info.get('avatar') else user.avatar_url
        
        db.session.commit()
        
        # Ensure user data folder exists
        user.ensure_data_folder()
        
        # Clear guest session
        session.pop('guest_id', None)
        session.pop('is_guest', None)
        
        # Store Discord token in session for future use
        session['discord_token'] = token['access_token']
        
        # Log in user
        login_user(user, remember=True)
        
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        current_app.logger.error(f"Discord OAuth error: {str(e)}")
        return redirect(url_for('dashboard') + '?error=discord_auth_failed')

@auth_bp.route('/callback/google')
def google_callback():
    """Handle Google OAuth callback."""
    try:
        print("Google callback received")
        print(f"Request args: {request.args}")
        
        # Check for OAuth errors
        if 'error' in request.args:
            error = request.args.get('error')
            error_description = request.args.get('error_description', 'No description')
            print(f"Google OAuth error: {error} - {error_description}")
            return redirect(url_for('auth.login') + f'?error=google_oauth_error&details={error}')
        
        token = oauth.google.authorize_access_token()
        print(f"Token received: {token.keys() if token else 'None'}")
        
        # Get user info from the token
        user_info = token.get('userinfo')
        if not user_info:
            print("No userinfo in token, trying to fetch manually")
            # Try to fetch user info manually using the OAuth client
            try:
                resp = oauth.google.get('userinfo')
                user_info = resp.json()
                print(f"Manual userinfo fetch: {user_info}")
            except Exception as e:
                print(f"Failed to fetch userinfo manually: {e}")
                # Try direct API call as fallback
                try:
                    headers = {'Authorization': f"Bearer {token['access_token']}"}
                    resp = requests.get('https://www.googleapis.com/oauth2/v3/userinfo', headers=headers)
                    user_info = resp.json()
                    print(f"Direct API userinfo fetch: {user_info}")
                except Exception as e2:
                    print(f"Failed direct API call: {e2}")
                    return redirect(url_for('auth.login') + '?error=google_auth_failed')
        
        if not user_info:
            print("Failed to get user info")
            return redirect(url_for('auth.login') + '?error=google_auth_failed')
        
        # Find or create user
        user = User.query.filter_by(provider='google', provider_id=str(user_info['sub'])).first()
        
        if not user:
            user = User(
                email=user_info.get('email', f"google_{user_info['sub']}@google.local"),
                name=user_info.get('name', 'Google User'),
                avatar_url=user_info.get('picture'),
                provider='google',
                provider_id=str(user_info['sub'])
            )
            db.session.add(user)
        else:
            # Update last login
            user.last_login = datetime.utcnow()
            # Update profile info
            user.name = user_info.get('name', user.name)
            user.avatar_url = user_info.get('picture', user.avatar_url)
        
        db.session.commit()
        
        # Ensure user data folder exists
        user.ensure_data_folder()
        
        # Clear guest session
        session.pop('guest_id', None)
        session.pop('is_guest', None)
        
        # Log in user
        login_user(user, remember=True)
        
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        current_app.logger.error(f"Google OAuth error: {str(e)}")
        return redirect(url_for('auth.login') + '?error=google_auth_failed')

@auth_bp.route('/logout')
def logout():
    """Log out current user."""
    logout_user()
    session.pop('guest_id', None)
    session.pop('is_guest', None)
    session.pop('discord_token', None)  # Clear Discord token
    
    # Check if there's a redirect parameter
    redirect_to = request.args.get('redirect')
    if redirect_to:
        return redirect(redirect_to)
    
    return redirect(url_for('dashboard'))

@auth_bp.route('/user')
def user_info():
    """Get current user information."""
    if current_user.is_authenticated:
        return jsonify({
            'is_authenticated': True,
            'is_guest': False,
            'user': {
                'id': current_user.id,
                'name': current_user.name,
                'email': current_user.email,
                'avatar_url': current_user.avatar_url,
                'provider': current_user.provider,
                'data_folder': current_user.get_data_folder(),
                'is_admin': current_user.is_admin()
            }
        })
    elif session.get('is_guest'):
        return jsonify({
            'is_authenticated': False,
            'is_guest': True,
            'user': {
                'id': session.get('guest_id'),
                'name': f"Guest {session.get('guest_id', '')[:8]}",
                'email': None,
                'avatar_url': None,
                'provider': 'guest',
                'data_folder': get_guest_data_folder()
            }
        })
    else:
        return jsonify({
            'is_authenticated': False,
            'is_guest': False,
            'user': None
        })