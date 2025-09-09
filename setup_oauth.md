# OAuth Setup Guide

This guide will help you set up Discord and Google OAuth for your Portfolio Strategy Analytics application.

## 1. Install Dependencies

First, install the new authentication dependencies:

```bash
pip install -r requirements.txt
```

## 2. Set Up Environment Variables

Create a `.env` file in the `simple-portfolio` directory with the following content:

```bash
# OAuth Configuration
DISCORD_CLIENT_ID=your_discord_client_id_here
DISCORD_CLIENT_SECRET=your_discord_client_secret_here

GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here

# Flask Configuration
SECRET_KEY=your_secret_key_here_generate_a_random_string
FLASK_ENV=development

# Database (optional - uses SQLite by default)
DATABASE_URL=sqlite:///portfolio_auth.db
```

## 3. Discord OAuth Setup

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application" and give it a name
3. Go to the "OAuth2" section
4. Copy the **Client ID** and **Client Secret**
5. Add redirect URI: `http://localhost:5000/auth/callback/discord`
6. In OAuth2 > URL Generator, select scopes: `identify` and `email`

## 4. Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Google+ API
4. Go to "Credentials" > "Create Credentials" > "OAuth 2.0 Client IDs"
5. Set application type to "Web application"
6. Add authorized redirect URI: `http://localhost:5000/auth/callback/google`
7. Copy the **Client ID** and **Client Secret**

## 5. Generate Secret Key

Generate a secure secret key for Flask:

```python
import secrets
print(secrets.token_urlsafe(32))
```

## 6. Update Your .env File

Replace the placeholder values in your `.env` file with the actual credentials:

```bash
DISCORD_CLIENT_ID=123456789012345678
DISCORD_CLIENT_SECRET=abcdef123456789
GOOGLE_CLIENT_ID=123456789-abcdef.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-abcdef123456789
SECRET_KEY=your_generated_secret_key_here
```

## 7. Features

### Authentication Options
- **Discord OAuth**: Sign in with Discord account
- **Google OAuth**: Sign in with Google account  
- **Guest Mode**: Use the app without signing in (temporary data)

### Data Partitioning
- **User Data**: Each authenticated user gets their own data folder
- **Guest Data**: Guest sessions get temporary folders
- **Isolation**: Users can only see their own uploaded files and data

### User Experience
- Smooth login/logout flow
- Profile information display
- Switch between authenticated and guest modes
- Persistent data for authenticated users
- Temporary data for guests

## 8. Directory Structure

The app creates the following directory structure:

```
data/
├── guest/           # Guest session data
│   ├── session1/    # Individual guest folders
│   └── session2/
└── users/           # Authenticated user data
    ├── 1/           # User ID folders
    ├── 2/
    └── ...
```

## 9. Running the Application

Once everything is set up:

```bash
cd simple-portfolio
python app.py
```

Visit `http://localhost:5000` and you'll see the new authentication options in the top-right dropdown menu.

## 10. Troubleshooting

- Make sure all OAuth redirect URIs match exactly
- Check that all environment variables are set correctly
- Verify that the `.env` file is in the correct directory
- Ensure Flask-Login and other dependencies are installed

## 11. Production Notes

For production deployment:
- Use HTTPS for all OAuth redirect URIs
- Set `FLASK_ENV=production`
- Use a more secure database (PostgreSQL recommended)
- Set proper CORS headers if needed
- Use environment variables or secure secret management