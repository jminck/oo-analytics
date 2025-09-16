# Local Development Setup

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables:**
   ```bash
   # Copy the example file
   cp env_example.txt .env
   
   # Edit .env with your OAuth credentials
   # (See OAuth Setup section below)
   ```

3. **Run the Application:**
   ```bash
   python app.py
   ```

4. **Access the App:**
   - Open http://localhost:5000 in your browser

## OAuth Setup for Local Development

### Google OAuth Setup:

1. **Get Credentials:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one
   - Enable Google+ API
   - Go to **APIs & Services** > **Credentials**
   - Click **Create Credentials** > **OAuth 2.0 Client ID**
   - Choose **Web application**

2. **Configure Redirect URIs:**
   - Add these to **Authorized redirect URIs**:
     ```
     http://localhost:5000/auth/callback/google
     http://127.0.0.1:5000/auth/callback/google
     ```

3. **Update .env file:**
   ```
   GOOGLE_CLIENT_ID=your_actual_client_id
   GOOGLE_CLIENT_SECRET=your_actual_client_secret
   ```

### Discord OAuth Setup:

1. **Get Credentials:**
   - Go to [Discord Developer Portal](https://discord.com/developers/applications)
   - Create a new application
   - Go to **OAuth2** > **General**

2. **Configure Redirect URIs:**
   - Add these to **Redirects**:
     ```
     http://localhost:5000/auth/callback/discord
     http://127.0.0.1:5000/auth/callback/discord
     ```

3. **Update .env file:**
   ```
   DISCORD_CLIENT_ID=your_actual_client_id
   DISCORD_CLIENT_SECRET=your_actual_client_secret
   ```

## Environment Variables

Create a `.env` file in the project root with:

```env
# Flask Configuration
SECRET_KEY=your-dev-secret-key-change-in-production

# Google OAuth
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here

# Discord OAuth
DISCORD_CLIENT_ID=your_discord_client_id_here
DISCORD_CLIENT_SECRET=your_discord_client_secret_here
```

## How It Works Locally

- **No redirect URI needed**: The app automatically generates `http://localhost:5000/auth/callback/google` and `http://localhost:5000/auth/callback/discord`
- **SQLite database**: Uses `instance/portfolio_auth.db` for user authentication
- **Guest mode**: Works without OAuth setup for testing
- **File storage**: Uses `data/` directory for uploaded files

## Testing Without OAuth

If you don't want to set up OAuth for local development:

1. **Use Guest Mode**: Click "Continue as Guest" on the login page
2. **Upload CSV files**: Test all functionality without authentication
3. **No OAuth required**: All features work in guest mode

## Troubleshooting

- **Module not found**: Run `pip install -r requirements.txt`
- **OAuth errors**: Check that redirect URIs match exactly in OAuth console
- **Database errors**: Delete `instance/portfolio_auth.db` to reset
- **Port already in use**: Change port in `app.py` or kill existing process

## Development vs Production

- **Local**: Uses dynamic redirect URIs and SQLite
- **Production**: Uses configured redirect URIs and can use Azure SQL Database
- **Environment variables**: Different sets for local vs Azure deployment
