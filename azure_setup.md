# Azure App Service OAuth Setup

## Environment Variables to Set in Azure

You need to set these environment variables in your Azure App Service configuration:

### Required OAuth Environment Variables:

1. **Google OAuth:**
   ```
   GOOGLE_CLIENT_ID=your_google_client_id
   GOOGLE_CLIENT_SECRET=your_google_client_secret
   GOOGLE_REDIRECT_URI=https://oo-analytics.azurewebsites.net/auth/callback/google
   ```

2. **Discord OAuth (if using):**
   ```
   DISCORD_CLIENT_ID=your_discord_client_id
   DISCORD_CLIENT_SECRET=your_discord_client_secret
   DISCORD_REDIRECT_URI=https://oo-analytics.azurewebsites.net/auth/callback/discord
   ```

3. **Flask Secret Key:**
   ```
   SECRET_KEY=your_very_secure_secret_key_here
   ```

## How to Set Environment Variables in Azure:

1. Go to your Azure App Service in the Azure Portal
2. Navigate to **Settings** > **Configuration**
3. Click on **Application settings** tab
4. Click **+ New application setting** for each variable above
5. Set the **Name** and **Value** for each environment variable
6. Click **Save** to apply the changes
7. Restart your app service

## Google OAuth Console Setup:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **APIs & Services** > **Credentials**
3. Find your OAuth 2.0 Client ID
4. Click **Edit**
5. In **Authorized redirect URIs**, make sure you have:
   ```
   https://oo-analytics.azurewebsites.net/auth/callback/google
   ```
6. Save the changes

## Discord OAuth Console Setup:

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Select your application
3. Navigate to **OAuth2** > **General**
4. In **Redirects**, make sure you have:
   ```
   https://oo-analytics.azurewebsites.net/auth/callback/discord
   ```
5. Save the changes

## Testing:

After setting up the environment variables and restarting your app:

1. Visit `https://oo-analytics.azurewebsites.net/`
2. Try logging in with Google
3. Try logging in with Discord
4. Check the Azure App Service logs if there are any issues

## Troubleshooting:

- **400 Bad Redirect URI**: Make sure the `GOOGLE_REDIRECT_URI` and `DISCORD_REDIRECT_URI` environment variables exactly match what's configured in their respective OAuth consoles
- **Environment variables not working**: Make sure you restarted the App Service after setting the variables
- **Still getting localhost errors**: Check that you're not using any hardcoded localhost URLs in your code
- **Discord OAuth not working**: Make sure you've added the Discord redirect URI to your Discord application settings
