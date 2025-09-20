# Admin Setup Guide

This guide explains how to set up admin access for the OO Analytics application.

## Admin Configuration

### Local Development

1. Create a `.env` file in the project root:
```bash
# Admin Configuration
ADMIN_EMAILS=your-email@gmail.com,another-admin@example.com
```

2. Restart the application to load the new configuration.

### Azure Deployment

1. Go to your Azure App Service in the Azure Portal
2. Navigate to **Configuration** > **Application settings**
3. Add a new setting:
   - **Name**: `ADMIN_EMAILS`
   - **Value**: `your-email@gmail.com,another-admin@example.com`
4. Click **Save** and restart the app

## Admin Features

Once configured, admin users will have access to:

- **Admin Panel**: Accessible via the user dropdown menu (only visible to admins)
- **User Management**: View all registered users and their activity
- **System Statistics**: Monitor storage usage, user counts, and database size
- **Data Cleanup**: Clean up old guest data directories
- **System Monitoring**: View application logs and system health

## Admin Panel URL

- **Local**: `http://localhost:5000/admin/`
- **Azure**: `https://your-app.azurewebsites.net/admin/`

## Security Notes

- Admin access is based on email addresses in the `ADMIN_EMAILS` environment variable
- Only authenticated users with admin emails can access the admin panel
- Admin status is checked on every request
- The admin panel is completely separate from the main application

## Adding/Removing Admins

To add or remove admin users:

1. **Local**: Update the `ADMIN_EMAILS` in your `.env` file
2. **Azure**: Update the `ADMIN_EMAILS` application setting in Azure Portal
3. Restart the application

## Admin Email Format

Admin emails should be:
- The same email address used for OAuth login (Google/Discord)
- Comma-separated for multiple admins
- Case-insensitive (automatically converted to lowercase)

Example:
```
ADMIN_EMAILS=john.doe@gmail.com,jane.smith@company.com,admin@example.com
```
