# Azure App Service Deployment Guide

This guide will help you deploy the Portfolio Analysis application to Azure App Service using best practices.

## Prerequisites

1. **Azure Account**: You need an active Azure subscription
2. **Azure CLI**: Install Azure CLI from https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
3. **Git**: Install Git from https://git-scm.com/downloads

## Step 1: Prepare Your Application

The application is already configured for Azure deployment with the following files:
- `requirements.txt` - Python dependencies (including Gunicorn)
- `gunicorn.conf.py` - Gunicorn configuration for production
- `app.py` - Flask application with WSGI entry point (`application = app`)
- No custom deployment scripts needed (uses Azure's built-in Python build)

## Step 2: Create Azure App Service

### Option A: Using Azure Portal

1. Go to [Azure Portal](https://portal.azure.com)
2. Click "Create a resource"
3. Search for "App Service" and select it
4. Click "Create"
5. Fill in the details:
   - **Resource Group**: Create new or use existing
   - **Name**: Choose a unique name (e.g., `portfolio-analysis-app`)
   - **Publish**: Code
   - **Runtime stack**: Python 3.9 (or latest available)
   - **Operating System**: Linux
   - **Region**: Choose closest to your users
   - **App Service Plan**: Create new or use existing
6. Click "Review + create" then "Create"

### Option B: Using Azure CLI

```bash
# Login to Azure
az login

# Create resource group
az group create --name portfolio-analysis-rg --location eastus

# Create App Service plan
az appservice plan create --name portfolio-analysis-plan --resource-group portfolio-analysis-rg --sku B1 --is-linux

# Create web app
az webapp create --resource-group portfolio-analysis-rg --plan portfolio-analysis-plan --name portfolio-analysis-app --runtime "PYTHON|3.9"
```

## Step 3: Configure Application Settings

### In Azure Portal:

1. Go to your App Service
2. Navigate to "Configuration" → "Application settings"
3. Add the following settings:

```
WEBSITES_PORT = 8000
SCM_DO_BUILD_DURING_DEPLOYMENT = true
ENABLE_ORYX_BUILD = true
STARTUP_COMMAND = gunicorn --config gunicorn.conf.py app:application
```

### Using Azure CLI:

```bash
az webapp config appsettings set --resource-group portfolio-analysis-rg --name portfolio-analysis-app --settings WEBSITES_PORT=8000 SCM_DO_BUILD_DURING_DEPLOYMENT=true ENABLE_ORYX_BUILD=true STARTUP_COMMAND="gunicorn --config gunicorn.conf.py app:application"
```

## Step 4: Deploy Your Application

### Option A: Deploy from Local Git

1. In Azure Portal, go to your App Service
2. Navigate to "Deployment Center"
3. Choose "Local Git/FTPS credentials"
4. Set up deployment credentials
5. Choose "Local Git" as source
6. Copy the Git URL

```bash
# In your project directory (simple-portfolio folder)
git init
git add .
git commit -m "Initial commit"
git remote add azure <your-git-url>
git push azure main
```

### Option B: Deploy using Azure CLI

```bash
# Navigate to your project directory
cd simple-portfolio

# Deploy using Azure CLI
az webapp deployment source config-local-git --name portfolio-analysis-app --resource-group portfolio-analysis-rg

# Get the Git URL
az webapp deployment list-publishing-credentials --name portfolio-analysis-app --resource-group portfolio-analysis-rg

# Deploy
git init
git add .
git commit -m "Initial commit"
git remote add azure <git-url-from-above>
git push azure main
```

### Option C: Deploy using GitHub Actions

1. Push your code to GitHub
2. In Azure Portal, go to "Deployment Center"
3. Choose "GitHub" as source
4. Connect your GitHub account
5. Select your repository and branch
6. Azure will automatically deploy when you push changes

## Step 5: Verify Deployment

1. Go to your App Service URL (e.g., `https://portfolio-analysis-app.azurewebsites.net`)
2. The application should load successfully
3. Check the logs in Azure Portal under "Log stream" if there are issues

## Step 6: Configure Custom Domain (Optional)

1. In Azure Portal, go to "Custom domains"
2. Add your domain
3. Configure DNS records as instructed

## Troubleshooting

### Common Issues:

1. **Application not starting**:
   - Check "Log stream" in Azure Portal
   - Verify `startup.txt` contains correct command
   - Ensure all dependencies are in `requirements.txt`

2. **Database issues**:
   - The app uses SQLite which is file-based
   - Data will be stored in the app's file system
   - Consider using Azure Database for production

3. **Port issues**:
   - Ensure `WEBSITES_PORT=8000` is set in application settings
   - Check that Gunicorn is binding to `0.0.0.0:8000`

4. **Python version issues**:
   - Set `PYTHON_VERSION=3.9` in application settings
   - Ensure your local Python version matches

### Viewing Logs:

```bash
# View real-time logs
az webapp log tail --name portfolio-analysis-app --resource-group portfolio-analysis-rg

# Download logs
az webapp log download --name portfolio-analysis-app --resource-group portfolio-analysis-rg
```

## Production Considerations

1. **Database**: Consider migrating from SQLite to Azure SQL Database or PostgreSQL
2. **File Storage**: Use Azure Blob Storage for file uploads
3. **Authentication**: Configure proper OAuth providers for production
4. **SSL**: Enable HTTPS (automatic with Azure App Service)
5. **Scaling**: Configure auto-scaling rules as needed
6. **Monitoring**: Set up Application Insights for monitoring

## Cost Optimization

- Start with Basic (B1) plan for testing
- Use Free tier for development
- Consider Consumption plan for low-traffic applications
- Monitor usage in Azure Portal

## Security

1. **Environment Variables**: Store sensitive data in Application Settings
2. **HTTPS**: Always use HTTPS in production
3. **Authentication**: Implement proper user authentication
4. **Input Validation**: Validate all user inputs
5. **Regular Updates**: Keep dependencies updated

## Support

- Azure Documentation: https://docs.microsoft.com/en-us/azure/app-service/
- Azure Support: Available through Azure Portal
- Community: Stack Overflow, Azure Forums 