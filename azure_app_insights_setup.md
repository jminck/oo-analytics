# Azure Application Insights Setup

## 1. Create Application Insights Resource

1. Go to Azure Portal
2. Create a new Application Insights resource
3. Choose your subscription and resource group
4. Name it (e.g., "oo-analytics-insights")
5. Choose "General" as the application type
6. Choose your region

## 2. Get the Instrumentation Key

1. Go to your Application Insights resource
2. In the Overview section, copy the "Instrumentation Key"
3. It will look like: `12345678-1234-1234-1234-123456789abc`

## 3. Configure Azure App Service

### Option A: Azure Portal
1. Go to your App Service
2. Go to "Configuration" → "Application settings"
3. Add a new setting:
   - **Name**: `APPINSIGHTS_INSTRUMENTATIONKEY`
   - **Value**: Your instrumentation key from step 2

### Option B: Azure CLI
```bash
az webapp config appsettings set --name your-app-name --resource-group your-resource-group --settings APPINSIGHTS_INSTRUMENTATIONKEY="your-instrumentation-key"
```

## 4. Local Development

For local development, Application Insights will be automatically disabled if:
- No `APPINSIGHTS_INSTRUMENTATIONKEY` environment variable is set
- The opencensus packages are not installed

To test locally with Application Insights:
1. Create a `.env` file in your project root
2. Add: `APPINSIGHTS_INSTRUMENTATIONKEY=your-instrumentation-key`
3. Install the packages: `pip install -r requirements.txt`

## 5. What Gets Tracked

The application will automatically track:
- **HTTP Requests**: All Flask routes with timing and status codes
- **Exceptions**: Any unhandled exceptions
- **Custom Events**: File uploads, user actions
- **Custom Metrics**: Performance metrics
- **Dependencies**: Database calls, external API calls

## 6. Viewing Telemetry

1. Go to your Application Insights resource in Azure Portal
2. Use the "Application Map" to see request flows
3. Use "Live Metrics" for real-time monitoring
4. Use "Logs" to query custom events and metrics

## 7. Custom Queries

Example KQL queries for your analytics app:

```kql
// File upload events
customEvents
| where name == "file_upload_success"
| project timestamp, customDimensions.trades_count, customDimensions.strategies_count

// Exception tracking
exceptions
| where outerMessage contains "Upload failed"
| project timestamp, outerMessage, customDimensions.user_id

// Performance metrics
requests
| where name contains "/api/"
| summarize avg(duration) by name
| order by avg_duration desc
```

## 8. Cost Management

- Application Insights has a free tier (5GB/month)
- Monitor usage in the "Usage and estimated costs" section
- Set up alerts for high usage
- Consider sampling for high-traffic applications
