"""
Azure Application Insights integration for the analytics app.
This module handles telemetry collection in production while being disabled locally.
"""

import os
import logging
from typing import Optional

# Only import Application Insights if the instrumentation key is available
try:
    from opencensus.ext.azure.log_exporter import AzureLogHandler
    from opencensus.ext.azure.trace_exporter import AzureExporter
    from opencensus.ext.flask.flask_middleware import FlaskMiddleware
    from opencensus.trace.samplers import ProbabilitySampler
    from opencensus.trace.tracer import Tracer
    from opencensus.trace import config_integration
    APP_INSIGHTS_AVAILABLE = True
except ImportError:
    APP_INSIGHTS_AVAILABLE = False
    # Create dummy classes for local development
    class AzureLogHandler:
        def __init__(self, *args, **kwargs):
            pass
    
    class AzureExporter:
        def __init__(self, *args, **kwargs):
            pass
    
    class FlaskMiddleware:
        def __init__(self, *args, **kwargs):
            pass
    
    class Tracer:
        def __init__(self, *args, **kwargs):
            pass

class ApplicationInsightsManager:
    """Manages Application Insights telemetry collection."""
    
    def __init__(self, app=None):
        self.app = app
        self.instrumentation_key = None
        self.is_enabled = False
        self.middleware = None
        self.logger = None
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize Application Insights with the Flask app."""
        self.app = app
        
        # Get instrumentation key from environment
        self.instrumentation_key = os.getenv('APPINSIGHTS_INSTRUMENTATIONKEY')
        
        # Only enable if we have a key and the packages are available
        if self.instrumentation_key and APP_INSIGHTS_AVAILABLE:
            self._setup_application_insights()
        else:
            # Log that Application Insights is disabled
            app.logger.info("Application Insights disabled - no instrumentation key or packages not available")
    
    def _setup_application_insights(self):
        """Set up Application Insights telemetry collection."""
        try:
            # Configure integrations
            config_integration.trace_integrations(['requests', 'logging'])
            
            # Set up Azure exporter
            exporter = AzureExporter(
                connection_string=f'InstrumentationKey={self.instrumentation_key}'
            )
            
            # Set up tracer with sampling
            tracer = Tracer(
                exporter=exporter,
                sampler=ProbabilitySampler(rate=1.0)  # Sample 100% of requests
            )
            
            # Set up Flask middleware
            self.middleware = FlaskMiddleware(
                self.app,
                exporter=exporter,
                sampler=ProbabilitySampler(rate=1.0)
            )
            
            # Set up logging handler
            handler = AzureLogHandler(
                connection_string=f'InstrumentationKey={self.instrumentation_key}'
            )
            handler.setLevel(logging.INFO)
            
            # Add handler to Flask app logger
            self.app.logger.addHandler(handler)
            self.app.logger.setLevel(logging.INFO)
            
            # Set up custom logger for application events
            self.logger = logging.getLogger('analytics_app')
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
            self.is_enabled = True
            self.app.logger.info("Application Insights enabled successfully")
            
        except Exception as e:
            self.app.logger.error(f"Failed to set up Application Insights: {e}")
            self.is_enabled = False
    
    def track_event(self, name: str, properties: Optional[dict] = None, measurements: Optional[dict] = None):
        """Track a custom event."""
        if self.is_enabled and self.logger:
            try:
                # Create a structured log message for Application Insights
                log_data = {
                    'event_name': name,
                    'properties': properties or {},
                    'measurements': measurements or {}
                }
                self.logger.info(f"Custom Event: {name}", extra=log_data)
            except Exception as e:
                self.app.logger.error(f"Failed to track event {name}: {e}")
    
    def track_exception(self, exception: Exception, properties: Optional[dict] = None):
        """Track an exception."""
        if self.is_enabled and self.logger:
            try:
                log_data = {
                    'exception_type': type(exception).__name__,
                    'exception_message': str(exception),
                    'properties': properties or {}
                }
                self.logger.error(f"Exception: {type(exception).__name__}: {str(exception)}", extra=log_data)
            except Exception as e:
                self.app.logger.error(f"Failed to track exception: {e}")
    
    def track_metric(self, name: str, value: float, properties: Optional[dict] = None):
        """Track a custom metric."""
        if self.is_enabled and self.logger:
            try:
                log_data = {
                    'metric_name': name,
                    'metric_value': value,
                    'properties': properties or {}
                }
                self.logger.info(f"Custom Metric: {name} = {value}", extra=log_data)
            except Exception as e:
                self.app.logger.error(f"Failed to track metric {name}: {e}")

# Global instance
app_insights = ApplicationInsightsManager()
