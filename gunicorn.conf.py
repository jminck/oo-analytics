# Gunicorn configuration file for Azure App Service
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
backlog = 2048

# Worker processes
# For Azure App Service, use 1 worker to avoid memory issues
workers = 1
worker_class = "sync"
worker_connections = 1000
timeout = 600
keepalive = 60

# Memory management
max_requests = 1000
max_requests_jitter = 100
worker_tmp_dir = "/dev/shm"  # Use RAM for temporary files

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "portfolio-app"

# Server mechanics
preload_app = True
daemon = False
tmp_upload_dir = None

# SSL
keyfile = None
certfile = None