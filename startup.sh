#!/bin/bash
echo "=== Starting Azure App Service startup script startup.sh ==="
echo "Current working directory: $(pwd)"
echo "Files in current directory: $(ls -la)"

echo "Creating necessary directories..."
mkdir -p /home/site/wwwroot/data/guest \
         /home/site/wwwroot/data/users \
         /home/site/wwwroot/instance \
         /home/site/wwwroot/logs \
         /home/site/wwwroot/backups \
         /home/site/wwwroot/db_templates

echo "Set proper permissions"
chmod 755 /home/site/wwwroot/data \
          /home/site/wwwroot/instance \
          /home/site/wwwroot/logs \
          /home/site/wwwroot/backups \
          /home/site/wwwroot/db_templates

echo "Checking auth database..."
if [ ! -f "/home/site/wwwroot/instance/portfolio_auth.db" ]; then
    echo "No auth database found - will be created on first run"
else
    echo "Auth database exists - preserving"
fi

echo "Checking Redis..."
if [ -n "$REDIS_URL" ]; then
    # External Redis configured — verify the connection
    python3 -c "
import os, redis, sys
try:
    r = redis.from_url(os.environ['REDIS_URL'], decode_responses=False, socket_connect_timeout=5)
    r.ping()
    print('Redis connection successful')
except Exception as e:
    print(f'Redis connection failed: {e}. Using in-memory cache.')
    sys.exit(0)
"
else
    # No external Redis configured — install and start a local Redis server inside the container.
    # The cache is ephemeral (cleared on restarts) which is fine for a caching layer.
    echo "REDIS_URL not set - starting local Redis server..."
    if ! command -v redis-server &>/dev/null; then
        echo "Installing Redis server..."
        apt-get update -y -qq && apt-get install -y --no-install-recommends redis-server 2>&1 | tail -3
    fi
    redis-server --daemonize yes --bind 127.0.0.1 --port 6379 --protected-mode yes --loglevel warning
    # Wait up to 5 seconds for Redis to be ready
    REDIS_READY=false
    for i in 1 2 3 4 5; do
        if redis-cli ping 2>/dev/null | grep -q PONG; then
            REDIS_READY=true
            break
        fi
        sleep 1
    done
    if [ "$REDIS_READY" = true ]; then
        echo "Local Redis server started successfully"
        export REDIS_URL=redis://localhost:6379/0
    else
        echo "Warning: local Redis failed to start - app will use in-memory cache"
    fi
fi

echo "Starting application with gunicorn..."
exec gunicorn --bind 0.0.0.0:8000 --workers 1 app:application