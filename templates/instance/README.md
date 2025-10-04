# Database Templates

This directory contains template database files for the Portfolio Strategy Analytics application.

## Setup Instructions

1. **Copy template files to instance directory:**
   ```bash
   cp templates/instance/portfolio_auth.db.template instance/portfolio_auth.db
   cp templates/instance/portfolio.db.template instance/portfolio.db
   ```

2. **Initialize the databases:**
   The application will automatically create the necessary tables and schema when it starts up.

## Important Notes

- The `instance/` directory is ignored by git to prevent committing live database files
- Template files are empty and will be populated by the application on first run
- Never commit the actual database files in `instance/` as they contain user data
