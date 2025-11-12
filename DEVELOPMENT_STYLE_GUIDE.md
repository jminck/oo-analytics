# Development Style Guide

This document defines the established patterns and conventions used throughout the application. **Always follow these patterns when adding new features or modifying existing code.**

## Table of Contents

1. [Backend Patterns](#backend-patterns)
2. [Frontend Patterns](#frontend-patterns)
3. [Chart Patterns](#chart-patterns)
4. [File Management Patterns](#file-management-patterns)
5. [Error Handling](#error-handling)
6. [Naming Conventions](#naming-conventions)

---

## Backend Patterns

### API Endpoints

#### Route Decorators
- Use `@app.route('/api/...')` for main app routes
- Use `@bp.route(...)` for blueprint routes
- Always specify HTTP methods: `methods=['GET']`, `methods=['POST']`, etc.

```python
@app.route('/api/portfolio/overview')
@guest_mode_required
def portfolio_overview():
    """Get portfolio overview metrics."""
    # Implementation
```

#### Authentication Decorators
- Use `@guest_mode_required` for endpoints that work with both authenticated and guest users
- Use `@login_required` for endpoints that require authentication only
- Always use `get_current_user_id()` to get user context

```python
@app.route('/api/upload', methods=['POST'])
@guest_mode_required
def upload_csv():
    user_id = get_current_user_id()
    # Implementation
```

#### Response Format
**ALWAYS** use this standard response format:

**Success Response:**
```python
return jsonify({
    'success': True,
    'data': {
        # Your data here
    }
})
```

**Error Response:**
```python
return jsonify({
    'success': False,
    'error': 'Error message here'
}), 400  # or appropriate status code
```

**With Cache Headers (for GET requests):**
```python
response = jsonify({
    'success': True,
    'data': {...}
})
return add_cache_busting_headers(response)
```

#### Error Handling Pattern
```python
@app.route('/api/endpoint')
@guest_mode_required
def my_endpoint():
    try:
        # Main logic here
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        print(f"ERROR in my_endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

#### User Context
- Always use `get_current_user_id()` to get the current user ID
- Use `get_current_data_folder()` to get user-specific data directory
- Never hardcode user paths

```python
user_id = get_current_user_id()
data_folder = get_current_data_folder()
```

#### Application Insights Tracking
- Track important events and exceptions in production:
```python
app_insights.track_event('event_name', {
    'user_id': get_current_user_id(),
    'additional_data': value
})

app_insights.track_exception(e, {
    'user_id': get_current_user_id(),
    'endpoint': 'endpoint_name'
})
```

### Data Models

#### Database Operations
- **ALWAYS** use `DatabaseManager` for database operations
- Never create direct SQLite connections outside of `DatabaseManager`
- Use connection context managers when possible

```python
from models import DatabaseManager

db_manager = DatabaseManager()
conn = db_manager.get_connection()
try:
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM trades WHERE user_id = ?', (user_id,))
    results = cursor.fetchall()
finally:
    conn.close()
```

#### Model Classes
- Follow the established structure: `Portfolio`, `Strategy`, `Trade`
- Use type hints where applicable: `def method(self) -> Dict:`
- Methods should return typed structures (Dict, List, etc.)

```python
class MyModel:
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'key': value
        }
```

#### Portfolio Loading
- Always check if portfolio has data before processing
- Use `portfolio.strategies` to check for strategies
- Handle empty portfolio cases gracefully

```python
if not portfolio or not portfolio.strategies:
    return jsonify({
        'success': False,
        'error': 'No portfolio data available'
    })
```

---

## Frontend Patterns

### Tables

#### HTML Structure
**ALWAYS** use this exact structure for sortable tables:

```html
<div class="table-responsive" style="max-height: 600px; overflow-x: auto; overflow-y: auto;">
    <table class="table table-hover" id="tableId">
        <thead class="table-dark">
            <tr>
                <th style="cursor:pointer;" 
                    data-key="column_key" 
                    onclick="sortTableName('column_key')" 
                    class="custom-tooltip sortable" 
                    data-tooltip="Tooltip text explaining this column">
                    Column Name <span id="sort-indicator-column_key"></span>
                </th>
                <!-- More columns -->
            </tr>
        </thead>
        <tbody>
            <!-- Table rows -->
        </tbody>
    </table>
</div>
```

#### Required Elements
- **Wrapper**: `<div class="table-responsive">` with optional max-height styling
- **Table**: `<table class="table table-hover" id="uniqueTableId">`
- **Header**: `<thead class="table-dark">`
- **Sortable Headers**: Must include:
  - `class="custom-tooltip sortable"`
  - `data-key="column_key"` (for sorting)
  - `data-tooltip="description"` (for tooltip)
  - `onclick="sortFunctionName('column_key')"`
  - Sort indicator: `<span id="sort-indicator-column_key"></span>`

#### Sorting Functions
**ALWAYS** follow this pattern:

```javascript
// State variables (at top of script section)
let tableNameSortKey = 'default_column';
let tableNameSortAsc = true;
let tableNameTableData = [];

// Sorting function
function sortTableName(key) {
    console.log('🔧 sortTableName called with key:', key);
    
    if (tableNameSortKey === key) {
        tableNameSortAsc = !tableNameSortAsc;
    } else {
        tableNameSortKey = key;
        tableNameSortAsc = key === 'text_column'; // true for text, false for numbers
    }
    
    // Re-render table
    setTimeout(() => {
        renderTableName();
    }, 10);
}

// Render function
function renderTableName() {
    // Sort data
    const sorted = [...tableNameTableData].sort((a, b) => {
        let valA, valB;
        // Handle different data types
        if (typeof a[tableNameSortKey] === 'string') {
            valA = a[tableNameSortKey].toLowerCase();
            valB = b[tableNameSortKey].toLowerCase();
        } else {
            valA = parseFloat(a[tableNameSortKey]) || 0;
            valB = parseFloat(b[tableNameSortKey]) || 0;
        }
        
        if (tableNameSortAsc) {
            return valA > valB ? 1 : -1;
        } else {
            return valA < valB ? 1 : -1;
        }
    });
    
    // Update indicators
    const keys = ['col1', 'col2', 'col3']; // All column keys
    keys.forEach(key => {
        const indicator = document.getElementById(`sort-indicator-${key}`);
        if (indicator) {
            if (key === tableNameSortKey) {
                indicator.textContent = tableNameSortAsc ? '▲' : '▼';
            } else {
                indicator.textContent = '';
            }
        }
    });
    
    // Render table body
    const tbody = document.querySelector('#tableId tbody');
    tbody.innerHTML = sorted.map(item => {
        return `<tr>
            <td>${item.col1}</td>
            <td>${item.col2}</td>
        </tr>`;
    }).join('');
    
    // Initialize tooltips after rendering
    setTimeout(() => {
        initializeCustomTooltips();
    }, 100);
}
```

#### Tooltip Initialization
- **ALWAYS** call `initializeCustomTooltips()` after rendering tables or updating DOM
- Use `setTimeout` with 100ms delay to ensure DOM is ready
- Tooltips use `data-tooltip` attribute on elements with `class="custom-tooltip"`

```javascript
setTimeout(() => {
    initializeCustomTooltips();
}, 100);
```

### Buttons

#### Bootstrap Classes
- Primary actions: `btn btn-primary`
- Secondary actions: `btn btn-outline-secondary`
- Info actions: `btn btn-outline-info`
- Danger actions: `btn btn-outline-danger` or `btn btn-danger`
- Small buttons: Add `btn-sm`

```html
<button class="btn btn-primary" onclick="doSomething()">
    <i class="fas fa-icon"></i> Button Text
</button>
```

#### Icons
- Use FontAwesome icons: `<i class="fas fa-icon-name"></i>`
- Place icons before text
- Common icons: `fa-upload`, `fa-download`, `fa-trash`, `fa-edit`, `fa-save`

### Cards

#### Structure
```html
<div class="card">
    <div class="card-header">
        <h5>Card Title</h5>
    </div>
    <div class="card-body">
        <!-- Card content -->
    </div>
</div>
```

#### Metric Cards
```html
<div class="card metric-card">
    <div class="card-body text-center">
        <h3>Value</h3>
        <p class="text-muted">Label</p>
    </div>
</div>
```

### Modals

#### Structure
```html
<div class="modal fade" id="modalId" tabindex="-1">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Modal Title</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
            </div>
            <div class="modal-body">
                <!-- Modal content -->
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-primary">Action</button>
                <button type="button" class="btn btn-outline-secondary" data-bs-dismiss="modal">Cancel</button>
            </div>
        </div>
    </div>
</div>
```

### Internationalization (i18n)

- **ALWAYS** use `data-i18n` attributes for user-facing text
- Add translations to `translations/en/frontend.json` and `translations/es/frontend.json`
- Use `data-i18n-title` for tooltips

```html
<button data-i18n="button_text">Default Text</button>
<span data-i18n="label_text">Default Label</span>
```

---

## Chart Patterns

### Chart Generation

#### Using ChartFactory
**ALWAYS** use `ChartFactory` to create charts:

```python
from charts import ChartFactory

chart_factory = ChartFactory(portfolio)
chart_data = chart_factory.create_chart('chart_type')
```

#### Chart Methods
- **ALWAYS** use `@cache_chart_result()` decorator for chart methods
- Return dict with Plotly JSON structure
- Use `_empty_chart(message)` for error cases

```python
@cache_chart_result()
def create_my_chart(self) -> Dict:
    """Create my custom chart."""
    try:
        # Chart generation logic
        fig = go.Figure()
        # ... configure figure ...
        return {
            'data': fig.to_dict()['data'],
            'layout': fig.to_dict()['layout']
        }
    except Exception as e:
        return self._empty_chart(f'Error: {str(e)}')
```

#### Empty Chart Pattern
```python
def _empty_chart(self, message: str = "No data available") -> Dict:
    """Return empty chart structure with message."""
    return {
        'data': [],
        'layout': {
            'title': {'text': message},
            'xaxis': {'visible': False},
            'yaxis': {'visible': False}
        }
    }
```

#### Color Palette
- Use predefined `self.colors` array in `ChartGenerator`
- Don't create custom color arrays
- Cycle through colors for multiple series

```python
color = self.colors[index % len(self.colors)]
```

### Chart Endpoint Pattern
```python
@app.route('/api/charts/<chart_type>')
def get_chart(chart_type):
    """Get a specific chart by type."""
    try:
        chart_factory = ChartFactory(portfolio)
        chart_data = chart_factory.create_chart(chart_type)
        
        if not isinstance(chart_data, dict):
            return jsonify({'error': 'Invalid chart data'}), 500
        
        response = make_response(jsonify(chart_data))
        return add_cache_busting_headers(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

---

## File Management Patterns

### File Operations
- **ALWAYS** use `FileManager` class for file operations
- Never directly manipulate file paths
- Use user-specific folders via `get_current_data_folder()`

```python
from file_manager import FileManager

file_manager = FileManager()
file_info = file_manager.save_file(file, friendly_name)
file_path = file_manager.get_file_path(filename)
```

### User Isolation
- Always use `get_current_data_folder()` for user-specific paths
- Never hardcode user directories
- Support both authenticated and guest users

```python
data_folder = get_current_data_folder()
user_file_path = os.path.join(data_folder, filename)
```

### Metadata Tracking
- Track friendly names, upload dates, file sizes
- Store metadata in `file_metadata.json`
- Use `FileManager` methods for metadata operations

---

## Error Handling

### Backend Error Handling
```python
try:
    # Operation
    result = perform_operation()
    return jsonify({'success': True, 'data': result})
except SpecificException as e:
    print(f"ERROR in function_name: {e}")
    import traceback
    traceback.print_exc()
    return jsonify({
        'success': False,
        'error': f'User-friendly error message: {str(e)}'
    }), 400
except Exception as e:
    print(f"ERROR in function_name: {e}")
    import traceback
    traceback.print_exc()
    app_insights.track_exception(e, {
        'user_id': get_current_user_id(),
        'endpoint': 'endpoint_name'
    })
    return jsonify({
        'success': False,
        'error': str(e)
    }), 500
```

### Frontend Error Handling
```javascript
try {
    const response = await fetch('/api/endpoint');
    const data = await response.json();
    
    if (!data.success) {
        console.error('API Error:', data.error);
        showError(data.error);
        return;
    }
    
    // Process data.success === true
    processData(data.data);
} catch (error) {
    console.error('Request failed:', error);
    showError('Request failed. Please try again.');
}
```

### Error Display
- Show user-friendly error messages
- Log detailed errors to console for debugging
- Use consistent error styling (Bootstrap alerts)

```html
<div class="alert alert-danger" id="errorMessage" style="display:none;">
    <i class="fas fa-exclamation-triangle"></i>
    <span id="errorText"></span>
</div>
```

---

## Naming Conventions

### Python
- **Functions**: `snake_case` - `get_portfolio_overview()`
- **Classes**: `PascalCase` - `Portfolio`, `ChartGenerator`
- **Constants**: `UPPER_SNAKE_CASE` - `MAX_FILE_SIZE`
- **Variables**: `snake_case` - `user_id`, `file_path`

### JavaScript
- **Functions**: `camelCase` - `sortTableName()`, `renderChart()`
- **Variables**: `camelCase` - `tableData`, `sortKey`
- **Constants**: `UPPER_SNAKE_CASE` - `MAX_RETRIES`
- **Table IDs**: `camelCase` with descriptive names - `strategyTable`, `liveVsBtTable`

### HTML/CSS
- **IDs**: `camelCase` - `strategyTable`, `uploadFileBtn`
- **Classes**: `kebab-case` for custom classes - `custom-tooltip`, `metric-card`
- **Data Attributes**: `kebab-case` - `data-tooltip`, `data-i18n`

### Database
- **Tables**: `snake_case` - `trades`, `strategies`
- **Columns**: `snake_case` - `date_opened`, `user_id`

---

## Best Practices

### Code Organization
- Keep related functionality together
- Use helper functions to avoid code duplication
- Follow the single responsibility principle

### Performance
- Use `@cache_chart_result()` for expensive chart operations
- Add `add_cache_busting_headers()` to GET responses
- Optimize database queries with proper indexes

### Security
- Always validate user input
- Use parameterized queries for database operations
- Never expose sensitive data in error messages

### Documentation
- Add docstrings to all functions and classes
- Document complex logic with inline comments
- Update this style guide when new patterns emerge

---

## Quick Reference Checklist

When adding a new feature, verify:

- [ ] API endpoint uses standard response format (`success`, `data`, `error`)
- [ ] Error handling includes try/except with proper logging
- [ ] User context retrieved via `get_current_user_id()` and `get_current_data_folder()`
- [ ] Cache headers added to GET responses
- [ ] Table uses standard sortable structure with tooltips
- [ ] Sorting function follows established pattern
- [ ] Tooltips initialized after DOM updates
- [ ] Chart uses `ChartFactory` pattern with `@cache_chart_result()`
- [ ] File operations use `FileManager` class
- [ ] i18n attributes added to user-facing text
- [ ] Error messages are user-friendly
- [ ] Code follows naming conventions

---

**Remember**: Consistency is key. When in doubt, look at existing code that does something similar and follow that pattern.

