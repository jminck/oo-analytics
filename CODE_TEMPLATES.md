# Code Templates

Copy-paste ready code snippets for common operations. These templates follow the established patterns documented in `DEVELOPMENT_STYLE_GUIDE.md`.

## Table of Contents

1. [API Endpoints](#api-endpoints)
2. [Frontend Tables](#frontend-tables)
3. [Chart Methods](#chart-methods)
4. [Data Models](#data-models)
5. [UI Elements](#ui-elements)

---

## API Endpoints

### Basic GET Endpoint

```python
@app.route('/api/your-endpoint')
@guest_mode_required
def your_endpoint():
    """Description of what this endpoint does."""
    try:
        # Get user context
        user_id = get_current_user_id()
        data_folder = get_current_data_folder()
        
        # Your logic here
        result = perform_operation()
        
        # Return success response
        response = jsonify({
            'success': True,
            'data': result
        })
        return add_cache_busting_headers(response)
        
    except Exception as e:
        print(f"ERROR in your_endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

### POST Endpoint with File Upload

```python
@app.route('/api/upload-endpoint', methods=['POST'])
@guest_mode_required
def upload_endpoint():
    """Upload and process file."""
    try:
        user_id = get_current_user_id()
        
        # Track event
        app_insights.track_event('upload_attempted', {
            'user_id': user_id,
            'has_file': 'file' in request.files
        })
        
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Process file
        file_manager = FileManager()
        file_info = file_manager.save_file(file)
        
        # Your processing logic here
        result = process_file(file_info)
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        app_insights.track_exception(e, {
            'user_id': get_current_user_id(),
            'endpoint': 'upload_endpoint'
        })
        print(f"ERROR in upload_endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

### POST Endpoint with JSON Data

```python
@app.route('/api/process-data', methods=['POST'])
@guest_mode_required
def process_data():
    """Process JSON data."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['field1', 'field2']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Process data
        result = perform_processing(data)
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        print(f"ERROR in process_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

---

## Frontend Tables

### Complete Sortable Table HTML

```html
<div class="table-responsive" style="max-height: 600px; overflow-x: auto; overflow-y: auto;">
    <table class="table table-hover" id="myTable">
        <thead class="table-dark">
            <tr>
                <th style="cursor:pointer;" 
                    data-key="column1" 
                    onclick="sortMyTable('column1')" 
                    class="custom-tooltip sortable" 
                    data-tooltip="Description of column 1">
                    Column 1 <span id="sort-indicator-column1"></span>
                </th>
                <th style="cursor:pointer;" 
                    data-key="column2" 
                    onclick="sortMyTable('column2')" 
                    class="custom-tooltip sortable" 
                    data-tooltip="Description of column 2">
                    Column 2 <span id="sort-indicator-column2"></span>
                </th>
                <!-- Add more columns as needed -->
            </tr>
        </thead>
        <tbody>
            <!-- Table rows will be populated by JavaScript -->
        </tbody>
    </table>
</div>
```

### Complete Table Sorting JavaScript

```javascript
// State variables (declare at top of script section)
let myTableSortKey = 'column1';
let myTableSortAsc = true;
let myTableData = [];

// Sorting function
function sortMyTable(key) {
    console.log('🔧 sortMyTable called with key:', key);
    console.log('🔧 Current sortKey:', myTableSortKey);
    console.log('🔧 Current sortAsc:', myTableSortAsc);
    
    if (myTableSortKey === key) {
        myTableSortAsc = !myTableSortAsc;
        console.log('🔧 Toggling sort direction');
    } else {
        myTableSortKey = key;
        // Default: true for text columns, false for numbers
        myTableSortAsc = key === 'column1'; // Adjust based on your columns
        console.log('🔧 Switching to new column:', key);
    }
    
    console.log('🔧 New sortKey:', myTableSortKey);
    console.log('🔧 New sortAsc:', myTableSortAsc);
    
    // Re-render table
    setTimeout(() => {
        renderMyTable();
    }, 10);
}

// Render function
function renderMyTable() {
    console.log('🔧 renderMyTable called');
    console.log('🔧 Data length:', myTableData ? myTableData.length : 'undefined');
    
    if (!myTableData || myTableData.length === 0) {
        console.log('🔧 No data to render');
        const tbody = document.querySelector('#myTable tbody');
        if (tbody) {
            tbody.innerHTML = '<tr><td colspan="2" class="text-center text-muted">No data available</td></tr>';
        }
        return;
    }
    
    const table = document.getElementById('myTable');
    if (!table) {
        console.log('🔧 Table not found');
        return;
    }
    
    const tbody = table.querySelector('tbody');
    if (!tbody) {
        console.log('🔧 tbody not found');
        return;
    }
    
    console.log('🔧 Sorting by:', myTableSortKey, 'Ascending:', myTableSortAsc);
    
    // Sort data
    const sorted = [...myTableData].sort((a, b) => {
        let valA, valB;
        
        // Handle different data types
        switch (myTableSortKey) {
            case 'column1': // Text column
                valA = (a.column1 || '').toString().toLowerCase();
                valB = (b.column1 || '').toString().toLowerCase();
                break;
            case 'column2': // Numeric column
                valA = parseFloat(a.column2) || 0;
                valB = parseFloat(b.column2) || 0;
                break;
            default:
                valA = parseFloat(a[myTableSortKey]) || 0;
                valB = parseFloat(b[myTableSortKey]) || 0;
        }
        
        if (myTableSortAsc) {
            return valA > valB ? 1 : -1;
        } else {
            return valA < valB ? 1 : -1;
        }
    });
    
    // Update sort indicators
    const keys = ['column1', 'column2']; // All column keys
    keys.forEach(key => {
        const indicator = document.getElementById(`sort-indicator-${key}`);
        if (indicator) {
            if (key === myTableSortKey) {
                indicator.textContent = myTableSortAsc ? '▲' : '▼';
            } else {
                indicator.textContent = '';
            }
        }
    });
    
    // Render table rows
    tbody.innerHTML = sorted.map(item => {
        return `<tr>
            <td>${escapeHtml(item.column1 || '')}</td>
            <td>${formatNumber(item.column2)}</td>
        </tr>`;
    }).join('');
    
    // Initialize tooltips after rendering
    setTimeout(() => {
        initializeCustomTooltips();
    }, 100);
}

// Helper function to escape HTML (prevent XSS)
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Helper function to format numbers
function formatNumber(value) {
    if (typeof value === 'number') {
        return value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }
    return value || '';
}
```

### Loading Table Data from API

```javascript
async function loadMyTableData() {
    try {
        const response = await fetch('/api/my-endpoint');
        const result = await response.json();
        
        if (!result.success) {
            console.error('API Error:', result.error);
            showError(result.error);
            return;
        }
        
        // Store data
        myTableData = result.data || [];
        
        // Render table
        renderMyTable();
        
    } catch (error) {
        console.error('Request failed:', error);
        showError('Failed to load data. Please try again.');
    }
}
```

---

## Chart Methods

### Chart Method Template

```python
@cache_chart_result()
def create_my_chart(self) -> Dict:
    """Create my custom chart with caching."""
    try:
        # Check if portfolio has data
        if not self.portfolio or not self.portfolio.strategies:
            return self._empty_chart("No portfolio data available")
        
        # Get data
        data = self._get_chart_data()
        
        if not data:
            return self._empty_chart("No data available")
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=data['x'],
            y=data['y'],
            mode='lines',
            name='My Series',
            line=dict(color=self.colors[0])
        ))
        
        # Update layout
        fig.update_layout(
            title='My Chart Title',
            xaxis_title='X Axis Label',
            yaxis_title='Y Axis Label',
            hovermode='closest',
            template='plotly_dark'  # or 'plotly' for light theme
        )
        
        # Return chart data
        return {
            'data': fig.to_dict()['data'],
            'layout': fig.to_dict()['layout']
        }
        
    except Exception as e:
        print(f"Error creating my_chart: {e}")
        import traceback
        traceback.print_exc()
        return self._empty_chart(f'Error creating chart: {str(e)}')
```

### Chart Endpoint Integration

```python
# In app.py, add to chart_methods dictionary in get_chart() function:
'my_chart': self.generator.create_my_chart,

# In ChartFactory.get_available_charts():
'my_chart': 'Description of my chart',
```

---

## Data Models

### New Model Class Template

```python
class MyModel:
    """Description of what this model represents."""
    
    def __init__(self, name: str):
        self.name = name
        self.items: List[Dict] = []
    
    def add_item(self, item: Dict):
        """Add an item to the model."""
        self.items.append(item)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'name': self.name,
            'count': len(self.items),
            'total': sum(item.get('value', 0) for item in self.items)
        }
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        return {
            'name': self.name,
            'items': self.items,
            'summary': self.get_summary()
        }
```

### Database Query Template

```python
def get_my_data(user_id: str) -> List[Dict]:
    """Get data from database."""
    db_manager = DatabaseManager()
    conn = db_manager.get_connection()
    
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT column1, column2, column3
            FROM my_table
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'column1': row[0],
                'column2': row[1],
                'column3': row[2]
            })
        
        return results
        
    finally:
        conn.close()
```

---

## UI Elements

### Button Template

```html
<!-- Primary button -->
<button class="btn btn-primary" onclick="doAction()">
    <i class="fas fa-icon-name"></i> Button Text
</button>

<!-- Secondary button -->
<button class="btn btn-outline-secondary" onclick="cancelAction()">
    <i class="fas fa-times"></i> Cancel
</button>

<!-- Small button -->
<button class="btn btn-primary btn-sm" onclick="smallAction()">
    <i class="fas fa-icon"></i> Small
</button>

<!-- Button with i18n -->
<button class="btn btn-primary" data-i18n="button_text" onclick="doAction()">
    Default Text
</button>
```

### Card Template

```html
<!-- Standard card -->
<div class="card">
    <div class="card-header">
        <h5>Card Title</h5>
    </div>
    <div class="card-body">
        <!-- Card content -->
    </div>
</div>

<!-- Metric card -->
<div class="card metric-card">
    <div class="card-body text-center">
        <h3 id="metricValue">0</h3>
        <p class="text-muted" data-i18n="metric_label">Metric Label</p>
    </div>
</div>
```

### Modal Template

```html
<div class="modal fade" id="myModal" tabindex="-1" aria-labelledby="myModalLabel" aria-hidden="true">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" id="myModalLabel">Modal Title</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body">
                <!-- Modal content -->
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-primary" onclick="confirmAction()">
                    <i class="fas fa-check"></i> Confirm
                </button>
                <button type="button" class="btn btn-outline-secondary" data-bs-dismiss="modal">
                    <i class="fas fa-times"></i> Cancel
                </button>
            </div>
        </div>
    </div>
</div>
```

### Error Display Template

```html
<!-- Error alert -->
<div class="alert alert-danger" id="errorAlert" style="display:none;">
    <i class="fas fa-exclamation-triangle"></i>
    <span id="errorMessage"></span>
</div>
```

```javascript
function showError(message) {
    const alert = document.getElementById('errorAlert');
    const messageSpan = document.getElementById('errorMessage');
    if (alert && messageSpan) {
        messageSpan.textContent = message;
        alert.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            alert.style.display = 'none';
        }, 5000);
    }
}
```

### Success Message Template

```html
<div class="alert alert-success" id="successAlert" style="display:none;">
    <i class="fas fa-check-circle"></i>
    <span id="successMessage"></span>
</div>
```

```javascript
function showSuccess(message) {
    const alert = document.getElementById('successAlert');
    const messageSpan = document.getElementById('successMessage');
    if (alert && messageSpan) {
        messageSpan.textContent = message;
        alert.style.display = 'block';
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            alert.style.display = 'none';
        }, 3000);
    }
}
```

---

## Quick Reference

### Common FontAwesome Icons
- `fa-upload` - Upload
- `fa-download` - Download
- `fa-trash` - Delete
- `fa-edit` - Edit
- `fa-save` - Save
- `fa-check` - Check/Confirm
- `fa-times` - Cancel/Close
- `fa-info-circle` - Information
- `fa-exclamation-triangle` - Warning
- `fa-table` - Table
- `fa-chart-line` - Chart

### Common Bootstrap Button Classes
- `btn btn-primary` - Primary action
- `btn btn-outline-primary` - Secondary primary action
- `btn btn-outline-secondary` - Secondary action
- `btn btn-outline-danger` - Dangerous action
- `btn btn-outline-info` - Informational action
- `btn btn-sm` - Small button
- `btn btn-lg` - Large button

---

**Remember**: Always refer to `DEVELOPMENT_STYLE_GUIDE.md` for detailed pattern explanations and best practices.

