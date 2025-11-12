# Consistency Checklist

Use this checklist before submitting code to ensure consistency with established patterns. Check each item that applies to your changes.

## Pre-Implementation Checklist

Before starting implementation, verify:

- [ ] I have reviewed `DEVELOPMENT_STYLE_GUIDE.md` for relevant patterns
- [ ] I have checked `CODE_TEMPLATES.md` for applicable templates
- [ ] I understand the existing codebase patterns for similar features
- [ ] I have identified which patterns apply to my changes

---

## API Endpoints

If you're adding or modifying an API endpoint:

- [ ] Route uses `@app.route('/api/...')` or `@bp.route(...)` decorator
- [ ] Appropriate auth decorator is used (`@guest_mode_required` or `@login_required`)
- [ ] Response format follows standard: `{'success': bool, 'data': dict, 'error': str}`
- [ ] Success responses use `jsonify({'success': True, 'data': ...})`
- [ ] Error responses use `jsonify({'success': False, 'error': 'message'})` with status code
- [ ] GET requests include `add_cache_busting_headers(response)`
- [ ] User context retrieved via `get_current_user_id()` and `get_current_data_folder()`
- [ ] Error handling uses try/except with proper logging (`print()` and `traceback.print_exc()`)
- [ ] Production errors tracked with `app_insights.track_exception()`
- [ ] Input validation performed before processing
- [ ] Database operations use `DatabaseManager` class
- [ ] File operations use `FileManager` class

---

## Frontend Tables

If you're adding or modifying a table:

- [ ] Table wrapped in `<div class="table-responsive">` with max-height styling
- [ ] Table uses `class="table table-hover"` with unique `id`
- [ ] Header uses `class="table-dark"`
- [ ] Sortable headers include:
  - [ ] `class="custom-tooltip sortable"`
  - [ ] `data-key="column_key"` attribute
  - [ ] `data-tooltip="description"` attribute
  - [ ] `onclick="sortFunctionName('key')"` handler
  - [ ] Sort indicator: `<span id="sort-indicator-key"></span>`
- [ ] Sorting function follows standard pattern:
  - [ ] State variables: `{tableName}SortKey`, `{tableName}SortAsc`, `{tableName}TableData`
  - [ ] Function name: `sort{TableName}Table(key)`
  - [ ] Toggles direction when same column clicked
  - [ ] Updates all sort indicators
  - [ ] Re-renders table after sorting
- [ ] Render function:
  - [ ] Handles empty data gracefully
  - [ ] Sorts data array before rendering
  - [ ] Updates sort indicators
  - [ ] Calls `initializeCustomTooltips()` after rendering (with setTimeout)
- [ ] Table data loaded from API with proper error handling

---

## Frontend UI Elements

If you're adding UI elements:

### Buttons
- [ ] Uses Bootstrap classes (`btn btn-primary`, `btn btn-outline-secondary`, etc.)
- [ ] Includes FontAwesome icon if appropriate (`<i class="fas fa-icon"></i>`)
- [ ] Has `onclick` handler or proper event listener
- [ ] Uses `data-i18n` attribute for translatable text

### Cards
- [ ] Uses `class="card"` for standard cards
- [ ] Uses `class="card metric-card"` for metric cards
- [ ] Has `card-header` and `card-body` structure
- [ ] Follows established card patterns

### Modals
- [ ] Uses Bootstrap modal structure
- [ ] Has proper `id` attribute
- [ ] Includes `modal-header`, `modal-body`, `modal-footer`
- [ ] Footer buttons follow button patterns
- [ ] Close button uses `data-bs-dismiss="modal"`

### Tooltips
- [ ] Elements use `class="custom-tooltip"`
- [ ] Tooltip text in `data-tooltip` attribute
- [ ] `initializeCustomTooltips()` called after DOM updates
- [ ] Called with `setTimeout(..., 100)` to ensure DOM ready

---

## Charts

If you're adding or modifying a chart:

- [ ] Chart method uses `@cache_chart_result()` decorator
- [ ] Method returns dict with Plotly structure: `{'data': ..., 'layout': ...}`
- [ ] Error cases use `_empty_chart(message)`
- [ ] Chart uses `ChartFactory` pattern: `ChartFactory(portfolio).create_chart('type')`
- [ ] Chart endpoint follows standard pattern in `get_chart()` function
- [ ] Chart added to `get_available_charts()` dictionary
- [ ] Color palette uses `self.colors` array from `ChartGenerator`
- [ ] Chart handles empty data gracefully

---

## Data Models

If you're adding or modifying data models:

- [ ] Class follows established structure (like `Portfolio`, `Strategy`, `Trade`)
- [ ] Methods use type hints: `def method(self) -> Dict:`
- [ ] Database operations use `DatabaseManager` class
- [ ] No direct SQLite connections outside of `DatabaseManager`
- [ ] Methods return typed structures (Dict, List, etc.)
- [ ] Empty data cases handled gracefully

---

## File Management

If you're working with files:

- [ ] Uses `FileManager` class for file operations
- [ ] User-specific paths via `get_current_data_folder()`
- [ ] Never hardcodes user directories
- [ ] Supports both authenticated and guest users
- [ ] Metadata tracked (friendly names, upload dates, file sizes)

---

## Error Handling

- [ ] Backend: Try/except blocks with proper logging
- [ ] Backend: User-friendly error messages
- [ ] Backend: Detailed errors logged to console with `traceback.print_exc()`
- [ ] Frontend: API errors handled with user feedback
- [ ] Frontend: Error messages displayed in consistent format (Bootstrap alerts)
- [ ] Production errors tracked with `app_insights.track_exception()`

---

## Internationalization (i18n)

If you're adding user-facing text:

- [ ] Text uses `data-i18n` attribute
- [ ] Translations added to `translations/en/frontend.json`
- [ ] Translations added to `translations/es/frontend.json`
- [ ] Tooltips use `data-i18n-title` if translatable

---

## Code Quality

- [ ] Function and class names follow naming conventions:
  - [ ] Python: `snake_case` for functions, `PascalCase` for classes
  - [ ] JavaScript: `camelCase` for functions and variables
  - [ ] HTML IDs: `camelCase`
  - [ ] CSS classes: `kebab-case` for custom classes
- [ ] Code is organized and follows single responsibility principle
- [ ] Docstrings added to functions and classes
- [ ] Complex logic documented with inline comments
- [ ] No code duplication (reused existing functions where possible)

---

## Performance

- [ ] Expensive operations use caching (`@cache_chart_result()`)
- [ ] Database queries use proper indexes
- [ ] GET responses include cache-busting headers
- [ ] Frontend: Tables render efficiently (no unnecessary re-renders)
- [ ] Frontend: Tooltips initialized only when needed

---

## Security

- [ ] User input validated before processing
- [ ] Database queries use parameterized statements (no SQL injection)
- [ ] Sensitive data not exposed in error messages
- [ ] User context properly isolated (no data leakage between users)
- [ ] File uploads validated (type, size, etc.)

---

## Testing Checklist

Before submitting:

- [ ] Code tested with both authenticated and guest users
- [ ] Error cases tested (empty data, invalid input, etc.)
- [ ] Tables sort correctly in all columns
- [ ] Tooltips display correctly
- [ ] Charts render correctly with data
- [ ] Charts handle empty data gracefully
- [ ] API responses follow standard format
- [ ] Error messages are user-friendly
- [ ] No console errors in browser
- [ ] No Python errors in server logs

---

## Documentation

- [ ] New endpoints documented in README.md (if public-facing)
- [ ] Complex logic has inline comments
- [ ] Function docstrings explain purpose and parameters
- [ ] This checklist updated if new patterns emerge

---

## Final Review

- [ ] All applicable items above are checked
- [ ] Code follows patterns from existing similar features
- [ ] No deviations from established patterns without good reason
- [ ] Code reviewed for consistency with style guide
- [ ] Ready for code review

---

## Notes

If you had to deviate from established patterns, document why here:

```
Reason for deviation:
[Your explanation]
```

---

**Remember**: Consistency is more important than perfection. When in doubt, follow existing patterns rather than creating new ones.

