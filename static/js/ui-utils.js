/**
 * UI Utility Functions for Consistent UI Element Creation
 * 
 * These functions help maintain consistency when creating tables, buttons, tooltips,
 * and other UI elements across the application.
 */

/**
 * Create a sortable table header with tooltip
 * 
 * @param {string} key - Column key for sorting
 * @param {string} label - Column label text
 * @param {string} tooltip - Tooltip text explaining the column
 * @param {string} sortFunction - Name of the sorting function to call
 * @param {string} sortIndicatorId - ID for the sort indicator span
 * @param {object} style - Optional inline styles object
 * @returns {string} HTML string for the table header
 */
function createSortableTableHeader(key, label, tooltip, sortFunction, sortIndicatorId = null, style = {}) {
    const indicatorId = sortIndicatorId || `sort-indicator-${key}`;
    const styleStr = Object.entries(style)
        .map(([prop, value]) => `${prop.replace(/([A-Z])/g, '-$1').toLowerCase()}: ${value}`)
        .join('; ');
    
    return `<th style="cursor:pointer; ${styleStr}" 
            data-key="${key}" 
            onclick="${sortFunction}('${key}')" 
            class="custom-tooltip sortable" 
            data-tooltip="${tooltip}">
            ${label} <span id="${indicatorId}"></span>
        </th>`;
}

/**
 * Create a complete sortable table structure
 * 
 * @param {string} tableId - Unique ID for the table
 * @param {Array} headers - Array of header objects: {key, label, tooltip, sortFunction}
 * @param {Array} data - Array of data objects for table rows
 * @param {function} rowRenderer - Function to render each row: (item) => string
 * @param {object} options - Optional configuration: {maxHeight, classes, theadClass}
 * @returns {string} Complete HTML table structure
 */
function createTableStructure(tableId, headers, data, rowRenderer, options = {}) {
    const maxHeight = options.maxHeight || '600px';
    const tableClasses = options.classes || 'table table-hover';
    const theadClass = options.theadClass || 'table-dark';
    
    let html = `<div class="table-responsive" style="max-height: ${maxHeight}; overflow-x: auto; overflow-y: auto;">`;
    html += `<table class="${tableClasses}" id="${tableId}">`;
    html += `<thead class="${theadClass}"><tr>`;
    
    headers.forEach(header => {
        html += createSortableTableHeader(
            header.key,
            header.label,
            header.tooltip,
            header.sortFunction,
            header.sortIndicatorId,
            header.style
        );
    });
    
    html += `</tr></thead><tbody>`;
    
    if (data && data.length > 0) {
        data.forEach(item => {
            html += rowRenderer(item);
        });
    } else {
        html += `<tr><td colspan="${headers.length}" class="text-center text-muted">No data available</td></tr>`;
    }
    
    html += `</tbody></table></div>`;
    return html;
}

/**
 * Initialize table sorting with standard pattern
 * 
 * @param {string} tableId - ID of the table
 * @param {string} sortFunctionName - Name of the sorting function
 * @param {string} defaultSortKey - Default column to sort by
 * @param {Array} columnKeys - Array of all column keys for indicator updates
 */
function initializeTableSorting(tableId, sortFunctionName, defaultSortKey, columnKeys) {
    // Create state variables
    window[`${tableId}SortKey`] = defaultSortKey;
    window[`${tableId}SortAsc`] = true;
    window[`${tableId}TableData`] = [];
    
    // Create sorting function
    window[sortFunctionName] = function(key) {
        console.log(`🔧 ${sortFunctionName} called with key:`, key);
        
        const sortKeyVar = `${tableId}SortKey`;
        const sortAscVar = `${tableId}SortAsc`;
        
        if (window[sortKeyVar] === key) {
            window[sortAscVar] = !window[sortAscVar];
        } else {
            window[sortKeyVar] = key;
            // Default: true for text columns, false for numeric
            window[sortAscVar] = typeof key === 'string' && key.includes('name');
        }
        
        // Update indicators
        columnKeys.forEach(colKey => {
            const indicator = document.getElementById(`sort-indicator-${colKey}`);
            if (indicator) {
                if (colKey === window[sortKeyVar]) {
                    indicator.textContent = window[sortAscVar] ? '▲' : '▼';
                } else {
                    indicator.textContent = '';
                }
            }
        });
        
        // Trigger re-render (assumes render function exists)
        const renderFunction = `render${tableId.charAt(0).toUpperCase() + tableId.slice(1)}`;
        if (typeof window[renderFunction] === 'function') {
            setTimeout(() => {
                window[renderFunction]();
            }, 10);
        }
    };
}

/**
 * Create a standard button element
 * 
 * @param {string} classes - Bootstrap button classes (e.g., 'btn btn-primary')
 * @param {string} text - Button text
 * @param {string} icon - FontAwesome icon class (e.g., 'fa-upload')
 * @param {string} onClick - onClick handler function name or inline code
 * @param {string} id - Optional button ID
 * @param {object} attributes - Additional HTML attributes
 * @returns {string} HTML button element
 */
function createButton(classes, text, icon = null, onClick = null, id = null, attributes = {}) {
    let html = '<button';
    
    if (id) html += ` id="${id}"`;
    html += ` class="${classes}"`;
    
    if (onClick) {
        html += ` onclick="${onClick}"`;
    }
    
    Object.entries(attributes).forEach(([key, value]) => {
        html += ` ${key}="${value}"`;
    });
    
    html += '>';
    
    if (icon) {
        html += `<i class="fas ${icon}"></i> `;
    }
    
    html += text;
    html += '</button>';
    
    return html;
}

/**
 * Create a card structure
 * 
 * @param {string} type - Card type: 'default', 'metric', 'chart'
 * @param {string} content - Card body content
 * @param {string} title - Optional card title/header
 * @param {string} id - Optional card ID
 * @returns {string} HTML card structure
 */
function createCard(type, content, title = null, id = null) {
    let html = '<div';
    if (id) html += ` id="${id}"`;
    
    const cardClass = type === 'metric' ? 'card metric-card' : 'card';
    html += ` class="${cardClass}">`;
    
    if (title) {
        html += `<div class="card-header"><h5>${title}</h5></div>`;
    }
    
    html += `<div class="card-body">${content}</div>`;
    html += '</div>';
    
    return html;
}

/**
 * Initialize tooltips for elements with custom-tooltip class
 * This should be called after DOM updates
 * 
 * Note: This assumes initializeCustomTooltips() exists in the main dashboard
 * This is a wrapper that ensures it's called with proper timing
 */
function initializeTooltips() {
    setTimeout(() => {
        if (typeof initializeCustomTooltips === 'function') {
            initializeCustomTooltips();
        } else {
            console.warn('initializeCustomTooltips() not found. Tooltips may not work.');
        }
    }, 100);
}

/**
 * Update sort indicators for a table
 * 
 * @param {string} tableId - Table ID
 * @param {string} activeKey - Currently active sort key
 * @param {boolean} ascending - Sort direction
 * @param {Array} allKeys - All column keys that have indicators
 */
function updateSortIndicators(tableId, activeKey, ascending, allKeys) {
    allKeys.forEach(key => {
        const indicator = document.getElementById(`sort-indicator-${key}`);
        if (indicator) {
            if (key === activeKey) {
                indicator.textContent = ascending ? '▲' : '▼';
            } else {
                indicator.textContent = '';
            }
        }
    });
}

/**
 * Sort table data array
 * 
 * @param {Array} data - Array of data objects
 * @param {string} sortKey - Key to sort by
 * @param {boolean} ascending - Sort direction
 * @param {function} valueExtractor - Optional function to extract sort value: (item, key) => value
 * @returns {Array} Sorted array
 */
function sortTableData(data, sortKey, ascending, valueExtractor = null) {
    const sorted = [...data].sort((a, b) => {
        let valA, valB;
        
        if (valueExtractor) {
            valA = valueExtractor(a, sortKey);
            valB = valueExtractor(b, sortKey);
        } else {
            valA = a[sortKey];
            valB = b[sortKey];
        }
        
        // Handle different types
        if (typeof valA === 'string' && typeof valB === 'string') {
            valA = valA.toLowerCase();
            valB = valB.toLowerCase();
        } else {
            valA = parseFloat(valA) || 0;
            valB = parseFloat(valB) || 0;
        }
        
        if (ascending) {
            return valA > valB ? 1 : -1;
        } else {
            return valA < valB ? 1 : -1;
        }
    });
    
    return sorted;
}

/**
 * Create a modal structure
 * 
 * @param {string} modalId - Unique modal ID
 * @param {string} title - Modal title
 * @param {string} body - Modal body content
 * @param {Array} footerButtons - Array of button objects: {text, classes, onClick, icon}
 * @param {string} size - Modal size: 'sm', 'lg', 'xl' (default: 'lg')
 * @returns {string} Complete modal HTML
 */
function createModal(modalId, title, body, footerButtons = [], size = 'lg') {
    let html = `<div class="modal fade" id="${modalId}" tabindex="-1">`;
    html += `<div class="modal-dialog modal-${size}">`;
    html += `<div class="modal-content">`;
    
    // Header
    html += `<div class="modal-header">`;
    html += `<h5 class="modal-title">${title}</h5>`;
    html += `<button type="button" class="btn-close" data-bs-dismiss="modal"></button>`;
    html += `</div>`;
    
    // Body
    html += `<div class="modal-body">${body}</div>`;
    
    // Footer
    if (footerButtons.length > 0) {
        html += `<div class="modal-footer">`;
        footerButtons.forEach(btn => {
            html += createButton(
                btn.classes || 'btn btn-primary',
                btn.text,
                btn.icon,
                btn.onClick,
                btn.id,
                btn.attributes || {}
            );
        });
        html += `</div>`;
    }
    
    html += `</div></div></div>`;
    return html;
}

// Export functions for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        createSortableTableHeader,
        createTableStructure,
        initializeTableSorting,
        createButton,
        createCard,
        initializeTooltips,
        updateSortIndicators,
        sortTableData,
        createModal
    };
}

