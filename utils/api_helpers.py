"""
API helper functions for consistent response formatting and user context management.
"""

from flask import jsonify
from functools import wraps
from typing import Dict, Any, Optional, Tuple


def success_response(data: Dict[str, Any] = None, message: str = None) -> Dict:
    """
    Create a standard success response.
    
    Args:
        data: Response data dictionary
        message: Optional success message
        
    Returns:
        Standard success response dict
    """
    response = {'success': True}
    if data is not None:
        response['data'] = data
    if message:
        response['message'] = message
    return response


def error_response(error: str, status_code: int = 400, additional_data: Dict = None) -> Tuple[Dict, int]:
    """
    Create a standard error response.
    
    Args:
        error: Error message string
        status_code: HTTP status code (default: 400)
        additional_data: Optional additional error data
        
    Returns:
        Tuple of (error response dict, status code)
    """
    response = {
        'success': False,
        'error': error
    }
    if additional_data:
        response.update(additional_data)
    return response, status_code


def get_user_context():
    """
    Get current user context (user_id and data_folder).
    This is a wrapper that can be extended with additional context.
    
    Returns:
        Dict with user_id and data_folder
    """
    # Import here to avoid circular dependencies
    from app import get_current_user_id, get_current_data_folder
    
    return {
        'user_id': get_current_user_id(),
        'data_folder': get_current_data_folder()
    }


def api_endpoint_wrapper(func):
    """
    Decorator wrapper for API endpoints that provides:
    - Standard error handling
    - User context injection
    - Response formatting
    
    Usage:
        @app.route('/api/endpoint')
        @api_endpoint_wrapper
        def my_endpoint():
            # Access user context via get_user_context()
            return success_response({'result': 'data'})
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
            # If result is already a tuple (response, status_code), return as-is
            if isinstance(result, tuple) and len(result) == 2:
                response_dict, status_code = result
                if isinstance(response_dict, dict):
                    # Ensure it has success field
                    if 'success' not in response_dict:
                        response_dict = success_response(response_dict)
                    return jsonify(response_dict), status_code
                return result
            
            # If result is a dict, wrap it if needed
            if isinstance(result, dict):
                if 'success' not in result:
                    result = success_response(result)
                return jsonify(result)
            
            # Otherwise return as-is (might be a Response object)
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            # Try to get user context for logging
            try:
                from app import get_current_user_id, app_insights
                user_id = get_current_user_id()
                app_insights.track_exception(e, {
                    'user_id': user_id,
                    'endpoint': func.__name__
                })
            except:
                pass
            
            return jsonify(error_response(str(e), 500))
    
    return wrapper

