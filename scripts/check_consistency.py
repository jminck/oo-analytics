#!/usr/bin/env python3
"""
Consistency checker for development style guide compliance.

This script checks code against the patterns defined in DEVELOPMENT_STYLE_GUIDE.md
and reports violations.
"""

import os
import re
import sys
import ast
from pathlib import Path
from typing import List, Dict, Tuple
import json

# Color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class ConsistencyChecker:
    """Check code for consistency violations."""
    
    def __init__(self, root_dir: str = None):
        self.root_dir = Path(root_dir or os.getcwd())
        self.violations = []
        self.warnings = []
        
    def check_all(self) -> bool:
        """Run all consistency checks."""
        print(f"{BLUE}🔍 Running consistency checks...{RESET}\n")
        
        # Check Python files
        python_files = list(self.root_dir.rglob('*.py'))
        python_files = [f for f in python_files if not self._should_skip(f)]
        
        for py_file in python_files:
            self.check_python_file(py_file)
        
        # Check HTML/JS files
        html_files = list(self.root_dir.rglob('*.html'))
        html_files = [f for f in html_files if 'templates' in str(f)]
        
        for html_file in html_files:
            self.check_html_file(html_file)
        
        # Check JavaScript files
        js_files = list(self.root_dir.rglob('*.js'))
        js_files = [f for f in js_files if not self._should_skip(f)]
        
        for js_file in js_files:
            self.check_js_file(js_file)
        
        # Print results
        self.print_results()
        
        return len(self.violations) == 0
    
    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = [
            '__pycache__',
            '.git',
            'node_modules',
            'venv',
            'env',
            '.venv',
            'migrations',
        ]
        path_str = str(file_path)
        return any(pattern in path_str for pattern in skip_patterns)
    
    def check_python_file(self, file_path: Path):
        """Check Python file for API endpoint patterns."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            self.warnings.append(f"{file_path}: Could not read file: {e}")
            return
        
        # Check for API endpoints
        if '@app.route' in content or '@bp.route' in content:
            self.check_api_endpoints(file_path, content, lines)
        
        # Check for jsonify usage
        if 'jsonify' in content:
            self.check_jsonify_usage(file_path, content, lines)
    
    def check_api_endpoints(self, file_path: Path, content: str, lines: List[str]):
        """Check API endpoints follow standard patterns."""
        # Find all route decorators
        route_pattern = r'@(app|bp)\.route\([\'"]([^\'"]+)[\'"]'
        routes = re.finditer(route_pattern, content)
        
        for route_match in routes:
            route_line_num = content[:route_match.start()].count('\n') + 1
            route_path = route_match.group(2)
            
            # Find the function definition after this decorator
            func_start = route_match.end()
            func_match = re.search(r'def\s+(\w+)\s*\(', content[func_start:func_start+500])
            
            if not func_match:
                continue
            
            func_name = func_match.group(1)
            func_line_num = route_line_num + content[route_match.end():route_match.end()+func_match.start()].count('\n')
            
            # Check for try/except
            func_content = self._get_function_content(content, func_line_num)
            if 'try:' not in func_content:
                self.violations.append({
                    'file': str(file_path),
                    'line': func_line_num,
                    'type': 'api_endpoint',
                    'message': f"API endpoint '{func_name}' missing try/except error handling"
                })
            
            # Check for standard response format
            if 'jsonify' in func_content:
                if "'success':" not in func_content and '"success":' not in func_content:
                    self.violations.append({
                        'file': str(file_path),
                        'line': func_line_num,
                        'type': 'api_response',
                        'message': f"API endpoint '{func_name}' response missing 'success' field"
                    })
            
            # Check for user context
            if '/api/' in route_path and 'get_current_user_id' not in func_content and 'get_current_data_folder' not in func_content:
                # Only warn, not error, as some endpoints might not need it
                if 'guest_mode_required' in func_content or 'login_required' in func_content:
                    self.warnings.append(f"{file_path}:{func_line_num} - Consider using get_current_user_id() in '{func_name}'")
    
    def check_jsonify_usage(self, file_path: Path, content: str, lines: List[str]):
        """Check jsonify calls follow standard format."""
        jsonify_pattern = r'jsonify\s*\(([^)]+)\)'
        matches = re.finditer(jsonify_pattern, content)
        
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            jsonify_content = match.group(1)
            
            # Check for success field
            if "'success':" not in jsonify_content and '"success":' not in jsonify_content:
                # Might be using helper function, check context
                context_start = max(0, match.start() - 100)
                context = content[context_start:match.start()]
                if 'success_response' not in context and 'error_response' not in context:
                    self.violations.append({
                        'file': str(file_path),
                        'line': line_num,
                        'type': 'jsonify_format',
                        'message': "jsonify() call missing 'success' field. Use success_response() or error_response() helpers."
                    })
    
    def check_html_file(self, file_path: Path):
        """Check HTML file for table patterns."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.warnings.append(f"{file_path}: Could not read file: {e}")
            return
        
        # Check for tables
        if '<table' in content:
            self.check_table_patterns(file_path, content)
    
    def check_table_patterns(self, file_path: Path, content: str):
        """Check table HTML follows standard patterns."""
        # Find all tables
        table_pattern = r'<table[^>]*id=["\'](\w+)["\']'
        tables = re.finditer(table_pattern, content)
        
        for table_match in tables:
            table_id = table_match.group(1)
            table_start = table_match.start()
            
            # Get table context (next 2000 chars)
            table_context = content[table_start:table_start+2000]
            
            # Check for table-responsive wrapper
            if 'table-responsive' not in content[max(0, table_start-200):table_start]:
                self.violations.append({
                    'file': str(file_path),
                    'line': content[:table_start].count('\n') + 1,
                    'type': 'table_structure',
                    'message': f"Table '{table_id}' missing 'table-responsive' wrapper div"
                })
            
            # Check for table-dark thead
            if '<thead' in table_context and 'table-dark' not in table_context:
                self.violations.append({
                    'file': str(file_path),
                    'line': content[:table_start].count('\n') + 1,
                    'type': 'table_structure',
                    'message': f"Table '{table_id}' thead missing 'table-dark' class"
                })
            
            # Check for sortable headers
            if '<th' in table_context:
                # Check if headers have sortable class
                th_matches = list(re.finditer(r'<th[^>]*>', table_context))
                for th_match in th_matches:
                    th_tag = th_match.group(0)
                    if 'sortable' in th_tag or 'custom-tooltip' in th_tag:
                        # This is a sortable header, check for required attributes
                        if 'data-key' not in th_tag:
                            line_num = content[:table_start + th_match.start()].count('\n') + 1
                            self.violations.append({
                                'file': str(file_path),
                                'line': line_num,
                                'type': 'table_header',
                                'message': f"Sortable header in table '{table_id}' missing 'data-key' attribute"
                            })
                        if 'data-tooltip' not in th_tag:
                            line_num = content[:table_start + th_match.start()].count('\n') + 1
                            self.warnings.append(f"{file_path}:{line_num} - Sortable header missing 'data-tooltip' attribute")
    
    def check_js_file(self, file_path: Path):
        """Check JavaScript file for sorting function patterns."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.warnings.append(f"{file_path}: Could not read file: {e}")
            return
        
        # Check for sorting functions
        sort_func_pattern = r'function\s+sort(\w+)Table\s*\([^)]*\)'
        matches = re.finditer(sort_func_pattern, content)
        
        for match in matches:
            func_name = match.group(0)
            func_start = match.start()
            func_end = self._find_function_end(content, func_start)
            func_content = content[func_start:func_end]
            
            # Check for state variables (should be declared before function)
            table_name = match.group(1).lower()
            state_vars = [
                f'{table_name}SortKey',
                f'{table_name}SortAsc',
                f'{table_name}TableData'
            ]
            
            # Check if state variables exist (check before function)
            before_func = content[max(0, func_start-500):func_start]
            line_count = content[:func_start].count('\n') + 1
            for var in state_vars:
                if var not in before_func and var not in func_content:
                    warning_msg = (
                        f"{file_path}:{line_count} - "
                        f"Sorting function '{func_name}' might be missing state variable '{var}'"
                    )
                    self.warnings.append(warning_msg)
    
    def _get_function_content(self, content: str, func_line: int) -> str:
        """Get the content of a function starting at a line."""
        lines = content.split('\n')
        if func_line > len(lines):
            return ""
        
        # Find function start
        func_start_idx = func_line - 1
        indent_level = len(lines[func_start_idx]) - len(lines[func_start_idx].lstrip())
        
        # Collect function lines
        func_lines = [lines[func_start_idx]]
        for i in range(func_line, min(func_line + 100, len(lines))):
            line = lines[i]
            if line.strip() and not line.startswith(' ' * (indent_level + 1)) and not line.startswith('\t'):
                if line.strip().startswith('def ') or line.strip().startswith('class '):
                    break
            func_lines.append(line)
        
        return '\n'.join(func_lines)
    
    def _find_function_end(self, content: str, start_pos: int) -> int:
        """Find the end of a JavaScript function."""
        brace_count = 0
        in_function = False
        
        for i in range(start_pos, min(start_pos + 2000, len(content))):
            char = content[i]
            if char == '{':
                brace_count += 1
                in_function = True
            elif char == '}':
                brace_count -= 1
                if in_function and brace_count == 0:
                    return i + 1
        
        return min(start_pos + 2000, len(content))
    
    def print_results(self):
        """Print check results."""
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}Consistency Check Results{RESET}")
        print(f"{BLUE}{'='*60}{RESET}\n")
        
        if self.violations:
            print(f"{RED}❌ Found {len(self.violations)} violation(s):{RESET}\n")
            for violation in self.violations:
                print(f"{RED}  [{violation['type']}]{RESET} {violation['file']}:{violation['line']}")
                print(f"      {violation['message']}\n")
        else:
            print(f"{GREEN}✅ No violations found!{RESET}\n")
        
        if self.warnings:
            print(f"{YELLOW}⚠️  Found {len(self.warnings)} warning(s):{RESET}\n")
            for warning in self.warnings[:10]:  # Limit to first 10
                print(f"{YELLOW}  {warning}{RESET}")
            if len(self.warnings) > 10:
                print(f"{YELLOW}  ... and {len(self.warnings) - 10} more warnings{RESET}")
            print()
        
        print(f"{BLUE}{'='*60}{RESET}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check code consistency')
    parser.add_argument('--path', default='.', help='Path to check (default: current directory)')
    parser.add_argument('--strict', action='store_true', help='Treat warnings as errors')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    
    args = parser.parse_args()
    
    checker = ConsistencyChecker(args.path)
    success = checker.check_all()
    
    if args.strict and checker.warnings:
        success = False
    
    if args.json:
        output = {
            'success': success,
            'violations': checker.violations,
            'warnings': checker.warnings
        }
        print(json.dumps(output, indent=2))
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

