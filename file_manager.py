"""
File manager utility for handling CSV files with friendly names and user partitioning.
"""

import json
import os
import glob
from datetime import datetime
from typing import Dict, List, Optional

# Configure paths for Azure App Service
if os.environ.get('WEBSITES_PORT'):
    # Azure App Service - use persistent paths for data
    DEFAULT_DATA_DIR = "/home/site/wwwroot/data"
else:
    # Local development
    DEFAULT_DATA_DIR = "data"

class FileManager:
    """Manages CSV files with friendly naming support and user-specific data folders."""
    
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.base_data_dir = data_dir
        self.data_dir = data_dir  # This will be updated based on current user
        self.metadata_file = os.path.join(data_dir, 'file_metadata.json')
        os.makedirs(data_dir, exist_ok=True)
        self.metadata = self._load_metadata()
    
    def set_user_data_dir(self, user_data_dir: str):
        """Set the current user's data directory."""
        self.data_dir = user_data_dir
        self.metadata_file = os.path.join(user_data_dir, 'file_metadata.json')
        os.makedirs(user_data_dir, exist_ok=True)
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load file metadata from JSON file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_metadata(self):
        """Save file metadata to JSON file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
    
    def save_file(self, file, friendly_name: Optional[str] = None) -> Dict:
        """
        Save uploaded file with optional friendly name.
        Returns dict with filename and friendly_name.
        """
        # Generate timestamped filename
        base_name = os.path.splitext(file.filename)[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        actual_filename = f"{base_name}_{timestamp}.csv"
        file_path = os.path.join(self.data_dir, actual_filename)
        
        # Save the file
        file.save(file_path)
        
        # Set friendly name (use original filename if not provided)
        if not friendly_name:
            friendly_name = base_name
        
        # Store metadata
        self.metadata[actual_filename] = {
            'friendly_name': friendly_name,
            'original_filename': file.filename,
            'upload_date': datetime.now().isoformat(),
            'file_size': os.path.getsize(file_path)
        }
        self._save_metadata()
        
        return {
            'filename': actual_filename,
            'friendly_name': friendly_name
        }
    
    def get_file_list(self) -> List[Dict]:
        """Get list of files with their friendly names, sorted newest first."""
        # Get all CSV files, excluding debug folder
        csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
        
        file_info = []
        for file_path in csv_files:
            # Skip files in debug folder
            if 'debug' in file_path.split(os.sep):
                continue
                
            filename = os.path.basename(file_path)
            mod_time = os.path.getmtime(file_path)
            mod_datetime = datetime.fromtimestamp(mod_time)
            
            # Get friendly name from metadata
            metadata = self.metadata.get(filename, {})
            friendly_name = metadata.get('friendly_name', filename.replace('.csv', ''))
            
            file_info.append({
                'filename': filename,
                'friendly_name': friendly_name,
                'display_name': friendly_name,
                'upload_date': mod_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': mod_time,
                'original_filename': metadata.get('original_filename', filename),
                'file_size': metadata.get('file_size', os.path.getsize(file_path))
            })
        
        # Sort by timestamp (newest first)
        file_info.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Remove timestamp from response
        for info in file_info:
            del info['timestamp']
        
        return file_info
    
    def update_friendly_name(self, filename: str, new_friendly_name: str) -> bool:
        """Update the friendly name for a file."""
        if filename in self.metadata:
            self.metadata[filename]['friendly_name'] = new_friendly_name
            self._save_metadata()
            return True
        else:
            # File exists but no metadata - create it
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                self.metadata[filename] = {
                    'friendly_name': new_friendly_name,
                    'original_filename': filename,
                    'upload_date': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                    'file_size': os.path.getsize(file_path)
                }
                self._save_metadata()
                return True
        return False
    
    def get_friendly_name(self, filename: str) -> str:
        """Get friendly name for a file."""
        metadata = self.metadata.get(filename, {})
        return metadata.get('friendly_name', filename.replace('.csv', ''))
    
    def file_exists(self, filename: str) -> bool:
        """Check if file exists."""
        file_path = os.path.join(self.data_dir, filename)
        return os.path.exists(file_path)
    
    def get_file_path(self, filename: str) -> str:
        """Get full path to file."""
        return os.path.join(self.data_dir, filename)
    
    def cleanup_duplicates(self) -> Dict:
        """Remove duplicate files, keeping only the most recent version."""
        csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
        
        if not csv_files:
            return {'deleted_count': 0, 'message': 'No files to clean up'}
        
        # Group files by base name (extract base name from timestamped filename or capital pattern)
        file_groups = {}
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            
            # First, remove the __capital=XXXXX pattern if present
            base_name = filename
            if '__capital=' in base_name:
                base_name = base_name.split('__capital=')[0]
            
            # Then extract base name by removing timestamp pattern (_YYYYMMDD_HHMMSS)
            if '_' in base_name and base_name.count('_') >= 2:
                parts = base_name.split('_')
                if len(parts) >= 3:
                    # Try to find timestamp pattern
                    try:
                        # Check if last two parts look like date/time
                        date_part = parts[-2]  # YYYYMMDD
                        time_part = parts[-1].replace('.csv', '')  # HHMMSS
                        
                        if (len(date_part) == 8 and date_part.isdigit() and 
                            len(time_part) == 6 and time_part.isdigit()):
                            # Valid timestamp pattern, extract base name
                            base_name = '_'.join(parts[:-2])
                        else:
                            base_name = base_name.replace('.csv', '')
                    except:
                        base_name = base_name.replace('.csv', '')
                else:
                    base_name = base_name.replace('.csv', '')
            else:
                base_name = base_name.replace('.csv', '')
            
            if base_name not in file_groups:
                file_groups[base_name] = []
            
            file_groups[base_name].append({
                'filename': filename,
                'path': file_path,
                'mod_time': os.path.getmtime(file_path)
            })
        
        # For each group with multiple files, keep only the newest
        deleted_count = 0
        deleted_files = []
        for base_name, files in file_groups.items():
            if len(files) > 1:
                # Sort by modification time (newest first)
                files.sort(key=lambda x: x['mod_time'], reverse=True)
                
                # Keep the first (newest) file, delete the rest
                for file_info in files[1:]:
                    try:
                        os.remove(file_info['path'])
                        # Remove from metadata too
                        if file_info['filename'] in self.metadata:
                            del self.metadata[file_info['filename']]
                        deleted_count += 1
                        deleted_files.append(file_info['filename'])
                    except Exception as e:
                        print(f"Warning: Could not delete {file_info['filename']}: {e}")
        
        # Save updated metadata
        self._save_metadata()
        
        return {
            'deleted_count': deleted_count,
            'message': f'Cleaned up {deleted_count} duplicate files',
            'deleted_files': deleted_files
        } 