"""
JSON parser with trio support.
"""
import logging
from typing import List, Tuple, Optional, Dict, Any
import json

import trio
import aiofiles

from app.services.document_processing.parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)


class JSONParser(BaseParser):
    """
    Parser for JSON files with structure preservation.
    """
    
    async def __call__(self, 
                      file_path: Optional[str] = None, 
                      binary_content: Optional[bytes] = None, 
                      **kwargs) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
        """
        Parse JSON file content.

        Args:
            file_path: Path to the JSON file (optional)
            binary_content: Binary content of the JSON file (optional)
            **kwargs: Additional parser-specific parameters
                - flatten: Whether to flatten the JSON structure (default: False)
                - max_depth: Maximum depth to parse (default: None)
                - include_path: Whether to include JSON path in metadata (default: True)
                - extract_arrays: Whether to extract arrays as separate items (default: True)
            
        Returns:
            List of tuples containing (text_content, metadata_dict)
        """
        try:
            # Extract parameters
            flatten = kwargs.get('flatten', False)
            max_depth = kwargs.get('max_depth', None)
            include_path = kwargs.get('include_path', True)
            extract_arrays = kwargs.get('extract_arrays', True)
            
            # Get JSON content
            if binary_content is not None:
                # Parse from binary content
                encoding = self.detect_encoding(binary_content)
                json_content = binary_content.decode(encoding, errors='replace')
            
            elif file_path is not None:
                # Parse from file path
                async with aiofiles.open(file_path, 'rb') as f:
                    content = await f.read()
                    encoding = self.detect_encoding(content)
                    json_content = content.decode(encoding, errors='replace')
            
            else:
                raise ValueError("Either file_path or binary_content must be provided")
            
            # Parse JSON in thread
            parsed_content = await trio.to_thread.run_sync(
                self._parse_json_sync,
                json_content,
                flatten,
                max_depth,
                include_path,
                extract_arrays
            )
            
            return parsed_content
                
        except Exception as e:
            logger.error(f"Error parsing JSON file: {e}")
            # Return empty content in case of error
            return [(f"Error parsing JSON file: {str(e)}", {"type": "error", "error": str(e)})]
    
    def _parse_json_sync(self, 
                        json_content: str,
                        flatten: bool,
                        max_depth: Optional[int],
                        include_path: bool,
                        extract_arrays: bool) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Parse JSON content synchronously.
        """
        result = []
        
        try:
            # Parse JSON
            data = json.loads(json_content)
            
            if flatten:
                # Flatten JSON structure
                flattened_content = self._flatten_json(data, max_depth)
                if flattened_content:
                    result.append((flattened_content, {
                        "type": "json",
                        "format": "flattened"
                    }))
            else:
                # Process JSON structure
                result = self._process_json_value(data, "", 0, max_depth, include_path, extract_arrays)
            
            return result
            
        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
            return [(f"Error parsing JSON: {str(e)}", {"type": "error", "error": str(e)})]
    
    def _process_json_value(self, 
                           value: Any, 
                           path: str, 
                           depth: int,
                           max_depth: Optional[int],
                           include_path: bool,
                           extract_arrays: bool) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process a JSON value and its children.
        """
        result = []
        
        # Check depth limit
        if max_depth is not None and depth > max_depth:
            return result
        
        if isinstance(value, dict):
            # Process dictionary
            for key, val in value.items():
                current_path = f"{path}.{key}" if path else key
                child_results = self._process_json_value(
                    val, current_path, depth + 1, max_depth, include_path, extract_arrays
                )
                result.extend(child_results)
        
        elif isinstance(value, list):
            if extract_arrays:
                # Process array items separately
                for i, item in enumerate(value):
                    current_path = f"{path}[{i}]" if path else f"[{i}]"
                    child_results = self._process_json_value(
                        item, current_path, depth + 1, max_depth, include_path, extract_arrays
                    )
                    result.extend(child_results)
            else:
                # Process array as a single item
                array_text = json.dumps(value, ensure_ascii=False, indent=2)
                metadata = {
                    "type": "json_array",
                    "depth": depth,
                    "array_length": len(value)
                }
                
                if include_path:
                    metadata["path"] = path
                
                result.append((array_text, metadata))
        
        else:
            # Process primitive value
            if value is not None:
                text = str(value)
                
                # Create metadata
                metadata = {
                    "type": "json_value",
                    "value_type": type(value).__name__,
                    "depth": depth
                }
                
                if include_path:
                    metadata["path"] = path
                
                result.append((text, metadata))
        
        return result
    
    def _flatten_json(self, 
                     data: Any, 
                     max_depth: Optional[int],
                     current_depth: int = 0,
                     path: str = "") -> str:
        """
        Flatten JSON structure into a single text.
        """
        if max_depth is not None and current_depth > max_depth:
            return ""
        
        parts = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                
                if isinstance(value, (dict, list)):
                    # Recursive processing
                    child_text = self._flatten_json(value, max_depth, current_depth + 1, current_path)
                    if child_text:
                        parts.append(f"{key}: {child_text}")
                else:
                    # Primitive value
                    parts.append(f"{key}: {str(value)}")
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                
                if isinstance(item, (dict, list)):
                    # Recursive processing
                    child_text = self._flatten_json(item, max_depth, current_depth + 1, current_path)
                    if child_text:
                        parts.append(child_text)
                else:
                    # Primitive value
                    parts.append(str(item))
        
        else:
            # Primitive value
            parts.append(str(data))
        
        return " ".join(parts)
    
    def _extract_text_values(self, data: Any) -> List[str]:
        """
        Extract all text values from JSON structure.
        """
        text_values = []
        
        if isinstance(data, dict):
            for value in data.values():
                text_values.extend(self._extract_text_values(value))
        
        elif isinstance(data, list):
            for item in data:
                text_values.extend(self._extract_text_values(item))
        
        elif isinstance(data, str):
            text_values.append(data)
        
        elif data is not None:
            text_values.append(str(data))
        
        return text_values