"""
Advanced table structure recognition with merged cells, headers, and spanning elements detection.
"""
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class CellType(str, Enum):
    """Cell type enumeration."""
    HEADER = "header"
    DATA = "data"
    MERGED = "merged"
    EMPTY = "empty"


@dataclass
class TableCell:
    """Table cell representation."""
    row: int
    col: int
    content: str
    cell_type: CellType
    rowspan: int = 1
    colspan: int = 1
    confidence: float = 1.0
    bbox: Optional[List[float]] = None
    is_merged_source: bool = False
    merged_from: Optional[Tuple[int, int]] = None


@dataclass
class TableStructure:
    """Table structure representation."""
    rows: int
    cols: int
    cells: List[List[Optional[TableCell]]]
    headers: List[int]  # Row indices that are headers
    confidence: float
    bbox: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class AdvancedTableStructureRecognizer:
    """Advanced table structure recognition with merged cells and spanning elements."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        
        # Patterns for header detection
        self.header_patterns = [
            r'^[A-Z][A-Z\s]+$',  # All caps
            r'^\d+\.\s*[A-Z]',   # Numbered headers
            r'^[A-Z][a-z]+\s*:',  # Title with colon
            r'^\*\*.*\*\*$',     # Bold markdown
            r'^#.*',             # Markdown headers
        ]
        
        # Common header keywords
        self.header_keywords = {
            'name', 'title', 'description', 'type', 'value', 'amount', 'date',
            'time', 'status', 'category', 'id', 'number', 'code', 'total',
            'sum', 'count', 'average', 'min', 'max', 'percent', 'rate'
        }
    
    def recognize_structure(self, 
                          table_data: List[List[str]], 
                          bbox: Optional[List[float]] = None) -> TableStructure:
        """
        Recognize table structure from raw table data.
        
        Args:
            table_data: 2D list of cell contents
            bbox: Bounding box coordinates [x0, y0, x1, y1]
            
        Returns:
            TableStructure object with recognized structure
        """
        try:
            if not table_data or not table_data[0]:
                return TableStructure(0, 0, [], [], 0.0)
            
            rows = len(table_data)
            cols = max(len(row) for row in table_data) if table_data else 0
            
            # Normalize table data (ensure all rows have same length)
            normalized_data = self._normalize_table_data(table_data, rows, cols)
            
            # Create initial cell structure
            cells = self._create_initial_cells(normalized_data)
            
            # Detect merged cells
            cells = self._detect_merged_cells(cells, normalized_data)
            
            # Detect headers
            header_rows = self._detect_header_rows(normalized_data)
            
            # Calculate confidence
            confidence = self._calculate_structure_confidence(cells, header_rows, normalized_data)
            
            # Create table structure
            structure = TableStructure(
                rows=rows,
                cols=cols,
                cells=cells,
                headers=header_rows,
                confidence=confidence,
                bbox=bbox,
                metadata={
                    'has_merged_cells': self._has_merged_cells(cells),
                    'header_count': len(header_rows),
                    'empty_cells': self._count_empty_cells(cells),
                    'data_types': self._analyze_data_types(normalized_data)
                }
            )
            
            return structure
            
        except Exception as e:
            logger.error(f"Table structure recognition failed: {e}")
            return TableStructure(0, 0, [], [], 0.0)
    
    def _normalize_table_data(self, table_data: List[List[str]], rows: int, cols: int) -> List[List[str]]:
        """Normalize table data to ensure consistent dimensions."""
        try:
            normalized = []
            
            for i in range(rows):
                if i < len(table_data):
                    row = table_data[i][:cols]  # Truncate if too long
                    while len(row) < cols:  # Pad if too short
                        row.append("")
                    normalized.append(row)
                else:
                    normalized.append([""] * cols)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Table data normalization failed: {e}")
            return table_data
    
    def _create_initial_cells(self, table_data: List[List[str]]) -> List[List[Optional[TableCell]]]:
        """Create initial cell structure."""
        try:
            rows = len(table_data)
            cols = len(table_data[0]) if table_data else 0
            
            cells = []
            
            for row_idx in range(rows):
                cell_row = []
                for col_idx in range(cols):
                    content = table_data[row_idx][col_idx].strip()
                    
                    # Determine cell type
                    cell_type = self._determine_cell_type(content, row_idx, col_idx, table_data)
                    
                    cell = TableCell(
                        row=row_idx,
                        col=col_idx,
                        content=content,
                        cell_type=cell_type,
                        confidence=self._calculate_cell_confidence(content, cell_type)
                    )
                    
                    cell_row.append(cell)
                
                cells.append(cell_row)
            
            return cells
            
        except Exception as e:
            logger.error(f"Initial cell creation failed: {e}")
            return []
    
    def _determine_cell_type(self, content: str, row: int, col: int, table_data: List[List[str]]) -> CellType:
        """Determine the type of a cell."""
        try:
            if not content:
                return CellType.EMPTY
            
            # Check if it's a header based on position and content
            if self._is_likely_header(content, row, col, table_data):
                return CellType.HEADER
            
            return CellType.DATA
            
        except Exception as e:
            logger.error(f"Cell type determination failed: {e}")
            return CellType.DATA
    
    def _is_likely_header(self, content: str, row: int, col: int, table_data: List[List[str]]) -> bool:
        """Check if a cell is likely a header."""
        try:
            # First row is often headers
            if row == 0:
                return True
            
            # Check header patterns
            for pattern in self.header_patterns:
                if re.match(pattern, content, re.IGNORECASE):
                    return True
            
            # Check header keywords
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in self.header_keywords):
                return True
            
            # Check if content is different from data pattern in column
            if self._is_column_header(content, col, table_data):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Header likelihood check failed: {e}")
            return False
    
    def _is_column_header(self, content: str, col: int, table_data: List[List[str]]) -> bool:
        """Check if content is a column header based on column data."""
        try:
            if len(table_data) < 3:  # Need at least 3 rows to analyze
                return False
            
            # Get column data (excluding first few rows)
            column_data = []
            for row_idx in range(2, len(table_data)):
                if col < len(table_data[row_idx]):
                    cell_content = table_data[row_idx][col].strip()
                    if cell_content:
                        column_data.append(cell_content)
            
            if not column_data:
                return False
            
            # Check if header content is different from data pattern
            content_lower = content.lower()
            
            # If column contains mostly numbers and header is text
            numeric_count = sum(1 for data in column_data if self._is_numeric(data))
            if numeric_count / len(column_data) > 0.7 and not self._is_numeric(content):
                return True
            
            # If header is much shorter/longer than typical data
            avg_data_length = sum(len(data) for data in column_data) / len(column_data)
            if len(content) < avg_data_length * 0.5 or len(content) > avg_data_length * 2:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Column header check failed: {e}")
            return False
    
    def _detect_merged_cells(self, cells: List[List[Optional[TableCell]]], 
                           table_data: List[List[str]]) -> List[List[Optional[TableCell]]]:
        """Detect merged cells and spanning elements."""
        try:
            rows = len(cells)
            cols = len(cells[0]) if cells else 0
            
            # Track processed cells
            processed = set()
            
            for row_idx in range(rows):
                for col_idx in range(cols):
                    if (row_idx, col_idx) in processed:
                        continue
                    
                    cell = cells[row_idx][col_idx]
                    if not cell or not cell.content:
                        continue
                    
                    # Check for horizontal spanning
                    colspan = self._detect_horizontal_span(table_data, row_idx, col_idx)
                    
                    # Check for vertical spanning
                    rowspan = self._detect_vertical_span(table_data, row_idx, col_idx)
                    
                    if colspan > 1 or rowspan > 1:
                        # Update main cell
                        cell.colspan = colspan
                        cell.rowspan = rowspan
                        cell.is_merged_source = True
                        cell.cell_type = CellType.MERGED
                        
                        # Mark spanned cells as merged
                        for r in range(row_idx, row_idx + rowspan):
                            for c in range(col_idx, col_idx + colspan):
                                if r != row_idx or c != col_idx:
                                    if r < rows and c < cols:
                                        spanned_cell = cells[r][c]
                                        if spanned_cell:
                                            spanned_cell.cell_type = CellType.MERGED
                                            spanned_cell.merged_from = (row_idx, col_idx)
                                            spanned_cell.content = ""  # Clear content of merged cells
                                        processed.add((r, c))
                    
                    processed.add((row_idx, col_idx))
            
            return cells
            
        except Exception as e:
            logger.error(f"Merged cell detection failed: {e}")
            return cells
    
    def _detect_horizontal_span(self, table_data: List[List[str]], row: int, col: int) -> int:
        """Detect horizontal cell spanning."""
        try:
            if row >= len(table_data) or col >= len(table_data[row]):
                return 1
            
            content = table_data[row][col].strip()
            if not content:
                return 1
            
            span = 1
            cols = len(table_data[row])
            
            # Look for empty cells to the right that might be part of this cell
            for next_col in range(col + 1, cols):
                next_content = table_data[row][next_col].strip()
                
                # If next cell is empty or contains continuation markers
                if not next_content or next_content in ['', '-', '—', '→', '...']:
                    span += 1
                else:
                    break
            
            # Also check if content is unusually long for a single cell
            if len(content) > 50 and span == 1:
                # Check if splitting the content makes sense
                words = content.split()
                if len(words) > 5:
                    # Might be spanning multiple cells
                    estimated_span = min(len(words) // 3, cols - col)
                    if estimated_span > 1:
                        span = estimated_span
            
            return span
            
        except Exception as e:
            logger.error(f"Horizontal span detection failed: {e}")
            return 1
    
    def _detect_vertical_span(self, table_data: List[List[str]], row: int, col: int) -> int:
        """Detect vertical cell spanning."""
        try:
            if row >= len(table_data):
                return 1
            
            content = table_data[row][col].strip()
            if not content:
                return 1
            
            span = 1
            rows = len(table_data)
            
            # Look for empty cells below that might be part of this cell
            for next_row in range(row + 1, rows):
                if col >= len(table_data[next_row]):
                    break
                
                next_content = table_data[next_row][col].strip()
                
                # If next cell is empty or contains continuation markers
                if not next_content or next_content in ['', '|', '↓', '...']:
                    span += 1
                else:
                    break
            
            return span
            
        except Exception as e:
            logger.error(f"Vertical span detection failed: {e}")
            return 1
    
    def _detect_header_rows(self, table_data: List[List[str]]) -> List[int]:
        """Detect which rows are headers."""
        try:
            if not table_data:
                return []
            
            header_rows = []
            rows = len(table_data)
            
            for row_idx in range(min(3, rows)):  # Check first 3 rows
                row_data = table_data[row_idx]
                
                # Count header-like cells in this row
                header_count = 0
                total_cells = len([cell for cell in row_data if cell.strip()])
                
                if total_cells == 0:
                    continue
                
                for col_idx, content in enumerate(row_data):
                    if self._is_likely_header(content, row_idx, col_idx, table_data):
                        header_count += 1
                
                # If majority of cells look like headers
                if header_count / total_cells > 0.6:
                    header_rows.append(row_idx)
            
            return header_rows
            
        except Exception as e:
            logger.error(f"Header row detection failed: {e}")
            return []
    
    def _calculate_cell_confidence(self, content: str, cell_type: CellType) -> float:
        """Calculate confidence score for a cell."""
        try:
            if not content:
                return 0.5  # Neutral confidence for empty cells
            
            confidence = 0.8  # Base confidence
            
            # Adjust based on content characteristics
            if cell_type == CellType.HEADER:
                # Headers with clear patterns get higher confidence
                for pattern in self.header_patterns:
                    if re.match(pattern, content, re.IGNORECASE):
                        confidence += 0.1
                        break
                
                # Headers with keywords get higher confidence
                if any(keyword in content.lower() for keyword in self.header_keywords):
                    confidence += 0.1
            
            elif cell_type == CellType.DATA:
                # Well-formatted data gets higher confidence
                if self._is_numeric(content) or self._is_date(content):
                    confidence += 0.1
            
            # Penalize very short or very long content
            if len(content) < 2:
                confidence -= 0.2
            elif len(content) > 100:
                confidence -= 0.1
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Cell confidence calculation failed: {e}")
            return 0.5
    
    def _calculate_structure_confidence(self, 
                                     cells: List[List[Optional[TableCell]]], 
                                     header_rows: List[int], 
                                     table_data: List[List[str]]) -> float:
        """Calculate overall structure confidence."""
        try:
            if not cells or not cells[0]:
                return 0.0
            
            total_confidence = 0.0
            cell_count = 0
            
            # Average cell confidence
            for row in cells:
                for cell in row:
                    if cell:
                        total_confidence += cell.confidence
                        cell_count += 1
            
            if cell_count == 0:
                return 0.0
            
            avg_cell_confidence = total_confidence / cell_count
            
            # Bonus for having headers
            header_bonus = 0.1 if header_rows else 0.0
            
            # Bonus for consistent structure
            structure_bonus = self._calculate_structure_consistency(table_data)
            
            # Penalty for too many empty cells
            empty_ratio = self._count_empty_cells(cells) / cell_count
            empty_penalty = empty_ratio * 0.3
            
            final_confidence = avg_cell_confidence + header_bonus + structure_bonus - empty_penalty
            
            return max(0.0, min(1.0, final_confidence))
            
        except Exception as e:
            logger.error(f"Structure confidence calculation failed: {e}")
            return 0.5
    
    def _calculate_structure_consistency(self, table_data: List[List[str]]) -> float:
        """Calculate how consistent the table structure is."""
        try:
            if not table_data or len(table_data) < 2:
                return 0.0
            
            # Check column consistency
            col_count = len(table_data[0])
            consistent_rows = sum(1 for row in table_data if len(row) == col_count)
            row_consistency = consistent_rows / len(table_data)
            
            # Check data type consistency within columns
            type_consistency = 0.0
            if col_count > 0:
                for col_idx in range(col_count):
                    column_data = []
                    for row in table_data[1:]:  # Skip header
                        if col_idx < len(row) and row[col_idx].strip():
                            column_data.append(row[col_idx].strip())
                    
                    if column_data:
                        # Check if column has consistent data types
                        numeric_count = sum(1 for data in column_data if self._is_numeric(data))
                        date_count = sum(1 for data in column_data if self._is_date(data))
                        
                        if numeric_count / len(column_data) > 0.8:
                            type_consistency += 1.0
                        elif date_count / len(column_data) > 0.8:
                            type_consistency += 1.0
                        else:
                            type_consistency += 0.5
                
                type_consistency /= col_count
            
            return (row_consistency + type_consistency) / 2 * 0.2
            
        except Exception as e:
            logger.error(f"Structure consistency calculation failed: {e}")
            return 0.0
    
    def _has_merged_cells(self, cells: List[List[Optional[TableCell]]]) -> bool:
        """Check if table has merged cells."""
        try:
            for row in cells:
                for cell in row:
                    if cell and (cell.colspan > 1 or cell.rowspan > 1):
                        return True
            return False
            
        except Exception as e:
            logger.error(f"Merged cells check failed: {e}")
            return False
    
    def _count_empty_cells(self, cells: List[List[Optional[TableCell]]]) -> int:
        """Count empty cells in the table."""
        try:
            count = 0
            for row in cells:
                for cell in row:
                    if not cell or cell.cell_type == CellType.EMPTY or not cell.content:
                        count += 1
            return count
            
        except Exception as e:
            logger.error(f"Empty cells counting failed: {e}")
            return 0
    
    def _analyze_data_types(self, table_data: List[List[str]]) -> Dict[str, int]:
        """Analyze data types in the table."""
        try:
            types = {'numeric': 0, 'date': 0, 'text': 0, 'empty': 0}
            
            for row in table_data:
                for cell in row:
                    content = cell.strip()
                    if not content:
                        types['empty'] += 1
                    elif self._is_numeric(content):
                        types['numeric'] += 1
                    elif self._is_date(content):
                        types['date'] += 1
                    else:
                        types['text'] += 1
            
            return types
            
        except Exception as e:
            logger.error(f"Data type analysis failed: {e}")
            return {'text': 1}
    
    def _is_numeric(self, text: str) -> bool:
        """Check if text represents a number."""
        try:
            # Remove common formatting
            cleaned = re.sub(r'[,\s$%]', '', text)
            
            # Try to parse as float
            try:
                float(cleaned)
                return True
            except ValueError:
                pass
            
            # Check for percentage
            if text.endswith('%'):
                try:
                    float(text[:-1])
                    return True
                except ValueError:
                    pass
            
            return False
            
        except Exception:
            return False
    
    def _is_date(self, text: str) -> bool:
        """Check if text represents a date."""
        try:
            import re
            
            # Common date patterns
            date_patterns = [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
                r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY/MM/DD
                r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}',  # DD Mon YYYY
                r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2,4}',  # Mon DD, YYYY
            ]
            
            for pattern in date_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return True
            
            return False
            
        except Exception:
            return False
    
    def validate_structure(self, structure: TableStructure) -> Dict[str, Any]:
        """Validate the recognized table structure."""
        try:
            validation_results = {
                'is_valid': True,
                'issues': [],
                'warnings': [],
                'confidence': structure.confidence
            }
            
            # Check basic structure
            if structure.rows == 0 or structure.cols == 0:
                validation_results['is_valid'] = False
                validation_results['issues'].append("Table has no rows or columns")
            
            # Check cell consistency
            if structure.cells:
                expected_cells = structure.rows * structure.cols
                actual_cells = sum(len(row) for row in structure.cells)
                
                if actual_cells != expected_cells:
                    validation_results['warnings'].append(
                        f"Cell count mismatch: expected {expected_cells}, got {actual_cells}"
                    )
            
            # Check merged cells validity
            for row_idx, row in enumerate(structure.cells):
                for col_idx, cell in enumerate(row):
                    if cell and cell.is_merged_source:
                        # Check if merged area is valid
                        if (row_idx + cell.rowspan > structure.rows or 
                            col_idx + cell.colspan > structure.cols):
                            validation_results['issues'].append(
                                f"Invalid merged cell at ({row_idx}, {col_idx}): "
                                f"spans beyond table boundaries"
                            )
            
            # Check confidence threshold
            if structure.confidence < self.confidence_threshold:
                validation_results['warnings'].append(
                    f"Low confidence score: {structure.confidence:.2f} < {self.confidence_threshold}"
                )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Structure validation failed: {e}")
            return {
                'is_valid': False,
                'issues': [f"Validation error: {str(e)}"],
                'warnings': [],
                'confidence': 0.0
            }


# Global instance
table_structure_recognizer = AdvancedTableStructureRecognizer()