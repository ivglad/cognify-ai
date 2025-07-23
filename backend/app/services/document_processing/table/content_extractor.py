"""
Table content extraction and formatting with HTML and Markdown generation.
"""
import re
import html
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.services.document_processing.table.structure_recognizer import (
    TableStructure, TableCell, CellType, table_structure_recognizer
)
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class OutputFormat(str, Enum):
    """Output format enumeration."""
    HTML = "html"
    MARKDOWN = "markdown"
    CSV = "csv"
    JSON = "json"
    PLAIN_TEXT = "plain_text"


@dataclass
class FormattingOptions:
    """Table formatting options."""
    include_headers: bool = True
    preserve_formatting: bool = True
    escape_html: bool = True
    add_row_numbers: bool = False
    add_column_numbers: bool = False
    max_cell_length: Optional[int] = None
    empty_cell_placeholder: str = ""
    merge_cell_indicator: str = "↗"


@dataclass
class ExtractedTable:
    """Extracted table with multiple format representations."""
    structure: TableStructure
    html: str
    markdown: str
    csv: str
    json: Dict[str, Any]
    plain_text: str
    metadata: Dict[str, Any]


class TableContentExtractor:
    """Extract and format table content with proper structure preservation."""
    
    def __init__(self):
        self.structure_recognizer = table_structure_recognizer
    
    def extract_and_format(self, 
                          table_data: List[List[str]], 
                          bbox: Optional[List[float]] = None,
                          options: Optional[FormattingOptions] = None) -> ExtractedTable:
        """
        Extract table content and generate multiple format representations.
        
        Args:
            table_data: 2D list of cell contents
            bbox: Bounding box coordinates
            options: Formatting options
            
        Returns:
            ExtractedTable with all format representations
        """
        try:
            if options is None:
                options = FormattingOptions()
            
            # Recognize table structure
            structure = self.structure_recognizer.recognize_structure(table_data, bbox)
            
            # Generate different formats
            html_content = self._generate_html(structure, options)
            markdown_content = self._generate_markdown(structure, options)
            csv_content = self._generate_csv(structure, options)
            json_content = self._generate_json(structure, options)
            plain_text_content = self._generate_plain_text(structure, options)
            
            # Generate metadata
            metadata = self._generate_metadata(structure, table_data)
            
            return ExtractedTable(
                structure=structure,
                html=html_content,
                markdown=markdown_content,
                csv=csv_content,
                json=json_content,
                plain_text=plain_text_content,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Table extraction and formatting failed: {e}")
            return self._create_empty_table()
    
    def _generate_html(self, structure: TableStructure, options: FormattingOptions) -> str:
        """Generate HTML representation of the table."""
        try:
            if not structure.cells:
                return "<table></table>"
            
            html_parts = ['<table class="extracted-table">']
            
            # Add table header if headers exist
            if structure.headers and options.include_headers:
                html_parts.append('<thead>')
                for header_row_idx in structure.headers:
                    if header_row_idx < len(structure.cells):
                        html_parts.append('<tr>')
                        row = structure.cells[header_row_idx]
                        
                        for cell in row:
                            if cell and cell.cell_type != CellType.MERGED or cell.is_merged_source:
                                html_parts.append(self._format_html_cell(cell, 'th', options))
                        
                        html_parts.append('</tr>')
                html_parts.append('</thead>')
            
            # Add table body
            html_parts.append('<tbody>')
            
            for row_idx, row in enumerate(structure.cells):
                # Skip header rows if they're already processed
                if row_idx in structure.headers and options.include_headers:
                    continue
                
                html_parts.append('<tr>')
                
                # Add row number if requested
                if options.add_row_numbers:
                    html_parts.append(f'<td class="row-number">{row_idx + 1}</td>')
                
                for cell in row:
                    if cell and (cell.cell_type != CellType.MERGED or cell.is_merged_source):
                        html_parts.append(self._format_html_cell(cell, 'td', options))
                
                html_parts.append('</tr>')
            
            html_parts.append('</tbody>')
            html_parts.append('</table>')
            
            return '\n'.join(html_parts)
            
        except Exception as e:
            logger.error(f"HTML generation failed: {e}")
            return "<table><tr><td>Error generating HTML</td></tr></table>"
    
    def _format_html_cell(self, cell: TableCell, tag: str, options: FormattingOptions) -> str:
        """Format a single cell for HTML output."""
        try:
            content = cell.content
            
            # Escape HTML if requested
            if options.escape_html:
                content = html.escape(content)
            
            # Truncate if max length specified
            if options.max_cell_length and len(content) > options.max_cell_length:
                content = content[:options.max_cell_length] + "..."
            
            # Handle empty cells
            if not content:
                content = options.empty_cell_placeholder
            
            # Build cell attributes
            attributes = []
            
            if cell.colspan > 1:
                attributes.append(f'colspan="{cell.colspan}"')
            
            if cell.rowspan > 1:
                attributes.append(f'rowspan="{cell.rowspan}"')
            
            # Add CSS classes based on cell type
            css_classes = [f"cell-{cell.cell_type.value}"]
            if cell.is_merged_source:
                css_classes.append("merged-source")
            
            attributes.append(f'class="{" ".join(css_classes)}"')
            
            # Add confidence as data attribute
            attributes.append(f'data-confidence="{cell.confidence:.2f}"')
            
            attr_string = " " + " ".join(attributes) if attributes else ""
            
            return f'<{tag}{attr_string}>{content}</{tag}>'
            
        except Exception as e:
            logger.error(f"HTML cell formatting failed: {e}")
            return f'<{tag}>Error</{tag}>'
    
    def _generate_markdown(self, structure: TableStructure, options: FormattingOptions) -> str:
        """Generate Markdown representation of the table."""
        try:
            if not structure.cells:
                return ""
            
            markdown_parts = []
            
            # Process rows
            processed_header = False
            
            for row_idx, row in enumerate(structure.cells):
                row_parts = []
                
                # Add row number if requested
                if options.add_row_numbers:
                    row_parts.append(str(row_idx + 1))
                
                for cell in row:
                    if cell and (cell.cell_type != CellType.MERGED or cell.is_merged_source):
                        content = self._format_markdown_cell(cell, options)
                        
                        # Handle spanning cells in markdown (limited support)
                        if cell.colspan > 1:
                            content += f" {options.merge_cell_indicator}"
                        
                        row_parts.append(content)
                    elif cell and cell.cell_type == CellType.MERGED:
                        # Skip merged cells but maintain structure
                        continue
                
                # Create markdown row
                markdown_row = "| " + " | ".join(row_parts) + " |"
                markdown_parts.append(markdown_row)
                
                # Add header separator after first header row
                if (row_idx in structure.headers or 
                    (row_idx == 0 and not processed_header and options.include_headers)):
                    separator_parts = ["---"] * len(row_parts)
                    separator_row = "| " + " | ".join(separator_parts) + " |"
                    markdown_parts.append(separator_row)
                    processed_header = True
            
            return '\n'.join(markdown_parts)
            
        except Exception as e:
            logger.error(f"Markdown generation failed: {e}")
            return "Error generating Markdown"
    
    def _format_markdown_cell(self, cell: TableCell, options: FormattingOptions) -> str:
        """Format a single cell for Markdown output."""
        try:
            content = cell.content
            
            # Escape markdown special characters
            content = content.replace('|', '\\|')
            content = content.replace('\n', '<br>')
            
            # Truncate if max length specified
            if options.max_cell_length and len(content) > options.max_cell_length:
                content = content[:options.max_cell_length] + "..."
            
            # Handle empty cells
            if not content:
                content = options.empty_cell_placeholder
            
            # Add formatting for headers
            if cell.cell_type == CellType.HEADER:
                content = f"**{content}**"
            
            return content
            
        except Exception as e:
            logger.error(f"Markdown cell formatting failed: {e}")
            return "Error"
    
    def _generate_csv(self, structure: TableStructure, options: FormattingOptions) -> str:
        """Generate CSV representation of the table."""
        try:
            import csv
            import io
            
            if not structure.cells:
                return ""
            
            output = io.StringIO()
            writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
            
            for row_idx, row in enumerate(structure.cells):
                csv_row = []
                
                # Add row number if requested
                if options.add_row_numbers:
                    csv_row.append(str(row_idx + 1))
                
                for cell in row:
                    if cell and (cell.cell_type != CellType.MERGED or cell.is_merged_source):
                        content = cell.content
                        
                        # Truncate if max length specified
                        if options.max_cell_length and len(content) > options.max_cell_length:
                            content = content[:options.max_cell_length] + "..."
                        
                        # Handle empty cells
                        if not content:
                            content = options.empty_cell_placeholder
                        
                        csv_row.append(content)
                    elif cell and cell.cell_type == CellType.MERGED:
                        # For merged cells, add empty string to maintain structure
                        csv_row.append("")
                
                writer.writerow(csv_row)
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"CSV generation failed: {e}")
            return "Error generating CSV"
    
    def _generate_json(self, structure: TableStructure, options: FormattingOptions) -> Dict[str, Any]:
        """Generate JSON representation of the table."""
        try:
            if not structure.cells:
                return {"table": [], "metadata": {}}
            
            json_data = {
                "table": [],
                "headers": [],
                "metadata": {
                    "rows": structure.rows,
                    "cols": structure.cols,
                    "confidence": structure.confidence,
                    "has_merged_cells": any(
                        cell.colspan > 1 or cell.rowspan > 1 
                        for row in structure.cells 
                        for cell in row 
                        if cell
                    )
                }
            }
            
            # Extract headers
            if structure.headers and options.include_headers:
                for header_row_idx in structure.headers:
                    if header_row_idx < len(structure.cells):
                        header_row = []
                        for cell in structure.cells[header_row_idx]:
                            if cell and (cell.cell_type != CellType.MERGED or cell.is_merged_source):
                                header_row.append({
                                    "content": cell.content,
                                    "colspan": cell.colspan,
                                    "rowspan": cell.rowspan
                                })
                        json_data["headers"].append(header_row)
            
            # Extract data rows
            for row_idx, row in enumerate(structure.cells):
                # Skip header rows
                if row_idx in structure.headers and options.include_headers:
                    continue
                
                json_row = []
                
                for col_idx, cell in enumerate(row):
                    if cell:
                        cell_data = {
                            "content": cell.content,
                            "type": cell.cell_type.value,
                            "row": cell.row,
                            "col": cell.col,
                            "colspan": cell.colspan,
                            "rowspan": cell.rowspan,
                            "confidence": cell.confidence
                        }
                        
                        if cell.merged_from:
                            cell_data["merged_from"] = cell.merged_from
                        
                        json_row.append(cell_data)
                    else:
                        json_row.append(None)
                
                json_data["table"].append(json_row)
            
            return json_data
            
        except Exception as e:
            logger.error(f"JSON generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_plain_text(self, structure: TableStructure, options: FormattingOptions) -> str:
        """Generate plain text representation of the table."""
        try:
            if not structure.cells:
                return ""
            
            # Calculate column widths
            col_widths = self._calculate_column_widths(structure, options)
            
            text_parts = []
            
            for row_idx, row in enumerate(structure.cells):
                row_parts = []
                
                # Add row number if requested
                if options.add_row_numbers:
                    row_parts.append(f"{row_idx + 1:3}")
                
                for col_idx, cell in enumerate(row):
                    if cell and (cell.cell_type != CellType.MERGED or cell.is_merged_source):
                        content = cell.content
                        
                        # Truncate if max length specified
                        if options.max_cell_length and len(content) > options.max_cell_length:
                            content = content[:options.max_cell_length] + "..."
                        
                        # Handle empty cells
                        if not content:
                            content = options.empty_cell_placeholder
                        
                        # Pad content to column width
                        width = col_widths.get(col_idx, 10)
                        padded_content = content.ljust(width)[:width]
                        row_parts.append(padded_content)
                    elif col_idx < len(col_widths):
                        # Empty cell
                        width = col_widths.get(col_idx, 10)
                        row_parts.append(" " * width)
                
                text_parts.append(" | ".join(row_parts))
                
                # Add separator after headers
                if row_idx in structure.headers and options.include_headers:
                    separator_parts = ["-" * col_widths.get(i, 10) for i in range(len(row_parts))]
                    text_parts.append("-+-".join(separator_parts))
            
            return '\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"Plain text generation failed: {e}")
            return "Error generating plain text"
    
    def _calculate_column_widths(self, structure: TableStructure, options: FormattingOptions) -> Dict[int, int]:
        """Calculate optimal column widths for plain text formatting."""
        try:
            col_widths = {}
            
            for row in structure.cells:
                for col_idx, cell in enumerate(row):
                    if cell and (cell.cell_type != CellType.MERGED or cell.is_merged_source):
                        content_length = len(cell.content)
                        
                        # Apply max length limit
                        if options.max_cell_length:
                            content_length = min(content_length, options.max_cell_length)
                        
                        # Update column width
                        current_width = col_widths.get(col_idx, 0)
                        col_widths[col_idx] = max(current_width, content_length, 5)  # Minimum width of 5
            
            return col_widths
            
        except Exception as e:
            logger.error(f"Column width calculation failed: {e}")
            return {}
    
    def _generate_metadata(self, structure: TableStructure, table_data: List[List[str]]) -> Dict[str, Any]:
        """Generate metadata about the extracted table."""
        try:
            metadata = {
                "extraction_info": {
                    "rows": structure.rows,
                    "cols": structure.cols,
                    "confidence": structure.confidence,
                    "has_headers": len(structure.headers) > 0,
                    "header_rows": structure.headers,
                    "has_merged_cells": structure.metadata.get("has_merged_cells", False) if structure.metadata else False
                },
                "content_analysis": {
                    "total_cells": structure.rows * structure.cols,
                    "empty_cells": structure.metadata.get("empty_cells", 0) if structure.metadata else 0,
                    "data_types": structure.metadata.get("data_types", {}) if structure.metadata else {},
                    "avg_cell_length": self._calculate_avg_cell_length(table_data),
                    "max_cell_length": self._calculate_max_cell_length(table_data)
                },
                "quality_metrics": {
                    "structure_confidence": structure.confidence,
                    "completeness": self._calculate_completeness(structure),
                    "consistency": self._calculate_consistency(structure)
                }
            }
            
            # Add bounding box if available
            if structure.bbox:
                metadata["spatial_info"] = {
                    "bbox": structure.bbox,
                    "width": structure.bbox[2] - structure.bbox[0],
                    "height": structure.bbox[3] - structure.bbox[1]
                }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_avg_cell_length(self, table_data: List[List[str]]) -> float:
        """Calculate average cell content length."""
        try:
            total_length = 0
            cell_count = 0
            
            for row in table_data:
                for cell in row:
                    total_length += len(cell.strip())
                    cell_count += 1
            
            return total_length / cell_count if cell_count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Average cell length calculation failed: {e}")
            return 0.0
    
    def _calculate_max_cell_length(self, table_data: List[List[str]]) -> int:
        """Calculate maximum cell content length."""
        try:
            max_length = 0
            
            for row in table_data:
                for cell in row:
                    max_length = max(max_length, len(cell.strip()))
            
            return max_length
            
        except Exception as e:
            logger.error(f"Max cell length calculation failed: {e}")
            return 0
    
    def _calculate_completeness(self, structure: TableStructure) -> float:
        """Calculate table completeness (ratio of non-empty cells)."""
        try:
            if not structure.cells:
                return 0.0
            
            total_cells = 0
            non_empty_cells = 0
            
            for row in structure.cells:
                for cell in row:
                    total_cells += 1
                    if cell and cell.content.strip():
                        non_empty_cells += 1
            
            return non_empty_cells / total_cells if total_cells > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Completeness calculation failed: {e}")
            return 0.0
    
    def _calculate_consistency(self, structure: TableStructure) -> float:
        """Calculate table consistency score."""
        try:
            if not structure.cells or structure.rows < 2:
                return 1.0
            
            # Check row length consistency
            row_lengths = [len([c for c in row if c and c.content.strip()]) for row in structure.cells]
            avg_length = sum(row_lengths) / len(row_lengths)
            
            consistency_score = 1.0
            for length in row_lengths:
                if avg_length > 0:
                    deviation = abs(length - avg_length) / avg_length
                    consistency_score -= deviation * 0.1
            
            return max(0.0, min(1.0, consistency_score))
            
        except Exception as e:
            logger.error(f"Consistency calculation failed: {e}")
            return 0.5
    
    def _create_empty_table(self) -> ExtractedTable:
        """Create an empty table structure for error cases."""
        empty_structure = TableStructure(0, 0, [], [], 0.0)
        
        return ExtractedTable(
            structure=empty_structure,
            html="<table></table>",
            markdown="",
            csv="",
            json={"table": [], "metadata": {}},
            plain_text="",
            metadata={"error": "Failed to extract table"}
        )
    
    def extract_table_relationships(self, structure: TableStructure) -> Dict[str, Any]:
        """Extract relationships between table cells and structure."""
        try:
            relationships = {
                "header_data_mapping": {},
                "merged_cell_groups": [],
                "column_relationships": {},
                "row_relationships": {}
            }
            
            # Map headers to data columns
            if structure.headers:
                for header_row_idx in structure.headers:
                    if header_row_idx < len(structure.cells):
                        header_row = structure.cells[header_row_idx]
                        for col_idx, cell in enumerate(header_row):
                            if cell and cell.content.strip():
                                # Find data cells in this column
                                data_cells = []
                                for row_idx in range(len(structure.cells)):
                                    if row_idx not in structure.headers:
                                        if (col_idx < len(structure.cells[row_idx]) and 
                                            structure.cells[row_idx][col_idx] and
                                            structure.cells[row_idx][col_idx].content.strip()):
                                            data_cells.append({
                                                "row": row_idx,
                                                "col": col_idx,
                                                "content": structure.cells[row_idx][col_idx].content
                                            })
                                
                                relationships["header_data_mapping"][f"{header_row_idx}_{col_idx}"] = {
                                    "header": cell.content,
                                    "data_cells": data_cells
                                }
            
            # Find merged cell groups
            for row_idx, row in enumerate(structure.cells):
                for col_idx, cell in enumerate(row):
                    if cell and cell.is_merged_source:
                        merged_group = {
                            "source": {"row": row_idx, "col": col_idx},
                            "content": cell.content,
                            "spans": {"rows": cell.rowspan, "cols": cell.colspan},
                            "affected_cells": []
                        }
                        
                        # Find all cells affected by this merge
                        for r in range(row_idx, row_idx + cell.rowspan):
                            for c in range(col_idx, col_idx + cell.colspan):
                                if r != row_idx or c != col_idx:
                                    merged_group["affected_cells"].append({"row": r, "col": c})
                        
                        relationships["merged_cell_groups"].append(merged_group)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Table relationship extraction failed: {e}")
            return {}


# Global instance
table_content_extractor = TableContentExtractor()