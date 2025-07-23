"""
Advanced table chunking with structure preservation, header repetition, and logical row grouping.
"""
import math
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.services.document_processing.table.structure_recognizer import (
    TableStructure, TableCell, CellType, table_structure_recognizer
)
from app.services.document_processing.table.content_extractor import (
    table_content_extractor, TableContent
)
from app.services.document_processing.table.data_type_detector import (
    table_data_type_detector, DataType
)
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ChunkingStrategy(str, Enum):
    """Table chunking strategy enumeration."""
    ROW_BASED = "row_based"
    COLUMN_BASED = "column_based"
    SEMANTIC_GROUPS = "semantic_groups"
    SIZE_BASED = "size_based"
    HYBRID = "hybrid"


@dataclass
class ChunkingOptions:
    """Table chunking configuration options."""
    max_chunk_size: int = 1000  # Maximum tokens per chunk
    min_chunk_size: int = 100   # Minimum tokens per chunk
    overlap_rows: int = 1       # Number of overlapping rows between chunks
    preserve_headers: bool = True
    include_context: bool = True
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    max_rows_per_chunk: int = 20
    max_cols_per_chunk: int = 10
    semantic_grouping: bool = True
    cross_page_handling: bool = True


@dataclass
class TableChunk:
    """Table chunk with preserved structure."""
    chunk_id: str
    content: str
    html_content: str
    markdown_content: str
    structure: TableStructure
    metadata: Dict[str, Any]
    token_count: int
    row_range: Tuple[int, int]
    col_range: Tuple[int, int]
    has_headers: bool
    chunk_type: str
    relationships: Dict[str, Any]


class AdvancedTableChunker:
    """Advanced table chunking with intelligent structure preservation."""
    
    def __init__(self, options: Optional[ChunkingOptions] = None):
        self.options = options or ChunkingOptions()
        self.structure_recognizer = table_structure_recognizer
        self.content_extractor = table_content_extractor
        self.data_type_detector = table_data_type_detector
    
    def chunk_table(self, 
                   table_data: List[List[str]], 
                   bbox: Optional[List[float]] = None,
                   document_context: Optional[Dict[str, Any]] = None) -> List[TableChunk]:
        """
        Chunk table with structure preservation and intelligent splitting.
        
        Args:
            table_data: 2D list of cell contents
            bbox: Bounding box coordinates
            document_context: Additional document context
            
        Returns:
            List of table chunks with preserved structure
        """
        try:
            if not table_data or not table_data[0]:
                return []
            
            # Recognize table structure
            structure = self.structure_recognizer.recognize_structure(table_data, bbox)
            
            # Analyze data types
            type_analysis = self.data_type_detector.analyze_table_types(structure)
            
            # Determine optimal chunking strategy
            strategy = self._determine_chunking_strategy(structure, type_analysis)
            
            # Generate chunks based on strategy
            chunks = self._generate_chunks(structure, table_data, strategy, document_context)
            
            # Post-process chunks
            chunks = self._post_process_chunks(chunks, structure)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Table chunking failed: {e}")
            return []  
  
    def _determine_chunking_strategy(self, 
                                   structure: TableStructure, 
                                   type_analysis) -> ChunkingStrategy:
        """Determine optimal chunking strategy based on table characteristics."""
        try:
            # Analyze table characteristics
            total_cells = structure.rows * structure.cols
            has_merged_cells = structure.metadata.get("has_merged_cells", False) if structure.metadata else False
            
            # Small tables - keep whole
            if total_cells <= 50 and structure.rows <= 10:
                return ChunkingStrategy.SIZE_BASED
            
            # Wide tables with many columns - use column-based chunking
            if structure.cols > self.options.max_cols_per_chunk:
                return ChunkingStrategy.COLUMN_BASED
            
            # Long tables with many rows - use row-based chunking
            if structure.rows > self.options.max_rows_per_chunk:
                return ChunkingStrategy.ROW_BASED
            
            # Tables with semantic groups - use semantic chunking
            if self.options.semantic_grouping and self._has_semantic_groups(structure, type_analysis):
                return ChunkingStrategy.SEMANTIC_GROUPS
            
            # Default to hybrid approach
            return ChunkingStrategy.HYBRID
            
        except Exception as e:
            logger.error(f"Strategy determination failed: {e}")
            return ChunkingStrategy.HYBRID
    
    def _has_semantic_groups(self, structure: TableStructure, type_analysis) -> bool:
        """Check if table has semantic groups that can be used for chunking."""
        try:
            # Check for categorical columns that might indicate groupings
            if hasattr(type_analysis, 'columns'):
                for column in type_analysis.columns:
                    if (column.primary_type.data_type == DataType.CATEGORICAL and
                        column.statistics.get('category_count', 0) < structure.rows * 0.5):
                        return True
            
            # Check for repeated patterns in first column (often indicates groups)
            if structure.cells and len(structure.cells) > 3:
                first_col_values = []
                for row in structure.cells[1:]:  # Skip header
                    if row and row[0] and row[0].content.strip():
                        first_col_values.append(row[0].content.strip())
                
                if first_col_values:
                    unique_values = set(first_col_values)
                    if len(unique_values) < len(first_col_values) * 0.7:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Semantic groups check failed: {e}")
            return False
    
    def _generate_chunks(self, 
                        structure: TableStructure, 
                        table_data: List[List[str]], 
                        strategy: ChunkingStrategy,
                        document_context: Optional[Dict[str, Any]]) -> List[TableChunk]:
        """Generate chunks based on the selected strategy."""
        try:
            if strategy == ChunkingStrategy.ROW_BASED:
                return self._chunk_by_rows(structure, table_data, document_context)
            elif strategy == ChunkingStrategy.COLUMN_BASED:
                return self._chunk_by_columns(structure, table_data, document_context)
            elif strategy == ChunkingStrategy.SEMANTIC_GROUPS:
                return self._chunk_by_semantic_groups(structure, table_data, document_context)
            elif strategy == ChunkingStrategy.SIZE_BASED:
                return self._chunk_by_size(structure, table_data, document_context)
            else:  # HYBRID
                return self._chunk_hybrid(structure, table_data, document_context)
                
        except Exception as e:
            logger.error(f"Chunk generation failed: {e}")
            return []
    
    def _chunk_by_rows(self, 
                      structure: TableStructure, 
                      table_data: List[List[str]], 
                      document_context: Optional[Dict[str, Any]]) -> List[TableChunk]:
        """Chunk table by rows with header preservation."""
        try:
            chunks = []
            chunk_id = 0
            
            # Get header rows
            header_rows = []
            if self.options.preserve_headers and structure.headers:
                for header_idx in structure.headers:
                    if header_idx < len(table_data):
                        header_rows.append(table_data[header_idx])
            
            # Calculate rows per chunk
            data_rows = [i for i in range(len(table_data)) if i not in structure.headers]
            
            if not data_rows:
                return chunks
            
            rows_per_chunk = min(self.options.max_rows_per_chunk, 
                                max(1, len(data_rows) // max(1, len(data_rows) // self.options.max_rows_per_chunk)))
            
            # Create chunks
            for start_idx in range(0, len(data_rows), rows_per_chunk - self.options.overlap_rows):
                end_idx = min(start_idx + rows_per_chunk, len(data_rows))
                
                if start_idx >= end_idx:
                    break
                
                # Build chunk data
                chunk_data = []
                
                # Add headers
                chunk_data.extend(header_rows)
                
                # Add data rows
                for row_idx in range(start_idx, end_idx):
                    if row_idx < len(data_rows):
                        actual_row_idx = data_rows[row_idx]
                        if actual_row_idx < len(table_data):
                            chunk_data.append(table_data[actual_row_idx])
                
                # Create chunk
                chunk = self._create_chunk(
                    chunk_id=f"row_chunk_{chunk_id}",
                    chunk_data=chunk_data,
                    row_range=(data_rows[start_idx], data_rows[end_idx - 1]),
                    col_range=(0, structure.cols - 1),
                    chunk_type="row_based",
                    has_headers=len(header_rows) > 0,
                    document_context=document_context
                )
                
                if chunk:
                    chunks.append(chunk)
                    chunk_id += 1
                
                # Break if we've reached the end
                if end_idx >= len(data_rows):
                    break
            
            return chunks
            
        except Exception as e:
            logger.error(f"Row-based chunking failed: {e}")
            return [] 
   
    def _chunk_by_columns(self, 
                         structure: TableStructure, 
                         table_data: List[List[str]], 
                         document_context: Optional[Dict[str, Any]]) -> List[TableChunk]:
        """Chunk table by columns for wide tables."""
        try:
            chunks = []
            chunk_id = 0
            
            cols_per_chunk = self.options.max_cols_per_chunk
            
            for start_col in range(0, structure.cols, cols_per_chunk):
                end_col = min(start_col + cols_per_chunk, structure.cols)
                
                # Build chunk data with selected columns
                chunk_data = []
                for row in table_data:
                    chunk_row = row[start_col:end_col]
                    chunk_data.append(chunk_row)
                
                # Create chunk
                chunk = self._create_chunk(
                    chunk_id=f"col_chunk_{chunk_id}",
                    chunk_data=chunk_data,
                    row_range=(0, structure.rows - 1),
                    col_range=(start_col, end_col - 1),
                    chunk_type="column_based",
                    has_headers=len(structure.headers) > 0,
                    document_context=document_context
                )
                
                if chunk:
                    chunks.append(chunk)
                    chunk_id += 1
            
            return chunks
            
        except Exception as e:
            logger.error(f"Column-based chunking failed: {e}")
            return []
    
    def _chunk_by_semantic_groups(self, 
                                 structure: TableStructure, 
                                 table_data: List[List[str]], 
                                 document_context: Optional[Dict[str, Any]]) -> List[TableChunk]:
        """Chunk table by semantic groups."""
        try:
            chunks = []
            chunk_id = 0
            
            # Get header rows
            header_rows = []
            if self.options.preserve_headers and structure.headers:
                for header_idx in structure.headers:
                    if header_idx < len(table_data):
                        header_rows.append(table_data[header_idx])
            
            # Find semantic groups based on first column values
            groups = self._identify_semantic_groups(table_data, structure.headers)
            
            for group_name, row_indices in groups.items():
                # Build chunk data
                chunk_data = []
                
                # Add headers
                chunk_data.extend(header_rows)
                
                # Add group rows
                for row_idx in row_indices:
                    if row_idx < len(table_data):
                        chunk_data.append(table_data[row_idx])
                
                # Create chunk
                chunk = self._create_chunk(
                    chunk_id=f"semantic_chunk_{chunk_id}_{group_name}",
                    chunk_data=chunk_data,
                    row_range=(min(row_indices), max(row_indices)),
                    col_range=(0, structure.cols - 1),
                    chunk_type="semantic_group",
                    has_headers=len(header_rows) > 0,
                    document_context=document_context
                )
                
                if chunk:
                    chunk.metadata['group_name'] = group_name
                    chunks.append(chunk)
                    chunk_id += 1
            
            return chunks
            
        except Exception as e:
            logger.error(f"Semantic group chunking failed: {e}")
            return []
    
    def _chunk_by_size(self, 
                      structure: TableStructure, 
                      table_data: List[List[str]], 
                      document_context: Optional[Dict[str, Any]]) -> List[TableChunk]:
        """Chunk table based on size constraints."""
        try:
            # For small tables, create a single chunk
            chunk = self._create_chunk(
                chunk_id="size_chunk_0",
                chunk_data=table_data,
                row_range=(0, structure.rows - 1),
                col_range=(0, structure.cols - 1),
                chunk_type="size_based",
                has_headers=len(structure.headers) > 0,
                document_context=document_context
            )
            
            return [chunk] if chunk else []
            
        except Exception as e:
            logger.error(f"Size-based chunking failed: {e}")
            return []
    
    def _chunk_hybrid(self, 
                     structure: TableStructure, 
                     table_data: List[List[str]], 
                     document_context: Optional[Dict[str, Any]]) -> List[TableChunk]:
        """Hybrid chunking approach combining multiple strategies."""
        try:
            # Start with row-based chunking
            chunks = self._chunk_by_rows(structure, table_data, document_context)
            
            # If chunks are too large, further split them
            refined_chunks = []
            for chunk in chunks:
                if chunk.token_count > self.options.max_chunk_size:
                    # Split large chunks by columns if needed
                    sub_chunks = self._split_large_chunk(chunk, structure)
                    refined_chunks.extend(sub_chunks)
                else:
                    refined_chunks.append(chunk)
            
            return refined_chunks
            
        except Exception as e:
            logger.error(f"Hybrid chunking failed: {e}")
            return []
    
    def _identify_semantic_groups(self, 
                                 table_data: List[List[str]], 
                                 header_indices: List[int]) -> Dict[str, List[int]]:
        """Identify semantic groups in table data."""
        try:
            groups = {}
            
            # Skip header rows
            data_rows = [i for i in range(len(table_data)) if i not in header_indices]
            
            if not data_rows or not table_data[0]:
                return groups
            
            # Group by first column values
            for row_idx in data_rows:
                if row_idx < len(table_data) and table_data[row_idx]:
                    first_cell = table_data[row_idx][0].strip()
                    
                    if first_cell:
                        if first_cell not in groups:
                            groups[first_cell] = []
                        groups[first_cell].append(row_idx)
            
            # Filter out single-item groups unless they're significant
            filtered_groups = {}
            for group_name, indices in groups.items():
                if len(indices) > 1 or len(groups) <= 3:
                    filtered_groups[group_name] = indices
            
            return filtered_groups
            
        except Exception as e:
            logger.error(f"Semantic group identification failed: {e}")
            return {}
    
    def _create_chunk(self, 
                     chunk_id: str, 
                     chunk_data: List[List[str]], 
                     row_range: Tuple[int, int], 
                     col_range: Tuple[int, int],
                     chunk_type: str, 
                     has_headers: bool,
                     document_context: Optional[Dict[str, Any]]) -> Optional[TableChunk]:
        """Create a table chunk with all necessary metadata."""
        try:
            if not chunk_data:
                return None
            
            # Extract table content using the content extractor
            # Create a simple table structure for the chunk
            chunk_structure = TableStructure(
                rows=len(chunk_data),
                cols=len(chunk_data[0]) if chunk_data else 0,
                cells=[],
                headers=[0] if has_headers else [],
                confidence=0.9,
                metadata={'chunk_type': chunk_type}
            )
            
            # Extract content using the table content extractor
            extracted_table = self.content_extractor.extract_content(chunk_structure, chunk_data)
            
            # Calculate token count (approximate)
            token_count = self._estimate_token_count(extracted_table.content if hasattr(extracted_table, 'content') else str(chunk_data))
            
            # Build metadata
            metadata = {
                'chunk_type': chunk_type,
                'row_count': len(chunk_data),
                'col_count': len(chunk_data[0]) if chunk_data else 0,
                'has_headers': has_headers,
                'extraction_confidence': chunk_structure.confidence,
                'data_types': extracted_table.metadata.get('data_types', {}) if hasattr(extracted_table, 'metadata') else {},
                'creation_timestamp': datetime.now().isoformat()
            }
            
            # Add document context if available
            if document_context:
                metadata['document_context'] = document_context
            
            # Build relationships
            relationships = self._build_chunk_relationships(chunk_data, row_range, col_range)
            
            # Get content from extracted table
            content = extracted_table.content if hasattr(extracted_table, 'content') else str(chunk_data)
            html_content = extracted_table.html if hasattr(extracted_table, 'html') else self._generate_html_table(chunk_data)
            markdown_content = extracted_table.markdown if hasattr(extracted_table, 'markdown') else self._generate_markdown_table(chunk_data)
            
            return TableChunk(
                chunk_id=chunk_id,
                content=content,
                html_content=html_content,
                markdown_content=markdown_content,
                structure=chunk_structure,
                metadata=metadata,
                token_count=token_count,
                row_range=row_range,
                col_range=col_range,
                has_headers=has_headers,
                chunk_type=chunk_type,
                relationships=relationships
            )
            
        except Exception as e:
            logger.error(f"Chunk creation failed: {e}")
            return None    
   
    def _split_large_chunk(self, chunk: TableChunk, structure: TableStructure) -> List[TableChunk]:
        """Split a large chunk into smaller ones."""
        try:
            # If chunk is too large, split by columns
            if chunk.token_count > self.options.max_chunk_size:
                # Extract original data from chunk structure
                chunk_data = []
                for row in chunk.structure.cells:
                    row_data = []
                    for cell in row:
                        if cell:
                            row_data.append(cell.content)
                        else:
                            row_data.append("")
                    chunk_data.append(row_data)
                
                # Split by columns
                sub_chunks = []
                cols_per_chunk = max(1, chunk.structure.cols // 2)
                
                for start_col in range(0, chunk.structure.cols, cols_per_chunk):
                    end_col = min(start_col + cols_per_chunk, chunk.structure.cols)
                    
                    sub_chunk_data = []
                    for row in chunk_data:
                        sub_chunk_data.append(row[start_col:end_col])
                    
                    sub_chunk = self._create_chunk(
                        chunk_id=f"{chunk.chunk_id}_split_{start_col}",
                        chunk_data=sub_chunk_data,
                        row_range=chunk.row_range,
                        col_range=(start_col, end_col - 1),
                        chunk_type=f"{chunk.chunk_type}_split",
                        has_headers=chunk.has_headers,
                        document_context=chunk.metadata.get('document_context')
                    )
                    
                    if sub_chunk:
                        sub_chunks.append(sub_chunk)
                
                return sub_chunks
            
            return [chunk]
            
        except Exception as e:
            logger.error(f"Large chunk splitting failed: {e}")
            return [chunk]
    
    def _build_chunk_relationships(self, 
                                  chunk_data: List[List[str]], 
                                  row_range: Tuple[int, int], 
                                  col_range: Tuple[int, int]) -> Dict[str, Any]:
        """Build relationships between chunk elements."""
        try:
            relationships = {
                'position': {
                    'row_start': row_range[0],
                    'row_end': row_range[1],
                    'col_start': col_range[0],
                    'col_end': col_range[1]
                },
                'adjacent_chunks': [],
                'header_mapping': {},
                'data_flow': []
            }
            
            # Analyze data flow within chunk
            if len(chunk_data) > 1:
                for row_idx in range(1, len(chunk_data)):
                    for col_idx in range(len(chunk_data[row_idx])):
                        if (col_idx < len(chunk_data[0]) and 
                            chunk_data[0][col_idx].strip() and 
                            chunk_data[row_idx][col_idx].strip()):
                            
                            relationships['data_flow'].append({
                                'header': chunk_data[0][col_idx],
                                'value': chunk_data[row_idx][col_idx],
                                'position': (row_idx, col_idx)
                            })
            
            return relationships
            
        except Exception as e:
            logger.error(f"Relationship building failed: {e}")
            return {}
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text."""
        try:
            # Simple estimation: ~4 characters per token
            return max(1, len(text) // 4)
            
        except Exception as e:
            logger.error(f"Token count estimation failed: {e}")
            return len(text.split())
    
    def _post_process_chunks(self, chunks: List[TableChunk], structure: TableStructure) -> List[TableChunk]:
        """Post-process chunks to add cross-references and optimize."""
        try:
            if not chunks:
                return chunks
            
            # Add cross-references between chunks
            for i, chunk in enumerate(chunks):
                # Add adjacent chunk references
                if i > 0:
                    chunk.relationships['adjacent_chunks'].append({
                        'type': 'previous',
                        'chunk_id': chunks[i-1].chunk_id
                    })
                
                if i < len(chunks) - 1:
                    chunk.relationships['adjacent_chunks'].append({
                        'type': 'next',
                        'chunk_id': chunks[i+1].chunk_id
                    })
            
            # Filter out chunks that are too small
            filtered_chunks = []
            for chunk in chunks:
                if chunk.token_count >= self.options.min_chunk_size or len(chunks) == 1:
                    filtered_chunks.append(chunk)
                else:
                    # Try to merge with adjacent chunk
                    if filtered_chunks:
                        last_chunk = filtered_chunks[-1]
                        if last_chunk.token_count + chunk.token_count <= self.options.max_chunk_size:
                            # Merge chunks (simplified)
                            last_chunk.content += "\n\n" + chunk.content
                            last_chunk.token_count += chunk.token_count
                            last_chunk.row_range = (last_chunk.row_range[0], chunk.row_range[1])
                        else:
                            filtered_chunks.append(chunk)
                    else:
                        filtered_chunks.append(chunk)
            
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"Chunk post-processing failed: {e}")
            return chunks
    
    def handle_cross_page_tables(self, 
                                table_chunks: List[List[List[str]]], 
                                page_contexts: List[Dict[str, Any]]) -> List[TableChunk]:
        """Handle tables that span across multiple pages."""
        try:
            if not self.options.cross_page_handling or len(table_chunks) <= 1:
                # Process each page separately
                all_chunks = []
                for i, page_table in enumerate(table_chunks):
                    page_context = page_contexts[i] if i < len(page_contexts) else {}
                    chunks = self.chunk_table(page_table, document_context=page_context)
                    all_chunks.extend(chunks)
                return all_chunks
            
            # Combine tables from multiple pages
            combined_table = []
            combined_context = {'pages': page_contexts, 'cross_page': True}
            
            # Add first page completely
            if table_chunks[0]:
                combined_table.extend(table_chunks[0])
            
            # Add subsequent pages (skip headers if they match)
            for i in range(1, len(table_chunks)):
                page_table = table_chunks[i]
                if not page_table:
                    continue
                
                # Check if first row is a header (similar to previous page)
                start_row = 0
                if (len(page_table) > 0 and len(combined_table) > 0 and
                    self._is_similar_header(page_table[0], combined_table[0])):
                    start_row = 1
                
                # Add data rows
                for row_idx in range(start_row, len(page_table)):
                    combined_table.append(page_table[row_idx])
            
            # Chunk the combined table
            return self.chunk_table(combined_table, document_context=combined_context)
            
        except Exception as e:
            logger.error(f"Cross-page table handling failed: {e}")
            # Fallback to processing pages separately
            all_chunks = []
            for i, page_table in enumerate(table_chunks):
                page_context = page_contexts[i] if i < len(page_contexts) else {}
                chunks = self.chunk_table(page_table, document_context=page_context)
                all_chunks.extend(chunks)
            return all_chunks
    
    def _generate_html_table(self, table_data: List[List[str]]) -> str:
        """Generate HTML representation of table data."""
        try:
            if not table_data:
                return "<table></table>"
            
            html = ["<table border='1' style='border-collapse: collapse;'>"]
            
            for i, row in enumerate(table_data):
                html.append("<tr>")
                tag = "th" if i == 0 else "td"
                for cell in row:
                    escaped_cell = str(cell).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    html.append(f"<{tag}>{escaped_cell}</{tag}>")
                html.append("</tr>")
            
            html.append("</table>")
            return "".join(html)
            
        except Exception as e:
            logger.error(f"HTML table generation failed: {e}")
            return "<table><tr><td>Error generating HTML</td></tr></table>"
    
    def _generate_markdown_table(self, table_data: List[List[str]]) -> str:
        """Generate Markdown representation of table data."""
        try:
            if not table_data:
                return "| No data |\n|---|\n"
            
            lines = []
            
            # Add header row
            if table_data:
                header_cells = [str(cell).replace("|", "\\|") for cell in table_data[0]]
                lines.append("| " + " | ".join(header_cells) + " |")
                
                # Add separator
                separator = ["---" for _ in header_cells]
                lines.append("| " + " | ".join(separator) + " |")
                
                # Add data rows
                for row in table_data[1:]:
                    data_cells = [str(cell).replace("|", "\\|") for cell in row]
                    # Pad row to match header length
                    while len(data_cells) < len(header_cells):
                        data_cells.append("")
                    lines.append("| " + " | ".join(data_cells[:len(header_cells)]) + " |")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Markdown table generation failed: {e}")
            return "| Error generating Markdown |\n|---|\n"

    def _is_similar_header(self, row1: List[str], row2: List[str]) -> bool:
        """Check if two rows are similar headers."""
        try:
            if len(row1) != len(row2):
                return False
            
            matches = 0
            for cell1, cell2 in zip(row1, row2):
                if cell1.strip().lower() == cell2.strip().lower():
                    matches += 1
            
            return matches / len(row1) > 0.8
            
        except Exception as e:
            logger.error(f"Header similarity check failed: {e}")
            return False


# Global instance
advanced_table_chunker = AdvancedTableChunker()