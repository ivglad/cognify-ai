"""
Table-specific chunking strategy with structure preservation.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import trio

from app.services.chunking.strategies.base_chunker import BaseChunker
from app.services.nlp.rag_tokenizer import rag_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class TableStructure:
    """Represents a table structure."""
    headers: List[str]
    rows: List[List[str]]
    start_pos: int
    end_pos: int
    table_type: str  # 'markdown', 'html', 'csv', 'plain'
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TableChunkingStrategy(BaseChunker):
    """
    Chunking strategy for table-containing documents with structure preservation.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 100,
                 max_rows_per_chunk: int = 50,
                 min_table_confidence: float = 0.7):
        """
        Initialize table chunker.
        
        Args:
            chunk_size: Target size for chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
            max_rows_per_chunk: Maximum number of table rows per chunk
            min_table_confidence: Minimum confidence for table detection
        """
        super().__init__(chunk_size, chunk_overlap)
        self.max_rows_per_chunk = max_rows_per_chunk
        self.min_table_confidence = min_table_confidence
        self.tokenizer = rag_tokenizer
        
        # Table detection patterns
        self.table_patterns = [
            # Markdown tables
            {
                'pattern': re.compile(r'(?:^|\n)(\|.+\|)\s*\n\s*(\|[-:\s|]+\|)\s*\n((?:\|.+\|\s*\n?)+)', 
                                    re.MULTILINE),
                'type': 'markdown',
                'confidence': 0.9
            },
            # HTML tables
            {
                'pattern': re.compile(r'<table[^>]*>(.*?)</table>', 
                                    re.IGNORECASE | re.DOTALL),
                'type': 'html',
                'confidence': 0.95
            },
            # CSV-like format
            {
                'pattern': re.compile(r'(?:^|\n)((?:[^,\n]+,){2,}[^,\n]*\n){3,}', 
                                    re.MULTILINE),
                'type': 'csv',
                'confidence': 0.8
            },
            # Tab-separated format
            {
                'pattern': re.compile(r'(?:^|\n)((?:[^\t\n]+\t){2,}[^\t\n]*\n){3,}', 
                                    re.MULTILINE),
                'type': 'tsv',
                'confidence': 0.8
            },
            # Plain text tables with alignment
            {
                'pattern': re.compile(r'(?:^|\n)((?:\s*[^\n]+\s{3,}[^\n]+\s*\n){3,})', 
                                    re.MULTILINE),
                'type': 'plain_aligned',
                'confidence': 0.6
            }
        ]
    
    async def chunk_text(self, 
                        text: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk text with table-aware processing.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata for the text
            
        Returns:
            List of chunk dictionaries with table metadata
        """
        if not text or not text.strip():
            return []
        
        try:
            # Initialize tokenizer if needed
            if not self.tokenizer._initialized:
                await self.tokenizer.initialize()
            
            # Detect tables in text
            tables = await self._detect_tables(text)
            
            if not tables:
                # Fallback to simple chunking if no tables detected
                return await self._simple_table_chunk(text, metadata)
            
            # Create table-aware chunks
            chunks = await self._create_table_chunks(tables, text, metadata)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Table chunking failed: {e}")
            # Fallback to simple chunking
            return await self._simple_table_chunk(text, metadata)
    
    async def _detect_tables(self, text: str) -> List[TableStructure]:
        """
        Detect tables in text.
        
        Args:
            text: Input text
            
        Returns:
            List of detected tables
        """
        tables = []
        
        try:
            # Try each table pattern
            for pattern_info in self.table_patterns:
                pattern = pattern_info['pattern']
                table_type = pattern_info['type']
                base_confidence = pattern_info['confidence']
                
                matches = list(pattern.finditer(text))
                
                for match in matches:
                    table = await self._parse_table_match(match, table_type, base_confidence)
                    
                    if table and table.confidence >= self.min_table_confidence:
                        tables.append(table)
            
            # Remove overlapping tables (keep highest confidence)
            tables = self._remove_overlapping_tables(tables)
            
            # Sort by position
            tables.sort(key=lambda x: x.start_pos)
            
            logger.debug(f"Detected {len(tables)} tables")
            
            return tables
            
        except Exception as e:
            logger.error(f"Table detection failed: {e}")
            return []
    
    async def _parse_table_match(self, 
                                match: re.Match, 
                                table_type: str, 
                                base_confidence: float) -> Optional[TableStructure]:
        """Parse a table match into a TableStructure."""
        try:
            table_text = match.group(0)
            
            if table_type == 'markdown':
                return await self._parse_markdown_table(match, base_confidence)
            elif table_type == 'html':
                return await self._parse_html_table(match, base_confidence)
            elif table_type in ['csv', 'tsv']:
                return await self._parse_delimited_table(match, table_type, base_confidence)
            elif table_type == 'plain_aligned':
                return await self._parse_plain_table(match, base_confidence)
            
            return None
            
        except Exception as e:
            logger.error(f"Table parsing failed for type {table_type}: {e}")
            return None
    
    async def _parse_markdown_table(self, match: re.Match, base_confidence: float) -> Optional[TableStructure]:
        """Parse a markdown table."""
        try:
            header_line = match.group(1)
            separator_line = match.group(2)
            data_lines = match.group(3)
            
            # Parse headers
            headers = [cell.strip() for cell in header_line.split('|')[1:-1]]
            
            # Parse rows
            rows = []
            for line in data_lines.strip().split('\n'):
                if line.strip() and '|' in line:
                    row = [cell.strip() for cell in line.split('|')[1:-1]]
                    if len(row) == len(headers):  # Ensure consistent column count
                        rows.append(row)
            
            # Calculate confidence
            confidence = base_confidence
            
            # Bonus for consistent column count
            if all(len(row) == len(headers) for row in rows):
                confidence += 0.05
            
            # Bonus for reasonable table size
            if 2 <= len(rows) <= 100:
                confidence += 0.05
            
            table = TableStructure(
                headers=headers,
                rows=rows,
                start_pos=match.start(),
                end_pos=match.end(),
                table_type='markdown',
                confidence=confidence,
                metadata={
                    'column_count': len(headers),
                    'row_count': len(rows),
                    'has_separator': True
                }
            )
            
            return table
            
        except Exception as e:
            logger.error(f"Markdown table parsing failed: {e}")
            return None
    
    async def _parse_html_table(self, match: re.Match, base_confidence: float) -> Optional[TableStructure]:
        """Parse an HTML table."""
        try:
            table_html = match.group(1)
            
            # Extract headers
            header_pattern = re.compile(r'<th[^>]*>(.*?)</th>', re.IGNORECASE | re.DOTALL)
            header_matches = header_pattern.findall(table_html)
            headers = [re.sub(r'<[^>]+>', '', header).strip() for header in header_matches]
            
            # Extract rows
            row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.IGNORECASE | re.DOTALL)
            row_matches = row_pattern.findall(table_html)
            
            rows = []
            for row_html in row_matches:
                # Skip header rows
                if '<th' in row_html.lower():
                    continue
                
                cell_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.IGNORECASE | re.DOTALL)
                cell_matches = cell_pattern.findall(row_html)
                
                if cell_matches:
                    row = [re.sub(r'<[^>]+>', '', cell).strip() for cell in cell_matches]
                    rows.append(row)
            
            # If no explicit headers found, use first row
            if not headers and rows:
                headers = rows[0]
                rows = rows[1:]
            
            confidence = base_confidence
            
            # Bonus for consistent structure
            if headers and all(len(row) == len(headers) for row in rows):
                confidence += 0.05
            
            table = TableStructure(
                headers=headers,
                rows=rows,
                start_pos=match.start(),
                end_pos=match.end(),
                table_type='html',
                confidence=confidence,
                metadata={
                    'column_count': len(headers) if headers else 0,
                    'row_count': len(rows),
                    'has_html_tags': True
                }
            )
            
            return table
            
        except Exception as e:
            logger.error(f"HTML table parsing failed: {e}")
            return None
    
    async def _parse_delimited_table(self, 
                                   match: re.Match, 
                                   table_type: str, 
                                   base_confidence: float) -> Optional[TableStructure]:
        """Parse a delimited table (CSV/TSV)."""
        try:
            table_text = match.group(1).strip()
            delimiter = ',' if table_type == 'csv' else '\t'
            
            lines = table_text.split('\n')
            
            # Parse headers (first line)
            headers = [cell.strip().strip('"\'') for cell in lines[0].split(delimiter)]
            
            # Parse rows
            rows = []
            for line in lines[1:]:
                if line.strip():
                    row = [cell.strip().strip('"\'') for cell in line.split(delimiter)]
                    if len(row) == len(headers):  # Ensure consistent column count
                        rows.append(row)
            
            confidence = base_confidence
            
            # Bonus for consistent column count
            if all(len(row) == len(headers) for row in rows):
                confidence += 0.1
            
            # Penalty for very few rows (might not be a table)
            if len(rows) < 3:
                confidence -= 0.2
            
            table = TableStructure(
                headers=headers,
                rows=rows,
                start_pos=match.start(),
                end_pos=match.end(),
                table_type=table_type,
                confidence=confidence,
                metadata={
                    'column_count': len(headers),
                    'row_count': len(rows),
                    'delimiter': delimiter
                }
            )
            
            return table
            
        except Exception as e:
            logger.error(f"Delimited table parsing failed: {e}")
            return None
    
    async def _parse_plain_table(self, match: re.Match, base_confidence: float) -> Optional[TableStructure]:
        """Parse a plain text aligned table."""
        try:
            table_text = match.group(1).strip()
            lines = table_text.split('\n')
            
            # Try to detect column boundaries by finding consistent spacing
            column_positions = self._detect_column_positions(lines)
            
            if len(column_positions) < 2:
                return None
            
            # Extract headers and rows
            headers = []
            rows = []
            
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                
                row = self._extract_row_by_positions(line, column_positions)
                
                if i == 0:
                    headers = row
                else:
                    rows.append(row)
            
            confidence = base_confidence
            
            # Penalty for inconsistent structure
            if not all(len(row) == len(headers) for row in rows):
                confidence -= 0.2
            
            table = TableStructure(
                headers=headers,
                rows=rows,
                start_pos=match.start(),
                end_pos=match.end(),
                table_type='plain_aligned',
                confidence=confidence,
                metadata={
                    'column_count': len(headers),
                    'row_count': len(rows),
                    'column_positions': column_positions
                }
            )
            
            return table
            
        except Exception as e:
            logger.error(f"Plain table parsing failed: {e}")
            return None
    
    def _detect_column_positions(self, lines: List[str]) -> List[int]:
        """Detect column positions in aligned text."""
        try:
            if not lines:
                return []
            
            # Find positions where there are consistent spaces across lines
            max_length = max(len(line) for line in lines)
            space_counts = [0] * max_length
            
            for line in lines:
                for i, char in enumerate(line):
                    if char == ' ':
                        space_counts[i] += 1
            
            # Find positions with high space frequency
            threshold = len(lines) * 0.7  # 70% of lines should have space at this position
            column_positions = [0]  # Start of first column
            
            for i, count in enumerate(space_counts):
                if count >= threshold and i > column_positions[-1] + 2:  # Minimum 2 chars between columns
                    column_positions.append(i)
            
            return column_positions
            
        except Exception:
            return []
    
    def _extract_row_by_positions(self, line: str, positions: List[int]) -> List[str]:
        """Extract row cells based on column positions."""
        try:
            cells = []
            
            for i in range(len(positions)):
                start = positions[i]
                end = positions[i + 1] if i + 1 < len(positions) else len(line)
                
                cell = line[start:end].strip()
                cells.append(cell)
            
            return cells
            
        except Exception:
            return []
    
    def _remove_overlapping_tables(self, tables: List[TableStructure]) -> List[TableStructure]:
        """Remove overlapping tables, keeping highest confidence ones."""
        if not tables:
            return tables
        
        try:
            # Sort by confidence (descending)
            sorted_tables = sorted(tables, key=lambda x: x.confidence, reverse=True)
            
            non_overlapping = []
            
            for table in sorted_tables:
                # Check if this table overlaps with any already selected table
                overlaps = False
                
                for selected_table in non_overlapping:
                    if (table.start_pos < selected_table.end_pos and 
                        table.end_pos > selected_table.start_pos):
                        overlaps = True
                        break
                
                if not overlaps:
                    non_overlapping.append(table)
            
            return non_overlapping
            
        except Exception as e:
            logger.error(f"Table overlap removal failed: {e}")
            return tables
    
    async def _create_table_chunks(self, 
                                 tables: List[TableStructure], 
                                 full_text: str,
                                 metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create chunks from tables and surrounding text."""
        chunks = []
        
        try:
            current_pos = 0
            
            for i, table in enumerate(tables):
                # Add text before table as regular chunk
                if table.start_pos > current_pos:
                    before_text = full_text[current_pos:table.start_pos].strip()
                    if before_text:
                        text_chunks = await self._chunk_regular_text(before_text, metadata)
                        chunks.extend(text_chunks)
                
                # Add table chunks
                table_chunks = await self._chunk_table(table, metadata)
                chunks.extend(table_chunks)
                
                current_pos = table.end_pos
            
            # Add remaining text after last table
            if current_pos < len(full_text):
                after_text = full_text[current_pos:].strip()
                if after_text:
                    text_chunks = await self._chunk_regular_text(after_text, metadata)
                    chunks.extend(text_chunks)
            
            # Add sequence numbers and relationships
            for i, chunk in enumerate(chunks):
                chunk['chunk_index'] = i
                chunk['total_chunks'] = len(chunks)
                
                # Add navigation metadata
                if i > 0:
                    chunk['previous_chunk'] = chunks[i - 1]['chunk_id']
                if i < len(chunks) - 1:
                    chunk['next_chunk'] = chunks[i + 1]['chunk_id']
            
            return chunks
            
        except Exception as e:
            logger.error(f"Table chunk creation failed: {e}")
            return []
    
    async def _chunk_table(self, 
                         table: TableStructure, 
                         base_metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk a single table."""
        chunks = []
        
        try:
            # If table is small enough, create single chunk
            if len(table.rows) <= self.max_rows_per_chunk:
                chunk = await self._create_single_table_chunk(table, base_metadata)
                chunks.append(chunk)
            else:
                # Split table into multiple chunks
                table_chunks = await self._split_table(table, base_metadata)
                chunks.extend(table_chunks)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Table chunking failed: {e}")
            return []
    
    async def _create_single_table_chunk(self, 
                                       table: TableStructure, 
                                       base_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a single chunk from a table."""
        try:
            # Format table as markdown
            table_text = await self._format_table_as_markdown(table)
            
            # Calculate token count
            tokens = await self.tokenizer.tokenize_text(table_text, remove_stopwords=False, stem_words=False)
            token_count = len(tokens)
            
            # Create chunk metadata
            chunk_metadata = {
                'type': 'table',
                'table_type': table.table_type,
                'table_confidence': table.confidence,
                'column_count': len(table.headers),
                'row_count': len(table.rows),
                'headers': table.headers,
                'is_complete_table': True,
                'token_count': token_count,
                'char_count': len(table_text),
                'start_pos': table.start_pos,
                'end_pos': table.end_pos
            }
            
            # Add table-specific metadata
            chunk_metadata.update(table.metadata)
            
            # Merge with base metadata
            if base_metadata:
                chunk_metadata.update(base_metadata)
            
            # Create chunk
            chunk = {
                'chunk_id': f"table_{table.table_type}_{hash(table_text) % 10000}",
                'text': table_text,
                'metadata': chunk_metadata
            }
            
            return chunk
            
        except Exception as e:
            logger.error(f"Single table chunk creation failed: {e}")
            return {
                'chunk_id': f"table_error_{hash(str(table)) % 10000}",
                'text': str(table.headers) + str(table.rows),
                'metadata': {'type': 'table_error', 'error': str(e)}
            }
    
    async def _split_table(self, 
                         table: TableStructure, 
                         base_metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split a large table into multiple chunks."""
        chunks = []
        
        try:
            # Calculate chunk boundaries
            rows_per_chunk = min(self.max_rows_per_chunk, len(table.rows))
            
            for i in range(0, len(table.rows), rows_per_chunk):
                end_idx = min(i + rows_per_chunk, len(table.rows))
                chunk_rows = table.rows[i:end_idx]
                
                # Create sub-table
                sub_table = TableStructure(
                    headers=table.headers,
                    rows=chunk_rows,
                    start_pos=table.start_pos,
                    end_pos=table.end_pos,
                    table_type=table.table_type,
                    confidence=table.confidence,
                    metadata=table.metadata.copy()
                )
                
                # Format as markdown
                table_text = await self._format_table_as_markdown(sub_table)
                
                # Calculate token count
                tokens = await self.tokenizer.tokenize_text(table_text, remove_stopwords=False, stem_words=False)
                token_count = len(tokens)
                
                # Create chunk metadata
                chunk_metadata = {
                    'type': 'table_split',
                    'table_type': table.table_type,
                    'table_confidence': table.confidence,
                    'column_count': len(table.headers),
                    'row_count': len(chunk_rows),
                    'total_table_rows': len(table.rows),
                    'headers': table.headers,
                    'is_complete_table': False,
                    'table_chunk_index': i // rows_per_chunk,
                    'table_total_chunks': (len(table.rows) + rows_per_chunk - 1) // rows_per_chunk,
                    'row_start': i,
                    'row_end': end_idx,
                    'token_count': token_count,
                    'char_count': len(table_text)
                }
                
                # Add table-specific metadata
                chunk_metadata.update(table.metadata)
                
                # Merge with base metadata
                if base_metadata:
                    chunk_metadata.update(base_metadata)
                
                # Create chunk
                chunk = {
                    'chunk_id': f"table_split_{table.table_type}_{i}_{hash(table_text) % 10000}",
                    'text': table_text,
                    'metadata': chunk_metadata
                }
                
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Table splitting failed: {e}")
            return []
    
    async def _format_table_as_markdown(self, table: TableStructure) -> str:
        """Format table as markdown."""
        try:
            lines = []
            
            # Add headers
            if table.headers:
                header_line = "| " + " | ".join(table.headers) + " |"
                lines.append(header_line)
                
                # Add separator
                separator = "| " + " | ".join(["---"] * len(table.headers)) + " |"
                lines.append(separator)
            
            # Add rows
            for row in table.rows:
                # Ensure row has same number of columns as headers
                padded_row = row + [""] * (len(table.headers) - len(row)) if table.headers else row
                row_line = "| " + " | ".join(padded_row[:len(table.headers)] if table.headers else padded_row) + " |"
                lines.append(row_line)
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Table formatting failed: {e}")
            return str(table.headers) + "\n" + "\n".join([str(row) for row in table.rows])
    
    async def _chunk_regular_text(self, 
                                text: str, 
                                metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk regular text (non-table content)."""
        try:
            # Use base chunker for regular text
            chunks = await super().chunk_text(text, metadata)
            
            # Add table-aware metadata
            for chunk in chunks:
                chunk['metadata']['type'] = 'text_with_tables'
                chunk['metadata']['is_table'] = False
            
            return chunks
            
        except Exception as e:
            logger.error(f"Regular text chunking failed: {e}")
            return []
    
    async def _simple_table_chunk(self, 
                                text: str, 
                                metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback to simple chunking when no tables are detected."""
        try:
            # Use base chunker for simple splitting
            simple_chunks = await super().chunk_text(text, metadata)
            
            # Add table metadata
            for chunk in simple_chunks:
                chunk['metadata']['type'] = 'table_fallback'
                chunk['metadata']['table_detection_failed'] = True
                chunk['metadata']['is_table'] = False
            
            return simple_chunks
            
        except Exception as e:
            logger.error(f"Simple table chunking failed: {e}")
            return []
    
    async def get_chunker_stats(self) -> Dict[str, Any]:
        """Get table chunker statistics."""
        base_stats = await super().get_chunker_stats()
        
        table_stats = {
            'max_rows_per_chunk': self.max_rows_per_chunk,
            'min_table_confidence': self.min_table_confidence,
            'supported_table_types': len(self.table_patterns),
            'tokenizer_initialized': self.tokenizer._initialized if hasattr(self.tokenizer, '_initialized') else False
        }
        
        base_stats.update(table_stats)
        return base_stats