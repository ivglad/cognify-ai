"""
Table data type detection and validation with automatic classification.
"""
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from app.services.document_processing.table.structure_recognizer import TableStructure, TableCell, CellType
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class DataType(str, Enum):
    """Data type enumeration."""
    INTEGER = "integer"
    FLOAT = "float"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    TEXT = "text"
    CATEGORICAL = "categorical"
    UNKNOWN = "unknown"


@dataclass
class DataTypeInfo:
    """Data type information."""
    data_type: DataType
    confidence: float
    pattern: Optional[str] = None
    format_info: Optional[Dict[str, Any]] = None
    validation_rules: Optional[Dict[str, Any]] = None


@dataclass
class ColumnAnalysis:
    """Column data type analysis."""
    column_index: int
    column_name: Optional[str]
    primary_type: DataTypeInfo
    secondary_types: List[DataTypeInfo]
    sample_values: List[str]
    statistics: Dict[str, Any]
    quality_metrics: Dict[str, Any]


@dataclass
class TableDataTypeAnalysis:
    """Complete table data type analysis."""
    columns: List[ColumnAnalysis]
    overall_quality: float
    type_distribution: Dict[str, int]
    recommendations: List[str]
    metadata: Dict[str, Any]


class TableDataTypeDetector:
    """Detect and classify data types in table columns with validation."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        
        # Regex patterns for different data types
        self.patterns = {
            DataType.INTEGER: [
                r'^-?\d+$',
                r'^-?\d{1,3}(,\d{3})*$'  # With thousand separators
            ],
            DataType.FLOAT: [
                r'^-?\d*\.\d+$',
                r'^-?\d+\.\d*$',
                r'^-?\d{1,3}(,\d{3})*\.\d+$',  # With thousand separators
                r'^-?\d+\.?\d*[eE][+-]?\d+$'   # Scientific notation
            ],
            DataType.CURRENCY: [
                r'^[\$€£¥₽]\s*-?\d+\.?\d*$',
                r'^-?\d+\.?\d*\s*[\$€£¥₽]$',
                r'^[\$€£¥₽]\s*-?\d{1,3}(,\d{3})*\.?\d*$',
                r'^\d+\.?\d*\s*(USD|EUR|GBP|JPY|RUB|dollars?|euros?|pounds?)$'
            ],
            DataType.PERCENTAGE: [
                r'^-?\d+\.?\d*\s*%$',
                r'^-?\d+\.?\d*\s*percent$',
                r'^-?\d+\.?\d*\s*pct$'
            ],
            DataType.DATE: [
                r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',  # MM/DD/YYYY or DD/MM/YYYY
                r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}$',    # YYYY/MM/DD
                r'^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}$',
                r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2,4}$',
                r'^\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}$'
            ],
            DataType.TIME: [
                r'^\d{1,2}:\d{2}$',
                r'^\d{1,2}:\d{2}:\d{2}$',
                r'^\d{1,2}:\d{2}\s*(AM|PM)$',
                r'^\d{1,2}:\d{2}:\d{2}\s*(AM|PM)$'
            ],
            DataType.DATETIME: [
                r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$',
                r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\d{2}$',
                r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$'
            ],
            DataType.BOOLEAN: [
                r'^(true|false)$',
                r'^(yes|no)$',
                r'^(y|n)$',
                r'^(1|0)$',
                r'^(on|off)$',
                r'^(enabled|disabled)$'
            ],
            DataType.EMAIL: [
                r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            ],
            DataType.URL: [
                r'^https?://[^\s/$.?#].[^\s]*$',
                r'^www\.[^\s/$.?#].[^\s]*$',
                r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[^\s]*)?$'
            ],
            DataType.PHONE: [
                r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$',  # US format
                r'^\+?[1-9]\d{1,14}$',  # International format
                r'^\(\d{3}\)\s?\d{3}-\d{4}$'  # (123) 456-7890
            ]
        }
        
        # Common categorical values
        self.categorical_indicators = {
            'status': ['active', 'inactive', 'pending', 'completed', 'cancelled'],
            'priority': ['low', 'medium', 'high', 'critical'],
            'grade': ['a', 'b', 'c', 'd', 'f'],
            'size': ['xs', 's', 'm', 'l', 'xl', 'xxl'],
            'gender': ['male', 'female', 'other'],
            'type': ['type1', 'type2', 'category1', 'category2']
        }
    
    def analyze_table_types(self, structure: TableStructure) -> TableDataTypeAnalysis:
        """
        Analyze data types for all columns in the table.
        
        Args:
            structure: Table structure with cells
            
        Returns:
            Complete table data type analysis
        """
        try:
            if not structure.cells:
                return self._create_empty_analysis()
            
            # Extract column data
            column_data = self._extract_column_data(structure)
            
            # Analyze each column
            column_analyses = []
            for col_idx, (column_name, values) in enumerate(column_data.items()):
                analysis = self._analyze_column(col_idx, column_name, values)
                column_analyses.append(analysis)
            
            # Calculate overall metrics
            overall_quality = self._calculate_overall_quality(column_analyses)
            type_distribution = self._calculate_type_distribution(column_analyses)
            recommendations = self._generate_recommendations(column_analyses)
            
            # Generate metadata
            metadata = {
                'total_columns': len(column_analyses),
                'analyzed_cells': sum(len(analysis.sample_values) for analysis in column_analyses),
                'confidence_threshold': self.confidence_threshold,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return TableDataTypeAnalysis(
                columns=column_analyses,
                overall_quality=overall_quality,
                type_distribution=type_distribution,
                recommendations=recommendations,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Table data type analysis failed: {e}")
            return self._create_empty_analysis()
    
    def _extract_column_data(self, structure: TableStructure) -> Dict[int, Tuple[Optional[str], List[str]]]:
        """Extract data from each column."""
        try:
            column_data = {}
            
            # Get column names from headers
            column_names = {}
            if structure.headers:
                for header_row_idx in structure.headers:
                    if header_row_idx < len(structure.cells):
                        header_row = structure.cells[header_row_idx]
                        for col_idx, cell in enumerate(header_row):
                            if cell and cell.content.strip():
                                column_names[col_idx] = cell.content.strip()
            
            # Extract data for each column
            for col_idx in range(structure.cols):
                column_values = []
                
                for row_idx, row in enumerate(structure.cells):
                    # Skip header rows
                    if row_idx in structure.headers:
                        continue
                    
                    if col_idx < len(row) and row[col_idx]:
                        cell = row[col_idx]
                        if cell.content.strip() and cell.cell_type != CellType.MERGED:
                            column_values.append(cell.content.strip())
                
                column_name = column_names.get(col_idx)
                column_data[col_idx] = (column_name, column_values)
            
            return column_data
            
        except Exception as e:
            logger.error(f"Column data extraction failed: {e}")
            return {}
    
    def _analyze_column(self, col_idx: int, column_name: Optional[str], values: List[str]) -> ColumnAnalysis:
        """Analyze data types for a single column."""
        try:
            if not values:
                return ColumnAnalysis(
                    column_index=col_idx,
                    column_name=column_name,
                    primary_type=DataTypeInfo(DataType.UNKNOWN, 0.0),
                    secondary_types=[],
                    sample_values=[],
                    statistics={},
                    quality_metrics={}
                )
            
            # Detect data types for each value
            type_scores = {data_type: 0 for data_type in DataType}
            type_matches = {data_type: [] for data_type in DataType}
            
            for value in values:
                detected_types = self._detect_value_types(value)
                for data_type, confidence in detected_types.items():
                    type_scores[data_type] += confidence
                    if confidence > 0.5:
                        type_matches[data_type].append(value)
            
            # Normalize scores
            total_values = len(values)
            for data_type in type_scores:
                type_scores[data_type] /= total_values
            
            # Determine primary and secondary types
            sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
            
            primary_type_name, primary_confidence = sorted_types[0]
            primary_type = DataTypeInfo(
                data_type=primary_type_name,
                confidence=primary_confidence,
                pattern=self._get_pattern_for_type(primary_type_name, type_matches[primary_type_name]),
                format_info=self._get_format_info(primary_type_name, type_matches[primary_type_name]),
                validation_rules=self._get_validation_rules(primary_type_name)
            )
            
            # Get secondary types (confidence > 0.1)
            secondary_types = []
            for data_type, confidence in sorted_types[1:]:
                if confidence > 0.1:
                    secondary_types.append(DataTypeInfo(
                        data_type=data_type,
                        confidence=confidence,
                        pattern=self._get_pattern_for_type(data_type, type_matches[data_type])
                    ))
            
            # Calculate statistics
            statistics = self._calculate_column_statistics(values, primary_type_name)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(values, primary_type, type_matches)
            
            # Get sample values (up to 5)
            sample_values = values[:5]
            
            return ColumnAnalysis(
                column_index=col_idx,
                column_name=column_name,
                primary_type=primary_type,
                secondary_types=secondary_types,
                sample_values=sample_values,
                statistics=statistics,
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            logger.error(f"Column analysis failed: {e}")
            return ColumnAnalysis(
                column_index=col_idx,
                column_name=column_name,
                primary_type=DataTypeInfo(DataType.UNKNOWN, 0.0),
                secondary_types=[],
                sample_values=[],
                statistics={},
                quality_metrics={}
            )
    
    def _detect_value_types(self, value: str) -> Dict[DataType, float]:
        """Detect possible data types for a single value."""
        try:
            type_confidences = {}
            
            # Check each data type pattern
            for data_type, patterns in self.patterns.items():
                confidence = 0.0
                
                for pattern in patterns:
                    if re.match(pattern, value, re.IGNORECASE):
                        confidence = 1.0
                        break
                
                if confidence > 0:
                    type_confidences[data_type] = confidence
            
            # Special handling for categorical data
            if not type_confidences:
                categorical_confidence = self._check_categorical(value)
                if categorical_confidence > 0:
                    type_confidences[DataType.CATEGORICAL] = categorical_confidence
            
            # Default to text if no other type matches
            if not type_confidences:
                type_confidences[DataType.TEXT] = 0.5
            
            return type_confidences
            
        except Exception as e:
            logger.error(f"Value type detection failed: {e}")
            return {DataType.TEXT: 0.5}
    
    def _check_categorical(self, value: str) -> float:
        """Check if value is likely categorical."""
        try:
            value_lower = value.lower()
            
            # Check against known categorical patterns
            for category, values in self.categorical_indicators.items():
                if value_lower in values:
                    return 0.9
            
            # Check for common categorical patterns
            if len(value) <= 20 and not any(char.isdigit() for char in value):
                # Short text without numbers might be categorical
                return 0.3
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Categorical check failed: {e}")
            return 0.0
    
    def _get_pattern_for_type(self, data_type: DataType, matches: List[str]) -> Optional[str]:
        """Get the most common pattern for a data type."""
        try:
            if not matches or data_type not in self.patterns:
                return None
            
            # Find which pattern matches most values
            pattern_counts = {}
            
            for pattern in self.patterns[data_type]:
                count = sum(1 for match in matches if re.match(pattern, match, re.IGNORECASE))
                if count > 0:
                    pattern_counts[pattern] = count
            
            if pattern_counts:
                return max(pattern_counts.items(), key=lambda x: x[1])[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return None
    
    def _get_format_info(self, data_type: DataType, matches: List[str]) -> Optional[Dict[str, Any]]:
        """Get format information for a data type."""
        try:
            if not matches:
                return None
            
            format_info = {}
            
            if data_type == DataType.DATE:
                # Analyze date formats
                formats = []
                for match in matches[:10]:  # Sample first 10
                    if re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', match):
                        formats.append('MM/DD/YYYY or DD/MM/YYYY')
                    elif re.match(r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}$', match):
                        formats.append('YYYY/MM/DD')
                
                format_info['common_formats'] = list(set(formats))
            
            elif data_type == DataType.CURRENCY:
                # Analyze currency formats
                currencies = []
                for match in matches[:10]:
                    if '$' in match:
                        currencies.append('USD')
                    elif '€' in match:
                        currencies.append('EUR')
                    elif '£' in match:
                        currencies.append('GBP')
                
                format_info['currencies'] = list(set(currencies))
            
            elif data_type in [DataType.INTEGER, DataType.FLOAT]:
                # Analyze numeric formats
                has_thousand_sep = any(',' in match for match in matches)
                format_info['has_thousand_separator'] = has_thousand_sep
                
                if data_type == DataType.FLOAT:
                    decimal_places = []
                    for match in matches:
                        if '.' in match:
                            decimal_places.append(len(match.split('.')[1]))
                    
                    if decimal_places:
                        format_info['avg_decimal_places'] = sum(decimal_places) / len(decimal_places)
            
            return format_info if format_info else None
            
        except Exception as e:
            logger.error(f"Format info generation failed: {e}")
            return None
    
    def _get_validation_rules(self, data_type: DataType) -> Optional[Dict[str, Any]]:
        """Get validation rules for a data type."""
        try:
            rules = {}
            
            if data_type == DataType.EMAIL:
                rules['pattern'] = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                rules['max_length'] = 254
            
            elif data_type == DataType.PHONE:
                rules['pattern'] = r'^\+?[1-9]\d{1,14}$'
                rules['min_length'] = 7
                rules['max_length'] = 15
            
            elif data_type == DataType.URL:
                rules['pattern'] = r'^https?://[^\s/$.?#].[^\s]*$'
                rules['max_length'] = 2048
            
            elif data_type == DataType.PERCENTAGE:
                rules['min_value'] = 0
                rules['max_value'] = 100
                rules['pattern'] = r'^-?\d+\.?\d*\s*%?$'
            
            return rules if rules else None
            
        except Exception as e:
            logger.error(f"Validation rules generation failed: {e}")
            return None
    
    def _calculate_column_statistics(self, values: List[str], data_type: DataType) -> Dict[str, Any]:
        """Calculate statistics for a column."""
        try:
            stats = {
                'total_values': len(values),
                'unique_values': len(set(values)),
                'null_count': 0,  # We don't have nulls in this context
                'avg_length': sum(len(v) for v in values) / len(values) if values else 0
            }
            
            # Type-specific statistics
            if data_type in [DataType.INTEGER, DataType.FLOAT, DataType.CURRENCY, DataType.PERCENTAGE]:
                numeric_values = []
                for value in values:
                    try:
                        # Clean numeric value
                        cleaned = re.sub(r'[^\d.-]', '', value)
                        if cleaned:
                            numeric_values.append(float(cleaned))
                    except ValueError:
                        continue
                
                if numeric_values:
                    stats.update({
                        'min_value': min(numeric_values),
                        'max_value': max(numeric_values),
                        'avg_value': sum(numeric_values) / len(numeric_values),
                        'numeric_count': len(numeric_values)
                    })
            
            elif data_type == DataType.CATEGORICAL:
                value_counts = {}
                for value in values:
                    value_counts[value] = value_counts.get(value, 0) + 1
                
                stats.update({
                    'categories': list(value_counts.keys()),
                    'category_count': len(value_counts),
                    'most_common': max(value_counts.items(), key=lambda x: x[1])[0] if value_counts else None
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Column statistics calculation failed: {e}")
            return {'total_values': len(values)}
    
    def _calculate_quality_metrics(self, values: List[str], primary_type: DataTypeInfo, type_matches: Dict[DataType, List[str]]) -> Dict[str, Any]:
        """Calculate quality metrics for a column."""
        try:
            total_values = len(values)
            if total_values == 0:
                return {}
            
            # Type consistency
            primary_matches = len(type_matches.get(primary_type.data_type, []))
            type_consistency = primary_matches / total_values
            
            # Completeness (non-empty values)
            non_empty_count = len([v for v in values if v.strip()])
            completeness = non_empty_count / total_values
            
            # Uniqueness
            unique_count = len(set(values))
            uniqueness = unique_count / total_values
            
            # Format consistency
            format_consistency = self._calculate_format_consistency(values, primary_type.data_type)
            
            return {
                'type_consistency': type_consistency,
                'completeness': completeness,
                'uniqueness': uniqueness,
                'format_consistency': format_consistency,
                'overall_quality': (type_consistency + completeness + format_consistency) / 3
            }
            
        except Exception as e:
            logger.error(f"Quality metrics calculation failed: {e}")
            return {}
    
    def _calculate_format_consistency(self, values: List[str], data_type: DataType) -> float:
        """Calculate format consistency for values of a specific type."""
        try:
            if data_type not in self.patterns or not values:
                return 1.0
            
            # Check how many values match the expected patterns
            consistent_count = 0
            
            for value in values:
                for pattern in self.patterns[data_type]:
                    if re.match(pattern, value, re.IGNORECASE):
                        consistent_count += 1
                        break
            
            return consistent_count / len(values)
            
        except Exception as e:
            logger.error(f"Format consistency calculation failed: {e}")
            return 0.5
    
    def _calculate_overall_quality(self, column_analyses: List[ColumnAnalysis]) -> float:
        """Calculate overall table quality score."""
        try:
            if not column_analyses:
                return 0.0
            
            quality_scores = []
            for analysis in column_analyses:
                if analysis.quality_metrics:
                    quality_scores.append(analysis.quality_metrics.get('overall_quality', 0.5))
                else:
                    quality_scores.append(0.5)
            
            return sum(quality_scores) / len(quality_scores)
            
        except Exception as e:
            logger.error(f"Overall quality calculation failed: {e}")
            return 0.5
    
    def _calculate_type_distribution(self, column_analyses: List[ColumnAnalysis]) -> Dict[str, int]:
        """Calculate distribution of data types across columns."""
        try:
            distribution = {}
            
            for analysis in column_analyses:
                data_type = analysis.primary_type.data_type.value
                distribution[data_type] = distribution.get(data_type, 0) + 1
            
            return distribution
            
        except Exception as e:
            logger.error(f"Type distribution calculation failed: {e}")
            return {}
    
    def _generate_recommendations(self, column_analyses: List[ColumnAnalysis]) -> List[str]:
        """Generate recommendations for improving data quality."""
        try:
            recommendations = []
            
            for analysis in column_analyses:
                col_name = analysis.column_name or f"Column {analysis.column_index}"
                
                # Low confidence recommendations
                if analysis.primary_type.confidence < self.confidence_threshold:
                    recommendations.append(
                        f"{col_name}: Low type detection confidence ({analysis.primary_type.confidence:.2f}). "
                        f"Consider data cleaning or validation."
                    )
                
                # Quality recommendations
                if analysis.quality_metrics:
                    quality = analysis.quality_metrics.get('overall_quality', 0)
                    if quality < 0.7:
                        recommendations.append(
                            f"{col_name}: Low data quality ({quality:.2f}). "
                            f"Check for inconsistent formatting or missing values."
                        )
                    
                    # Type consistency
                    type_consistency = analysis.quality_metrics.get('type_consistency', 1)
                    if type_consistency < 0.8:
                        recommendations.append(
                            f"{col_name}: Inconsistent data types ({type_consistency:.2f}). "
                            f"Some values don't match the expected {analysis.primary_type.data_type.value} format."
                        )
                    
                    # Format consistency
                    format_consistency = analysis.quality_metrics.get('format_consistency', 1)
                    if format_consistency < 0.8:
                        recommendations.append(
                            f"{col_name}: Inconsistent formatting ({format_consistency:.2f}). "
                            f"Consider standardizing the format."
                        )
            
            # Overall recommendations
            if len(recommendations) == 0:
                recommendations.append("Data quality looks good! No major issues detected.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            return ["Error generating recommendations"]
    
    def _create_empty_analysis(self) -> TableDataTypeAnalysis:
        """Create empty analysis for error cases."""
        return TableDataTypeAnalysis(
            columns=[],
            overall_quality=0.0,
            type_distribution={},
            recommendations=["No data to analyze"],
            metadata={"error": "Failed to analyze table"}
        )
    
    def validate_column_data(self, values: List[str], expected_type: DataType) -> Dict[str, Any]:
        """Validate column data against expected type."""
        try:
            validation_results = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'valid_count': 0,
                'invalid_count': 0
            }
            
            if expected_type not in self.patterns:
                validation_results['warnings'].append(f"No validation patterns for type {expected_type}")
                return validation_results
            
            patterns = self.patterns[expected_type]
            
            for i, value in enumerate(values):
                is_valid = False
                
                for pattern in patterns:
                    if re.match(pattern, value, re.IGNORECASE):
                        is_valid = True
                        break
                
                if is_valid:
                    validation_results['valid_count'] += 1
                else:
                    validation_results['invalid_count'] += 1
                    validation_results['errors'].append(f"Row {i + 1}: '{value}' doesn't match {expected_type} format")
            
            # Overall validation
            total_values = len(values)
            if total_values > 0:
                error_rate = validation_results['invalid_count'] / total_values
                if error_rate > 0.2:  # More than 20% errors
                    validation_results['is_valid'] = False
                elif error_rate > 0.1:  # More than 10% errors
                    validation_results['warnings'].append(f"High error rate: {error_rate:.1%}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Column data validation failed: {e}")
            return {
                'is_valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'valid_count': 0,
                'invalid_count': len(values)
            }


# Global instance
table_data_type_detector = TableDataTypeDetector()