"""
Performance optimization utilities for geometric processing and text analysis.
"""
import math
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import pyclipper

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class GeometricProcessor:
    """Geometric operations for PDF processing using pyclipper."""
    
    @staticmethod
    def unclip_polygon(box: List[float], unclip_ratio: float = 1.5) -> np.ndarray:
        """
        Expand polygon using pyclipper for better text detection.
        
        Args:
            box: List of coordinates [x0, y0, x1, y1, x2, y2, x3, y3] or [(x0,y0), (x1,y1), ...]
            unclip_ratio: Expansion ratio
            
        Returns:
            Expanded polygon coordinates as numpy array
        """
        try:
            # Convert box to polygon points
            if len(box) == 4:
                # Convert [x0, y0, x1, y1] to polygon points
                x0, y0, x1, y1 = box
                points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            elif len(box) == 8:
                # Convert [x0, y0, x1, y1, x2, y2, x3, y3] to polygon points
                points = [(box[i], box[i+1]) for i in range(0, 8, 2)]
            else:
                # Assume it's already in point format
                points = box
            
            # Create Shapely polygon
            poly = Polygon(points)
            
            # Calculate expansion distance
            distance = poly.area * unclip_ratio / poly.length
            
            # Use pyclipper for polygon expansion
            pco = pyclipper.PyclipperOffset()
            
            # Convert points to integer coordinates (pyclipper requirement)
            int_points = [(int(x * 1000), int(y * 1000)) for x, y in points]
            
            pco.AddPath(int_points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            
            # Execute expansion
            expanded_paths = pco.Execute(distance * 1000)
            
            if expanded_paths:
                # Convert back to float coordinates
                expanded_points = [(x / 1000.0, y / 1000.0) for x, y in expanded_paths[0]]
                return np.array(expanded_points)
            else:
                # Fallback to original polygon
                return np.array(points)
                
        except Exception as e:
            logger.error(f"Polygon expansion failed: {e}")
            # Fallback to simple expansion using Shapely
            try:
                poly = Polygon(points)
                expanded = poly.buffer(distance)
                return np.array(list(expanded.exterior.coords[:-1]))
            except:
                return np.array(points)
    
    @staticmethod
    def calculate_polygon_area(points: List[Tuple[float, float]]) -> float:
        """Calculate polygon area using the shoelace formula."""
        try:
            if len(points) < 3:
                return 0.0
            
            area = 0.0
            n = len(points)
            
            for i in range(n):
                j = (i + 1) % n
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            
            return abs(area) / 2.0
            
        except Exception as e:
            logger.error(f"Polygon area calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def calculate_polygon_perimeter(points: List[Tuple[float, float]]) -> float:
        """Calculate polygon perimeter."""
        try:
            if len(points) < 2:
                return 0.0
            
            perimeter = 0.0
            n = len(points)
            
            for i in range(n):
                j = (i + 1) % n
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                perimeter += math.sqrt(dx * dx + dy * dy)
            
            return perimeter
            
        except Exception as e:
            logger.error(f"Polygon perimeter calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm."""
        try:
            x, y = point
            n = len(polygon)
            inside = False
            
            p1x, p1y = polygon[0]
            for i in range(1, n + 1):
                p2x, p2y = polygon[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            
            return inside
            
        except Exception as e:
            logger.error(f"Point in polygon check failed: {e}")
            return False
    
    @staticmethod
    def merge_overlapping_boxes(boxes: List[List[float]], overlap_threshold: float = 0.5) -> List[List[float]]:
        """Merge overlapping bounding boxes."""
        try:
            if not boxes:
                return []
            
            # Convert boxes to Shapely polygons
            polygons = []
            for box in boxes:
                if len(box) == 4:
                    x0, y0, x1, y1 = box
                    poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                    polygons.append((poly, box))
            
            merged_boxes = []
            used = set()
            
            for i, (poly1, box1) in enumerate(polygons):
                if i in used:
                    continue
                
                # Find overlapping polygons
                overlapping = [box1]
                used.add(i)
                
                for j, (poly2, box2) in enumerate(polygons[i+1:], i+1):
                    if j in used:
                        continue
                    
                    # Check overlap
                    intersection = poly1.intersection(poly2)
                    union = poly1.union(poly2)
                    
                    if intersection.area / union.area > overlap_threshold:
                        overlapping.append(box2)
                        used.add(j)
                        poly1 = union
                
                # Create merged bounding box
                if len(overlapping) == 1:
                    merged_boxes.append(overlapping[0])
                else:
                    # Calculate bounding box of merged polygons
                    all_x = []
                    all_y = []
                    
                    for box in overlapping:
                        if len(box) == 4:
                            x0, y0, x1, y1 = box
                            all_x.extend([x0, x1])
                            all_y.extend([y0, y1])
                    
                    merged_box = [min(all_x), min(all_y), max(all_x), max(all_y)]
                    merged_boxes.append(merged_box)
            
            return merged_boxes
            
        except Exception as e:
            logger.error(f"Box merging failed: {e}")
            return boxes
    
    @staticmethod
    def calculate_box_overlap(box1: List[float], box2: List[float]) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        try:
            if len(box1) != 4 or len(box2) != 4:
                return 0.0
            
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            
            # Calculate intersection
            x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            
            intersection_area = x_overlap * y_overlap
            
            # Calculate union
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = area1 + area2 - intersection_area
            
            if union_area == 0:
                return 0.0
            
            return intersection_area / union_area
            
        except Exception as e:
            logger.error(f"Box overlap calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def rotate_point(point: Tuple[float, float], angle: float, center: Tuple[float, float] = (0, 0)) -> Tuple[float, float]:
        """Rotate point around center by angle (in radians)."""
        try:
            x, y = point
            cx, cy = center
            
            # Translate to origin
            x -= cx
            y -= cy
            
            # Rotate
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            
            new_x = x * cos_angle - y * sin_angle
            new_y = x * sin_angle + y * cos_angle
            
            # Translate back
            new_x += cx
            new_y += cy
            
            return (new_x, new_y)
            
        except Exception as e:
            logger.error(f"Point rotation failed: {e}")
            return point
    
    @staticmethod
    def normalize_coordinates(coordinates: List[Tuple[float, float]], 
                            width: float, 
                            height: float) -> List[Tuple[float, float]]:
        """Normalize coordinates to [0, 1] range."""
        try:
            if width <= 0 or height <= 0:
                return coordinates
            
            normalized = []
            for x, y in coordinates:
                norm_x = max(0, min(1, x / width))
                norm_y = max(0, min(1, y / height))
                normalized.append((norm_x, norm_y))
            
            return normalized
            
        except Exception as e:
            logger.error(f"Coordinate normalization failed: {e}")
            return coordinates
    
    @staticmethod
    def denormalize_coordinates(coordinates: List[Tuple[float, float]], 
                              width: float, 
                              height: float) -> List[Tuple[float, float]]:
        """Denormalize coordinates from [0, 1] range."""
        try:
            denormalized = []
            for x, y in coordinates:
                denorm_x = x * width
                denorm_y = y * height
                denormalized.append((denorm_x, denorm_y))
            
            return denormalized
            
        except Exception as e:
            logger.error(f"Coordinate denormalization failed: {e}")
            return coordinates


class CoordinateTransformer:
    """Coordinate transformation utilities."""
    
    @staticmethod
    def pdf_to_image_coordinates(pdf_coords: List[float], 
                               pdf_width: float, 
                               pdf_height: float,
                               image_width: int, 
                               image_height: int) -> List[float]:
        """Transform PDF coordinates to image coordinates."""
        try:
            if len(pdf_coords) != 4:
                return pdf_coords
            
            x0, y0, x1, y1 = pdf_coords
            
            # PDF coordinates are from bottom-left, image coordinates from top-left
            scale_x = image_width / pdf_width
            scale_y = image_height / pdf_height
            
            # Transform coordinates
            img_x0 = x0 * scale_x
            img_y0 = image_height - (y1 * scale_y)  # Flip Y axis
            img_x1 = x1 * scale_x
            img_y1 = image_height - (y0 * scale_y)  # Flip Y axis
            
            return [img_x0, img_y0, img_x1, img_y1]
            
        except Exception as e:
            logger.error(f"PDF to image coordinate transformation failed: {e}")
            return pdf_coords
    
    @staticmethod
    def image_to_pdf_coordinates(image_coords: List[float], 
                               pdf_width: float, 
                               pdf_height: float,
                               image_width: int, 
                               image_height: int) -> List[float]:
        """Transform image coordinates to PDF coordinates."""
        try:
            if len(image_coords) != 4:
                return image_coords
            
            x0, y0, x1, y1 = image_coords
            
            # Image coordinates are from top-left, PDF coordinates from bottom-left
            scale_x = pdf_width / image_width
            scale_y = pdf_height / image_height
            
            # Transform coordinates
            pdf_x0 = x0 * scale_x
            pdf_y0 = pdf_height - (y1 * scale_y)  # Flip Y axis
            pdf_x1 = x1 * scale_x
            pdf_y1 = pdf_height - (y0 * scale_y)  # Flip Y axis
            
            return [pdf_x0, pdf_y0, pdf_x1, pdf_y1]
            
        except Exception as e:
            logger.error(f"Image to PDF coordinate transformation failed: {e}")
            return image_coords
    
    @staticmethod
    def validate_coordinates(coords: List[float], 
                           max_width: float, 
                           max_height: float) -> List[float]:
        """Validate and clamp coordinates to valid ranges."""
        try:
            if len(coords) != 4:
                return coords
            
            x0, y0, x1, y1 = coords
            
            # Clamp coordinates
            x0 = max(0, min(max_width, x0))
            y0 = max(0, min(max_height, y0))
            x1 = max(0, min(max_width, x1))
            y1 = max(0, min(max_height, y1))
            
            # Ensure proper ordering
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            
            return [x0, y0, x1, y1]
            
        except Exception as e:
            logger.error(f"Coordinate validation failed: {e}")
            return coords


class PerformanceOptimizer:
    """Performance optimization utilities."""
    
    def __init__(self):
        self.geometric_processor = GeometricProcessor()
        self.coordinate_transformer = CoordinateTransformer()
        
        # Performance tracking
        self.operation_times = {}
        self.operation_counts = {}
    
    def track_operation(self, operation_name: str, duration: float):
        """Track operation performance."""
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
            self.operation_counts[operation_name] = 0
        
        self.operation_times[operation_name].append(duration)
        self.operation_counts[operation_name] += 1
        
        # Keep only last 100 measurements
        if len(self.operation_times[operation_name]) > 100:
            self.operation_times[operation_name] = self.operation_times[operation_name][-100:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        for operation in self.operation_times:
            times = self.operation_times[operation]
            if times:
                stats[operation] = {
                    'count': self.operation_counts[operation],
                    'avg_time_ms': sum(times) / len(times) * 1000,
                    'min_time_ms': min(times) * 1000,
                    'max_time_ms': max(times) * 1000,
                    'recent_samples': len(times)
                }
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.operation_times.clear()
        self.operation_counts.clear()
    
    def optimize_polygon_operations(self, polygons: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
        """Optimize multiple polygon operations."""
        try:
            if not polygons:
                return polygons
            
            # Remove duplicate polygons
            unique_polygons = []
            seen = set()
            
            for poly in polygons:
                poly_hash = hash(tuple(tuple(p) for p in poly))
                if poly_hash not in seen:
                    seen.add(poly_hash)
                    unique_polygons.append(poly)
            
            # Remove degenerate polygons (less than 3 points or zero area)
            valid_polygons = []
            for poly in unique_polygons:
                if len(poly) >= 3:
                    area = self.geometric_processor.calculate_polygon_area(poly)
                    if area > 1e-6:  # Minimum area threshold
                        valid_polygons.append(poly)
            
            return valid_polygons
            
        except Exception as e:
            logger.error(f"Polygon optimization failed: {e}")
            return polygons
    
    def batch_coordinate_transform(self, 
                                 coordinates_list: List[List[float]], 
                                 transform_func, 
                                 *args) -> List[List[float]]:
        """Batch coordinate transformation for better performance."""
        try:
            transformed = []
            for coords in coordinates_list:
                transformed_coords = transform_func(coords, *args)
                transformed.append(transformed_coords)
            
            return transformed
            
        except Exception as e:
            logger.error(f"Batch coordinate transformation failed: {e}")
            return coordinates_list


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()

cl
ass StringProcessor:
    """Optimized string processing utilities."""
    
    @staticmethod
    def calculate_edit_distance(str1: str, str2: str) -> int:
        """Calculate edit distance using optimized algorithm."""
        try:
            import editdistance
            return editdistance.eval(str1, str2)
        except ImportError:
            # Fallback to basic Levenshtein distance implementation
            return StringProcessor._levenshtein_distance(str1, str2)
        except Exception as e:
            logger.error(f"Edit distance calculation failed: {e}")
            return max(len(str1), len(str2))  # Worst case
    
    @staticmethod
    def _levenshtein_distance(str1: str, str2: str) -> int:
        """Basic Levenshtein distance implementation as fallback."""
        if len(str1) < len(str2):
            return StringProcessor._levenshtein_distance(str2, str1)
        
        if len(str2) == 0:
            return len(str1)
        
        previous_row = list(range(len(str2) + 1))
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def repair_json(malformed_json: str) -> str:
        """Repair malformed JSON using json-repair."""
        try:
            import json_repair
            return json_repair.repair_json(malformed_json)
        except ImportError:
            # Fallback to basic JSON repair
            return StringProcessor._basic_json_repair(malformed_json)
        except Exception as e:
            logger.error(f"JSON repair failed: {e}")
            return malformed_json
    
    @staticmethod
    def _basic_json_repair(json_str: str) -> str:
        """Basic JSON repair implementation as fallback."""
        try:
            # Remove common issues
            repaired = json_str.strip()
            
            # Fix missing quotes around keys
            import re
            repaired = re.sub(r'(\w+):', r'"\1":', repaired)
            
            # Fix single quotes to double quotes
            repaired = repaired.replace("'", '"')
            
            # Fix trailing commas
            repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
            
            # Ensure proper brackets
            if not repaired.startswith(('{', '[')):
                repaired = '{' + repaired
            if not repaired.endswith(('}', ']')):
                repaired = repaired + '}'
            
            return repaired
            
        except Exception as e:
            logger.error(f"Basic JSON repair failed: {e}")
            return json_str
    
    @staticmethod
    def words_to_numbers(text: str) -> str:
        """Convert word numbers to digits."""
        try:
            from word2number import w2n
            
            # Split text into words
            words = text.split()
            result_words = []
            
            for word in words:
                try:
                    # Try to convert word to number
                    number = w2n.word_to_num(word.lower())
                    result_words.append(str(number))
                except ValueError:
                    # Keep original word if conversion fails
                    result_words.append(word)
            
            return ' '.join(result_words)
            
        except ImportError:
            # Fallback to basic word-to-number conversion
            return StringProcessor._basic_words_to_numbers(text)
        except Exception as e:
            logger.error(f"Words to numbers conversion failed: {e}")
            return text
    
    @staticmethod
    def _basic_words_to_numbers(text: str) -> str:
        """Basic word-to-number conversion as fallback."""
        try:
            word_to_num = {
                'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
                'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
                'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
                'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
                'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
                'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
                'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000',
                'million': '1000000', 'billion': '1000000000'
            }
            
            words = text.split()
            result_words = []
            
            for word in words:
                lower_word = word.lower().strip('.,!?;:')
                if lower_word in word_to_num:
                    result_words.append(word_to_num[lower_word])
                else:
                    result_words.append(word)
            
            return ' '.join(result_words)
            
        except Exception as e:
            logger.error(f"Basic words to numbers conversion failed: {e}")
            return text
    
    @staticmethod
    def calculate_string_similarity(str1: str, str2: str) -> float:
        """Calculate string similarity using multiple methods."""
        try:
            if not str1 or not str2:
                return 0.0
            
            if str1 == str2:
                return 1.0
            
            # Calculate edit distance similarity
            max_len = max(len(str1), len(str2))
            edit_dist = StringProcessor.calculate_edit_distance(str1, str2)
            edit_similarity = 1.0 - (edit_dist / max_len)
            
            # Calculate Jaccard similarity (character-based)
            set1 = set(str1.lower())
            set2 = set(str2.lower())
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            jaccard_similarity = intersection / union if union > 0 else 0.0
            
            # Calculate longest common subsequence similarity
            lcs_length = StringProcessor._longest_common_subsequence(str1, str2)
            lcs_similarity = (2.0 * lcs_length) / (len(str1) + len(str2))
            
            # Weighted average of similarities
            similarity = (edit_similarity * 0.5 + jaccard_similarity * 0.3 + lcs_similarity * 0.2)
            
            return min(1.0, max(0.0, similarity))
            
        except Exception as e:
            logger.error(f"String similarity calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def _longest_common_subsequence(str1: str, str2: str) -> int:
        """Calculate longest common subsequence length."""
        try:
            m, n = len(str1), len(str2)
            
            # Create DP table
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            # Fill DP table
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if str1[i-1] == str2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
            
        except Exception as e:
            logger.error(f"LCS calculation failed: {e}")
            return 0
    
    @staticmethod
    def normalize_text(text: str, 
                      lowercase: bool = True,
                      remove_punctuation: bool = True,
                      remove_extra_spaces: bool = True,
                      remove_numbers: bool = False) -> str:
        """Normalize text with various options."""
        try:
            if not text:
                return text
            
            normalized = text
            
            # Convert to lowercase
            if lowercase:
                normalized = normalized.lower()
            
            # Remove punctuation
            if remove_punctuation:
                import string
                normalized = normalized.translate(str.maketrans('', '', string.punctuation))
            
            # Remove numbers
            if remove_numbers:
                import re
                normalized = re.sub(r'\d+', '', normalized)
            
            # Remove extra spaces
            if remove_extra_spaces:
                import re
                normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            return normalized
            
        except Exception as e:
            logger.error(f"Text normalization failed: {e}")
            return text
    
    @staticmethod
    def extract_numbers(text: str) -> List[Union[int, float]]:
        """Extract all numbers from text."""
        try:
            import re
            
            numbers = []
            
            # Find integers and floats
            number_pattern = r'-?\d+\.?\d*'
            matches = re.findall(number_pattern, text)
            
            for match in matches:
                try:
                    if '.' in match:
                        numbers.append(float(match))
                    else:
                        numbers.append(int(match))
                except ValueError:
                    continue
            
            return numbers
            
        except Exception as e:
            logger.error(f"Number extraction failed: {e}")
            return []
    
    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Clean and normalize whitespace in text."""
        try:
            if not text:
                return text
            
            import re
            
            # Replace multiple whitespace characters with single space
            cleaned = re.sub(r'\s+', ' ', text)
            
            # Remove leading and trailing whitespace
            cleaned = cleaned.strip()
            
            # Replace non-breaking spaces and other unicode spaces
            cleaned = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]', ' ', cleaned)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Whitespace cleaning failed: {e}")
            return text
    
    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to maximum length with suffix."""
        try:
            if not text or len(text) <= max_length:
                return text
            
            if max_length <= len(suffix):
                return suffix[:max_length]
            
            return text[:max_length - len(suffix)] + suffix
            
        except Exception as e:
            logger.error(f"Text truncation failed: {e}")
            return text
    
    @staticmethod
    def split_text_smart(text: str, max_chunk_size: int, overlap: int = 0) -> List[str]:
        """Smart text splitting that tries to preserve word boundaries."""
        try:
            if not text or len(text) <= max_chunk_size:
                return [text] if text else []
            
            chunks = []
            start = 0
            
            while start < len(text):
                end = start + max_chunk_size
                
                if end >= len(text):
                    # Last chunk
                    chunks.append(text[start:])
                    break
                
                # Try to find a good break point (space, punctuation)
                break_point = end
                for i in range(end, start + max_chunk_size // 2, -1):
                    if text[i] in ' \n\t.,!?;:':
                        break_point = i + 1
                        break
                
                chunk = text[start:break_point].strip()
                if chunk:
                    chunks.append(chunk)
                
                # Move start position with overlap
                start = break_point - overlap
                if start <= 0:
                    start = break_point
            
            return chunks
            
        except Exception as e:
            logger.error(f"Smart text splitting failed: {e}")
            return [text]


class TextAnalyzer:
    """Advanced text analysis utilities."""
    
    @staticmethod
    def calculate_readability_score(text: str) -> Dict[str, float]:
        """Calculate various readability scores."""
        try:
            if not text:
                return {'error': 'Empty text'}
            
            # Basic text statistics
            sentences = len([s for s in text.split('.') if s.strip()])
            words = len(text.split())
            characters = len(text.replace(' ', ''))
            syllables = TextAnalyzer._count_syllables(text)
            
            if sentences == 0 or words == 0:
                return {'error': 'Invalid text structure'}
            
            # Flesch Reading Ease Score
            flesch_ease = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
            
            # Flesch-Kincaid Grade Level
            flesch_kincaid = (0.39 * (words / sentences)) + (11.8 * (syllables / words)) - 15.59
            
            # Automated Readability Index
            ari = (4.71 * (characters / words)) + (0.5 * (words / sentences)) - 21.43
            
            return {
                'flesch_reading_ease': max(0, min(100, flesch_ease)),
                'flesch_kincaid_grade': max(0, flesch_kincaid),
                'automated_readability_index': max(0, ari),
                'word_count': words,
                'sentence_count': sentences,
                'character_count': characters,
                'syllable_count': syllables,
                'avg_words_per_sentence': words / sentences,
                'avg_syllables_per_word': syllables / words
            }
            
        except Exception as e:
            logger.error(f"Readability score calculation failed: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def _count_syllables(text: str) -> int:
        """Count syllables in text (approximation)."""
        try:
            import re
            
            # Convert to lowercase and remove punctuation
            text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            words = text.split()
            
            total_syllables = 0
            
            for word in words:
                if not word:
                    continue
                
                # Count vowel groups
                vowels = 'aeiouy'
                syllable_count = 0
                prev_was_vowel = False
                
                for char in word:
                    is_vowel = char in vowels
                    if is_vowel and not prev_was_vowel:
                        syllable_count += 1
                    prev_was_vowel = is_vowel
                
                # Handle silent 'e'
                if word.endswith('e') and syllable_count > 1:
                    syllable_count -= 1
                
                # Ensure at least one syllable per word
                if syllable_count == 0:
                    syllable_count = 1
                
                total_syllables += syllable_count
            
            return total_syllables
            
        except Exception as e:
            logger.error(f"Syllable counting failed: {e}")
            return len(text.split())  # Fallback to word count
    
    @staticmethod
    def extract_keywords(text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF-like scoring."""
        try:
            if not text:
                return []
            
            import re
            from collections import Counter
            
            # Normalize text
            normalized = StringProcessor.normalize_text(text, lowercase=True, remove_punctuation=True)
            words = normalized.split()
            
            # Remove common stop words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
            }
            
            # Filter words
            filtered_words = [word for word in words if len(word) > 2 and word not in stop_words]
            
            # Count word frequencies
            word_counts = Counter(filtered_words)
            
            # Calculate simple TF scores
            total_words = len(filtered_words)
            tf_scores = {word: count / total_words for word, count in word_counts.items()}
            
            # Sort by score and return top k
            sorted_keywords = sorted(tf_scores.items(), key=lambda x: x[1], reverse=True)
            
            return sorted_keywords[:top_k]
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Simple language detection based on character patterns."""
        try:
            if not text:
                return 'unknown'
            
            # Count different character types
            latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 256)
            cyrillic_chars = sum(1 for c in text if 0x0400 <= ord(c) <= 0x04FF)
            chinese_chars = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
            
            total_alpha = sum(1 for c in text if c.isalpha())
            
            if total_alpha == 0:
                return 'unknown'
            
            # Determine dominant script
            if chinese_chars / total_alpha > 0.3:
                return 'chinese'
            elif cyrillic_chars / total_alpha > 0.3:
                return 'russian'
            elif latin_chars / total_alpha > 0.7:
                return 'english'
            else:
                return 'mixed'
                
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return 'unknown'


# Add string processor to performance optimizer
performance_optimizer.string_processor = StringProcessor()
performance_optimizer.text_analyzer = TextAnalyzer()