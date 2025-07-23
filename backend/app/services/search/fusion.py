"""
Result fusion algorithms for combining sparse and dense search results.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import math
from collections import defaultdict, Counter
import numpy as np

import trio

from app.core.config import settings

logger = logging.getLogger(__name__)


class SearchResultFusion:
    """
    Advanced result fusion system for combining multiple search methods.
    """
    
    def __init__(self,
                 default_fusion_method: str = "rrf",
                 rrf_k: int = 60,
                 weighted_alpha: float = 0.5,
                 min_fusion_score: float = 0.1):
        """
        Initialize result fusion system.
        
        Args:
            default_fusion_method: Default fusion algorithm to use
            rrf_k: K parameter for Reciprocal Rank Fusion
            weighted_alpha: Alpha parameter for weighted fusion
            min_fusion_score: Minimum score threshold for fused results
        """
        self.default_fusion_method = default_fusion_method
        self.rrf_k = rrf_k
        self.weighted_alpha = weighted_alpha
        self.min_fusion_score = min_fusion_score
        
        # Available fusion methods
        self.fusion_methods = {
            'rrf': self._reciprocal_rank_fusion,
            'weighted': self._weighted_fusion,
            'borda': self._borda_count_fusion,
            'combsum': self._combsum_fusion,
            'combmnz': self._combmnz_fusion,
            'adaptive': self._adaptive_fusion
        }
        
        # Method weights for different scenarios
        self.method_weights = {
            'sparse': 1.0,
            'dense': 1.0,
            'kg_enhanced': 0.8,
            'reranked': 1.2
        }
    
    async def fuse_results(self, 
                         search_results: Dict[str, List[Dict[str, Any]]],
                         fusion_method: Optional[str] = None,
                         weights: Optional[Dict[str, float]] = None,
                         max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Fuse results from multiple search methods.
        
        Args:
            search_results: Dictionary mapping method names to result lists
            fusion_method: Fusion algorithm to use
            weights: Optional weights for different methods
            max_results: Maximum number of results to return
            
        Returns:
            Fused and ranked results
        """
        try:
            if not search_results:
                return []
            
            # Filter out empty result lists
            filtered_results = {k: v for k, v in search_results.items() if v}
            
            if not filtered_results:
                return []
            
            # If only one method has results, return those
            if len(filtered_results) == 1:
                method_name, results = next(iter(filtered_results.items()))
                return self._normalize_results(results, method_name)[:max_results]
            
            # Select fusion method
            fusion_method = fusion_method or self.default_fusion_method
            
            if fusion_method not in self.fusion_methods:
                logger.warning(f"Unknown fusion method: {fusion_method}, using default")
                fusion_method = self.default_fusion_method
            
            # Apply fusion algorithm
            fusion_func = self.fusion_methods[fusion_method]
            fused_results = await fusion_func(filtered_results, weights)
            
            # Filter by minimum score
            filtered_fused = [r for r in fused_results if r.get('fusion_score', 0) >= self.min_fusion_score]
            
            # Sort by fusion score
            filtered_fused.sort(key=lambda x: x.get('fusion_score', 0), reverse=True)
            
            # Add fusion metadata
            for i, result in enumerate(filtered_fused):
                result['fusion_rank'] = i + 1
                result['fusion_method'] = fusion_method
                result['num_methods'] = len(filtered_results)
            
            logger.debug(f"Fused {sum(len(results) for results in filtered_results.values())} results "
                        f"from {len(filtered_results)} methods into {len(filtered_fused)} results")
            
            return filtered_fused[:max_results]
            
        except Exception as e:
            logger.error(f"Result fusion failed: {e}")
            # Fallback: return results from first available method
            if search_results:
                first_method_results = next(iter(search_results.values()))
                return first_method_results[:max_results]
            return []
    
    async def _reciprocal_rank_fusion(self, 
                                    search_results: Dict[str, List[Dict[str, Any]]],
                                    weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) algorithm.
        
        RRF Score = sum(weight / (k + rank)) for each method
        """
        try:
            fused_scores = defaultdict(float)
            all_results = {}
            method_ranks = {}
            
            # Calculate RRF scores
            for method_name, results in search_results.items():
                method_weight = weights.get(method_name, 1.0) if weights else self.method_weights.get(method_name, 1.0)
                method_ranks[method_name] = {}
                
                for rank, result in enumerate(results):
                    result_id = self._get_result_id(result)
                    
                    # Store result data
                    if result_id not in all_results:
                        all_results[result_id] = result.copy()
                        all_results[result_id]['source_methods'] = []
                        all_results[result_id]['method_scores'] = {}
                        all_results[result_id]['method_ranks'] = {}
                    
                    # Add method information
                    all_results[result_id]['source_methods'].append(method_name)
                    all_results[result_id]['method_scores'][method_name] = result.get('score', 0.0)
                    all_results[result_id]['method_ranks'][method_name] = rank + 1
                    
                    # Calculate RRF score
                    rrf_score = method_weight / (self.rrf_k + rank + 1)
                    fused_scores[result_id] += rrf_score
                    
                    method_ranks[method_name][result_id] = rank + 1
            
            # Create fused results
            fused_results = []
            for result_id, fusion_score in fused_scores.items():
                result = all_results[result_id]
                result['fusion_score'] = fusion_score
                result['fusion_algorithm'] = 'rrf'
                result['rrf_k'] = self.rrf_k
                fused_results.append(result)
            
            return fused_results
            
        except Exception as e:
            logger.error(f"RRF fusion failed: {e}")
            return []
    
    async def _weighted_fusion(self, 
                             search_results: Dict[str, List[Dict[str, Any]]],
                             weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Weighted score fusion algorithm.
        
        Weighted Score = sum(weight * normalized_score) for each method
        """
        try:
            # Normalize scores for each method
            normalized_results = {}
            
            for method_name, results in search_results.items():
                if not results:
                    continue
                
                # Get score range for normalization
                scores = [r.get('score', 0.0) for r in results]
                max_score = max(scores) if scores else 1.0
                min_score = min(scores) if scores else 0.0
                score_range = max_score - min_score if max_score > min_score else 1.0
                
                # Normalize scores to [0, 1]
                normalized_results[method_name] = []
                for result in results:
                    normalized_result = result.copy()
                    original_score = result.get('score', 0.0)
                    normalized_score = (original_score - min_score) / score_range
                    normalized_result['normalized_score'] = normalized_score
                    normalized_results[method_name].append(normalized_result)
            
            # Combine results with weighted scores
            fused_scores = defaultdict(float)
            all_results = {}
            
            for method_name, results in normalized_results.items():
                method_weight = weights.get(method_name, 1.0) if weights else self.method_weights.get(method_name, 1.0)
                
                for result in results:
                    result_id = self._get_result_id(result)
                    
                    # Store result data
                    if result_id not in all_results:
                        all_results[result_id] = result.copy()
                        all_results[result_id]['source_methods'] = []
                        all_results[result_id]['method_scores'] = {}
                        all_results[result_id]['weighted_scores'] = {}
                    
                    # Add method information
                    all_results[result_id]['source_methods'].append(method_name)
                    all_results[result_id]['method_scores'][method_name] = result.get('score', 0.0)
                    
                    # Calculate weighted score
                    normalized_score = result.get('normalized_score', 0.0)
                    weighted_score = method_weight * normalized_score
                    all_results[result_id]['weighted_scores'][method_name] = weighted_score
                    
                    fused_scores[result_id] += weighted_score
            
            # Create fused results
            fused_results = []
            for result_id, fusion_score in fused_scores.items():
                result = all_results[result_id]
                result['fusion_score'] = fusion_score
                result['fusion_algorithm'] = 'weighted'
                fused_results.append(result)
            
            return fused_results
            
        except Exception as e:
            logger.error(f"Weighted fusion failed: {e}")
            return []
    
    async def _borda_count_fusion(self, 
                                search_results: Dict[str, List[Dict[str, Any]]],
                                weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Borda Count fusion algorithm.
        
        Borda Score = sum(weight * (n - rank)) for each method
        """
        try:
            fused_scores = defaultdict(float)
            all_results = {}
            
            for method_name, results in search_results.items():
                method_weight = weights.get(method_name, 1.0) if weights else self.method_weights.get(method_name, 1.0)
                n = len(results)
                
                for rank, result in enumerate(results):
                    result_id = self._get_result_id(result)
                    
                    # Store result data
                    if result_id not in all_results:
                        all_results[result_id] = result.copy()
                        all_results[result_id]['source_methods'] = []
                        all_results[result_id]['method_scores'] = {}
                        all_results[result_id]['borda_scores'] = {}
                    
                    # Add method information
                    all_results[result_id]['source_methods'].append(method_name)
                    all_results[result_id]['method_scores'][method_name] = result.get('score', 0.0)
                    
                    # Calculate Borda score
                    borda_score = method_weight * (n - rank)
                    all_results[result_id]['borda_scores'][method_name] = borda_score
                    fused_scores[result_id] += borda_score
            
            # Create fused results
            fused_results = []
            for result_id, fusion_score in fused_scores.items():
                result = all_results[result_id]
                result['fusion_score'] = fusion_score
                result['fusion_algorithm'] = 'borda'
                fused_results.append(result)
            
            return fused_results
            
        except Exception as e:
            logger.error(f"Borda Count fusion failed: {e}")
            return []
    
    async def _combsum_fusion(self, 
                            search_results: Dict[str, List[Dict[str, Any]]],
                            weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        CombSUM fusion algorithm.
        
        CombSUM Score = sum(weight * score) for each method
        """
        try:
            fused_scores = defaultdict(float)
            all_results = {}
            
            for method_name, results in search_results.items():
                method_weight = weights.get(method_name, 1.0) if weights else self.method_weights.get(method_name, 1.0)
                
                for result in results:
                    result_id = self._get_result_id(result)
                    
                    # Store result data
                    if result_id not in all_results:
                        all_results[result_id] = result.copy()
                        all_results[result_id]['source_methods'] = []
                        all_results[result_id]['method_scores'] = {}
                    
                    # Add method information
                    all_results[result_id]['source_methods'].append(method_name)
                    all_results[result_id]['method_scores'][method_name] = result.get('score', 0.0)
                    
                    # Add weighted score
                    score = result.get('score', 0.0)
                    fused_scores[result_id] += method_weight * score
            
            # Create fused results
            fused_results = []
            for result_id, fusion_score in fused_scores.items():
                result = all_results[result_id]
                result['fusion_score'] = fusion_score
                result['fusion_algorithm'] = 'combsum'
                fused_results.append(result)
            
            return fused_results
            
        except Exception as e:
            logger.error(f"CombSUM fusion failed: {e}")
            return []
    
    async def _combmnz_fusion(self, 
                            search_results: Dict[str, List[Dict[str, Any]]],
                            weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        CombMNZ fusion algorithm.
        
        CombMNZ Score = CombSUM * number_of_methods_with_result
        """
        try:
            # First calculate CombSUM scores
            combsum_results = await self._combsum_fusion(search_results, weights)
            
            # Convert to dictionary for easy lookup
            combsum_dict = {self._get_result_id(r): r for r in combsum_results}
            
            # Calculate CombMNZ scores
            fused_results = []
            for result_id, result in combsum_dict.items():
                combsum_score = result.get('fusion_score', 0.0)
                num_methods = len(result.get('source_methods', []))
                
                # CombMNZ = CombSUM * number of methods
                combmnz_score = combsum_score * num_methods
                
                result['fusion_score'] = combmnz_score
                result['fusion_algorithm'] = 'combmnz'
                result['combsum_score'] = combsum_score
                result['method_count'] = num_methods
                
                fused_results.append(result)
            
            return fused_results
            
        except Exception as e:
            logger.error(f"CombMNZ fusion failed: {e}")
            return []
    
    async def _adaptive_fusion(self, 
                             search_results: Dict[str, List[Dict[str, Any]]],
                             weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Adaptive fusion that selects the best method based on result characteristics.
        """
        try:
            # Analyze result characteristics
            method_analysis = {}
            
            for method_name, results in search_results.items():
                if not results:
                    continue
                
                scores = [r.get('score', 0.0) for r in results]
                analysis = {
                    'count': len(results),
                    'avg_score': sum(scores) / len(scores),
                    'max_score': max(scores),
                    'min_score': min(scores),
                    'score_variance': np.var(scores) if len(scores) > 1 else 0.0
                }
                method_analysis[method_name] = analysis
            
            # Select fusion strategy based on characteristics
            if len(search_results) == 2:
                # For two methods, use weighted fusion
                return await self._weighted_fusion(search_results, weights)
            elif any(analysis['score_variance'] > 0.1 for analysis in method_analysis.values()):
                # High variance in scores, use RRF
                return await self._reciprocal_rank_fusion(search_results, weights)
            else:
                # Low variance, use CombMNZ
                return await self._combmnz_fusion(search_results, weights)
            
        except Exception as e:
            logger.error(f"Adaptive fusion failed: {e}")
            # Fallback to RRF
            return await self._reciprocal_rank_fusion(search_results, weights)
    
    def _get_result_id(self, result: Dict[str, Any]) -> str:
        """Generate unique ID for a result."""
        try:
            # Try different ID fields in order of preference
            for id_field in ['chunk_id', 'document_id', 'id']:
                if id_field in result and result[id_field]:
                    return str(result[id_field])
            
            # Fallback: use content hash
            content = result.get('content', result.get('text', ''))
            return f"content_hash_{hash(content)}"
            
        except Exception:
            return f"result_hash_{hash(str(result))}"
    
    def _normalize_results(self, 
                         results: List[Dict[str, Any]], 
                         method_name: str) -> List[Dict[str, Any]]:
        """Normalize results from a single method."""
        try:
            normalized = []
            
            for i, result in enumerate(results):
                normalized_result = result.copy()
                normalized_result['fusion_score'] = result.get('score', 0.0)
                normalized_result['fusion_rank'] = i + 1
                normalized_result['fusion_method'] = 'single_method'
                normalized_result['source_methods'] = [method_name]
                normalized_result['num_methods'] = 1
                normalized.append(normalized_result)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Result normalization failed: {e}")
            return results
    
    async def evaluate_fusion_quality(self, 
                                    fused_results: List[Dict[str, Any]],
                                    ground_truth: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of fusion results.
        
        Args:
            fused_results: Fused search results
            ground_truth: Optional list of relevant result IDs
            
        Returns:
            Quality metrics
        """
        try:
            metrics = {
                'total_results': len(fused_results),
                'method_diversity': 0.0,
                'score_distribution': {},
                'fusion_coverage': 0.0
            }
            
            if not fused_results:
                return metrics
            
            # Calculate method diversity
            all_methods = set()
            method_counts = Counter()
            
            for result in fused_results:
                source_methods = result.get('source_methods', [])
                all_methods.update(source_methods)
                for method in source_methods:
                    method_counts[method] += 1
            
            metrics['unique_methods'] = len(all_methods)
            metrics['method_distribution'] = dict(method_counts)
            
            # Calculate diversity (entropy)
            if method_counts:
                total_method_instances = sum(method_counts.values())
                entropy = 0.0
                for count in method_counts.values():
                    p = count / total_method_instances
                    entropy -= p * math.log2(p) if p > 0 else 0
                metrics['method_diversity'] = entropy
            
            # Score distribution
            scores = [r.get('fusion_score', 0.0) for r in fused_results]
            if scores:
                metrics['score_distribution'] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': min(scores),
                    'max': max(scores)
                }
            
            # Coverage (how many results come from multiple methods)
            multi_method_results = sum(1 for r in fused_results if len(r.get('source_methods', [])) > 1)
            metrics['fusion_coverage'] = multi_method_results / len(fused_results)
            
            # Ground truth evaluation if available
            if ground_truth:
                relevant_found = 0
                for result in fused_results:
                    result_id = self._get_result_id(result)
                    if result_id in ground_truth:
                        relevant_found += 1
                
                metrics['precision'] = relevant_found / len(fused_results)
                metrics['recall'] = relevant_found / len(ground_truth)
                
                if metrics['precision'] + metrics['recall'] > 0:
                    metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
                else:
                    metrics['f1_score'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Fusion quality evaluation failed: {e}")
            return {'error': str(e)}
    
    async def optimize_fusion_parameters(self, 
                                       test_cases: List[Dict[str, Any]],
                                       optimization_metric: str = 'f1_score') -> Dict[str, Any]:
        """
        Optimize fusion parameters based on test cases.
        
        Args:
            test_cases: List of test cases with search_results and ground_truth
            optimization_metric: Metric to optimize for
            
        Returns:
            Optimized parameters
        """
        try:
            if not test_cases:
                return {}
            
            best_params = {
                'rrf_k': self.rrf_k,
                'weighted_alpha': self.weighted_alpha,
                'min_fusion_score': self.min_fusion_score
            }
            best_score = 0.0
            
            # Parameter ranges to test
            rrf_k_values = [30, 60, 90, 120]
            alpha_values = [0.3, 0.5, 0.7]
            min_score_values = [0.05, 0.1, 0.15, 0.2]
            
            # Test parameter combinations
            for rrf_k in rrf_k_values:
                for alpha in alpha_values:
                    for min_score in min_score_values:
                        # Set test parameters
                        original_params = (self.rrf_k, self.weighted_alpha, self.min_fusion_score)
                        self.rrf_k = rrf_k
                        self.weighted_alpha = alpha
                        self.min_fusion_score = min_score
                        
                        # Evaluate on test cases
                        total_score = 0.0
                        valid_cases = 0
                        
                        for test_case in test_cases:
                            search_results = test_case.get('search_results', {})
                            ground_truth = test_case.get('ground_truth', [])
                            
                            if search_results and ground_truth:
                                fused_results = await self.fuse_results(search_results)
                                metrics = await self.evaluate_fusion_quality(fused_results, ground_truth)
                                
                                if optimization_metric in metrics:
                                    total_score += metrics[optimization_metric]
                                    valid_cases += 1
                        
                        # Calculate average score
                        avg_score = total_score / valid_cases if valid_cases > 0 else 0.0
                        
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = {
                                'rrf_k': rrf_k,
                                'weighted_alpha': alpha,
                                'min_fusion_score': min_score
                            }
                        
                        # Restore original parameters
                        self.rrf_k, self.weighted_alpha, self.min_fusion_score = original_params
            
            # Set optimal parameters
            self.rrf_k = best_params['rrf_k']
            self.weighted_alpha = best_params['weighted_alpha']
            self.min_fusion_score = best_params['min_fusion_score']
            
            logger.info(f"Optimized fusion parameters: {best_params} (score: {best_score:.3f})")
            
            return {
                'optimized_parameters': best_params,
                'optimization_score': best_score,
                'optimization_metric': optimization_metric
            }
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            return {}
    
    async def get_fusion_stats(self) -> Dict[str, Any]:
        """Get fusion system statistics."""
        try:
            return {
                'default_fusion_method': self.default_fusion_method,
                'available_methods': list(self.fusion_methods.keys()),
                'parameters': {
                    'rrf_k': self.rrf_k,
                    'weighted_alpha': self.weighted_alpha,
                    'min_fusion_score': self.min_fusion_score
                },
                'method_weights': self.method_weights.copy()
            }
            
        except Exception as e:
            logger.error(f"Fusion stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
search_result_fusion = SearchResultFusion()