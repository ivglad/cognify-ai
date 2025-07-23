"""
Hierarchical summarization system for RAPTOR clusters.
"""
import logging
from typing import List, Dict, Any, Optional

import trio

from app.services.chunking.raptor import ClusterNode
from app.services.nlp.rag_tokenizer import rag_tokenizer
from app.core.cache import cache_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class HierarchicalSummarizer:
    """
    Hierarchical summarization system using LLM for RAPTOR clusters.
    """
    
    def __init__(self,
                 max_summary_length: int = 500,
                 min_summary_length: int = 100,
                 summary_overlap_ratio: float = 0.1,
                 batch_size: int = 5):
        """
        Initialize hierarchical summarizer.
        
        Args:
            max_summary_length: Maximum summary length in tokens
            min_summary_length: Minimum summary length in tokens
            summary_overlap_ratio: Overlap ratio between summaries
            batch_size: Batch size for LLM requests
        """
        self.max_summary_length = max_summary_length
        self.min_summary_length = min_summary_length
        self.summary_overlap_ratio = summary_overlap_ratio
        self.batch_size = batch_size
        
        self.tokenizer = rag_tokenizer
        self.cache = cache_manager
        
        # Summary prompts for different levels
        self.summary_prompts = {
            'leaf': """Summarize the following text content in a clear and concise manner. 
Focus on the main ideas and key information. Keep the summary between {min_length} and {max_length} tokens.

Content:
{content}

Summary:""",
            
            'cluster': """Create a comprehensive summary that combines the following related summaries. 
Identify common themes and synthesize the information into a coherent overview. 
Keep the summary between {min_length} and {max_length} tokens.

Related summaries:
{content}

Combined summary:""",
            
            'root': """Create a high-level executive summary that captures the essence of the following content summaries. 
Focus on the most important themes and insights across all the material. 
Keep the summary between {min_length} and {max_length} tokens.

Content summaries:
{content}

Executive summary:"""
        }
    
    async def summarize_raptor_tree(self, root_node: ClusterNode) -> ClusterNode:
        """
        Generate summaries for all nodes in RAPTOR tree.
        
        Args:
            root_node: Root node of RAPTOR tree
            
        Returns:
            Root node with summaries generated
        """
        if not root_node:
            return root_node
        
        try:
            # Initialize tokenizer
            if not self.tokenizer._initialized:
                await self.tokenizer.initialize()
            
            logger.info("Starting hierarchical summarization")
            
            # Process tree bottom-up (leaves first, then parents)
            await self._summarize_bottom_up(root_node)
            
            logger.info("Hierarchical summarization completed")
            
            return root_node
            
        except Exception as e:
            logger.error(f"Hierarchical summarization failed: {e}")
            return root_node
    
    async def _summarize_bottom_up(self, node: ClusterNode):
        """Summarize tree in bottom-up manner."""
        try:
            # First, recursively summarize all children
            if node.children:
                # Process children in parallel
                async with trio.open_nursery() as nursery:
                    for child in node.children:
                        nursery.start_soon(self._summarize_bottom_up, child)
                
                # After children are summarized, summarize this node
                await self._summarize_cluster_node(node)
            else:
                # Leaf node - summarize directly from content
                await self._summarize_leaf_node(node)
                
        except Exception as e:
            logger.error(f"Bottom-up summarization failed for node {node.node_id}: {e}")
    
    async def _summarize_leaf_node(self, node: ClusterNode):
        """Summarize a leaf node from its content."""
        try:
            if not node.summary or len(node.summary.strip()) == 0:
                logger.warning(f"Leaf node {node.node_id} has no content to summarize")
                node.summary = f"Content section {node.node_id}"
                return
            
            # Check cache first
            cache_key = f"summary_leaf:{hash(node.summary)}"
            cached_summary = await self.cache.get(cache_key)
            if cached_summary:
                node.summary = cached_summary
                return
            
            # Determine if summarization is needed
            content_tokens = await self._count_tokens(node.summary)
            
            if content_tokens <= self.max_summary_length:
                # Content is already short enough
                logger.debug(f"Leaf node {node.node_id} content is already concise ({content_tokens} tokens)")
                return
            
            # Generate summary using LLM
            summary = await self._generate_llm_summary(
                content=node.summary,
                prompt_type='leaf',
                target_length=min(self.max_summary_length, content_tokens // 2)
            )
            
            if summary:
                node.summary = summary
                # Cache the summary
                await self.cache.set(cache_key, summary, ttl=86400)  # 24 hours
                logger.debug(f"Generated leaf summary for {node.node_id}")
            else:
                logger.warning(f"Failed to generate summary for leaf node {node.node_id}")
                
        except Exception as e:
            logger.error(f"Leaf node summarization failed for {node.node_id}: {e}")
    
    async def _summarize_cluster_node(self, node: ClusterNode):
        """Summarize a cluster node from its children's summaries."""
        try:
            if not node.children:
                return
            
            # Collect children summaries
            child_summaries = []
            for child in node.children:
                if child.summary and child.summary.strip():
                    child_summaries.append(child.summary.strip())
            
            if not child_summaries:
                node.summary = f"Cluster {node.node_id} with {len(node.children)} sections"
                return
            
            # Combine child summaries
            combined_content = "\n\n".join(child_summaries)
            
            # Check cache
            cache_key = f"summary_cluster:{hash(combined_content)}"
            cached_summary = await self.cache.get(cache_key)
            if cached_summary:
                node.summary = cached_summary
                return
            
            # Determine prompt type based on level
            if node.metadata.get('is_root', False):
                prompt_type = 'root'
            else:
                prompt_type = 'cluster'
            
            # Generate summary
            summary = await self._generate_llm_summary(
                content=combined_content,
                prompt_type=prompt_type,
                target_length=self.max_summary_length
            )
            
            if summary:
                node.summary = summary
                # Cache the summary
                await self.cache.set(cache_key, summary, ttl=86400)
                logger.debug(f"Generated cluster summary for {node.node_id}")
            else:
                # Fallback: use truncated combination
                node.summary = self._create_fallback_summary(child_summaries, node)
                logger.warning(f"Used fallback summary for cluster node {node.node_id}")
                
        except Exception as e:
            logger.error(f"Cluster node summarization failed for {node.node_id}: {e}")
            node.summary = self._create_fallback_summary(
                [child.summary for child in node.children if child.summary], 
                node
            )
    
    async def _generate_llm_summary(self, 
                                  content: str, 
                                  prompt_type: str,
                                  target_length: int) -> Optional[str]:
        """Generate summary using LLM."""
        try:
            # For now, implement a simple extractive summarization
            # In a full implementation, this would call an actual LLM service
            
            # Get prompt template
            prompt_template = self.summary_prompts.get(prompt_type, self.summary_prompts['cluster'])
            
            # Calculate target lengths
            min_length = max(self.min_summary_length, target_length // 2)
            max_length = min(self.max_summary_length, target_length)
            
            # Format prompt
            prompt = prompt_template.format(
                content=content,
                min_length=min_length,
                max_length=max_length
            )
            
            # For now, use extractive summarization as placeholder
            summary = await self._extractive_summarization(content, target_length)
            
            return summary
            
        except Exception as e:
            logger.error(f"LLM summary generation failed: {e}")
            return None
    
    async def _extractive_summarization(self, content: str, target_length: int) -> str:
        """Simple extractive summarization as LLM placeholder."""
        try:
            # Split into sentences
            sentences = await self._split_into_sentences(content)
            
            if not sentences:
                return content[:target_length * 4]  # Rough character estimate
            
            # Score sentences by position and length
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                # Simple scoring: prefer earlier sentences and moderate length
                position_score = 1.0 / (i + 1)  # Earlier sentences get higher score
                length_score = min(1.0, len(sentence.split()) / 20)  # Prefer 10-20 word sentences
                
                total_score = position_score * 0.7 + length_score * 0.3
                scored_sentences.append((sentence, total_score))
            
            # Sort by score
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Select sentences until target length
            selected_sentences = []
            current_tokens = 0
            
            for sentence, score in scored_sentences:
                sentence_tokens = await self._count_tokens(sentence)
                
                if current_tokens + sentence_tokens <= target_length:
                    selected_sentences.append(sentence)
                    current_tokens += sentence_tokens
                
                if current_tokens >= target_length * 0.8:  # 80% of target
                    break
            
            # Reorder selected sentences by original position
            if selected_sentences:
                # Find original positions
                sentence_positions = []
                for selected in selected_sentences:
                    for i, original in enumerate(sentences):
                        if selected == original:
                            sentence_positions.append((i, selected))
                            break
                
                # Sort by position
                sentence_positions.sort(key=lambda x: x[0])
                
                # Extract sentences in order
                summary = ' '.join([sent for pos, sent in sentence_positions])
                return summary
            else:
                # Fallback: return first part of content
                return content[:target_length * 4]
                
        except Exception as e:
            logger.error(f"Extractive summarization failed: {e}")
            return content[:target_length * 4]
    
    async def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        try:
            # Simple sentence splitting
            import re
            
            # Split by sentence endings
            sentences = re.split(r'[.!?]+', text)
            
            # Clean and filter sentences
            cleaned_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:  # Minimum sentence length
                    cleaned_sentences.append(sentence)
            
            return cleaned_sentences
            
        except Exception as e:
            logger.error(f"Sentence splitting failed: {e}")
            return [text]
    
    async def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            tokens = await self.tokenizer.tokenize_text(text, remove_stopwords=False, stem_words=False)
            return len(tokens)
        except Exception:
            # Fallback: rough estimate
            return len(text.split())
    
    def _create_fallback_summary(self, summaries: List[str], node: ClusterNode) -> str:
        """Create fallback summary when LLM fails."""
        try:
            if not summaries:
                return f"Content cluster at level {node.level}"
            
            # Take first sentence from each summary
            first_sentences = []
            for summary in summaries[:3]:  # Limit to first 3
                sentences = summary.split('.')
                if sentences:
                    first_sentences.append(sentences[0].strip())
            
            if first_sentences:
                return '. '.join(first_sentences) + '.'
            else:
                return f"Combined content from {len(summaries)} sections"
                
        except Exception:
            return f"Content cluster with {len(summaries)} sections"
    
    async def validate_summaries(self, root_node: ClusterNode) -> Dict[str, Any]:
        """Validate quality of generated summaries."""
        try:
            stats = {
                'total_nodes': 0,
                'nodes_with_summaries': 0,
                'average_summary_length': 0,
                'summary_lengths': [],
                'validation_issues': []
            }
            
            await self._collect_summary_stats(root_node, stats)
            
            # Calculate averages
            if stats['summary_lengths']:
                stats['average_summary_length'] = sum(stats['summary_lengths']) / len(stats['summary_lengths'])
            
            # Validate summary quality
            if stats['nodes_with_summaries'] < stats['total_nodes']:
                missing_count = stats['total_nodes'] - stats['nodes_with_summaries']
                stats['validation_issues'].append(f"{missing_count} nodes missing summaries")
            
            # Check for very short summaries
            short_summaries = [length for length in stats['summary_lengths'] if length < self.min_summary_length]
            if short_summaries:
                stats['validation_issues'].append(f"{len(short_summaries)} summaries below minimum length")
            
            # Check for very long summaries
            long_summaries = [length for length in stats['summary_lengths'] if length > self.max_summary_length]
            if long_summaries:
                stats['validation_issues'].append(f"{len(long_summaries)} summaries above maximum length")
            
            return stats
            
        except Exception as e:
            logger.error(f"Summary validation failed: {e}")
            return {'error': str(e)}
    
    async def _collect_summary_stats(self, node: ClusterNode, stats: Dict[str, Any]):
        """Recursively collect summary statistics."""
        try:
            stats['total_nodes'] += 1
            
            if node.summary and node.summary.strip():
                stats['nodes_with_summaries'] += 1
                summary_length = await self._count_tokens(node.summary)
                stats['summary_lengths'].append(summary_length)
            
            # Process children
            for child in node.children:
                await self._collect_summary_stats(child, stats)
                
        except Exception as e:
            logger.error(f"Summary stats collection failed: {e}")
    
    async def get_summarizer_stats(self) -> Dict[str, Any]:
        """Get summarizer statistics."""
        try:
            return {
                'max_summary_length': self.max_summary_length,
                'min_summary_length': self.min_summary_length,
                'summary_overlap_ratio': self.summary_overlap_ratio,
                'batch_size': self.batch_size,
                'available_prompt_types': list(self.summary_prompts.keys()),
                'tokenizer_initialized': self.tokenizer._initialized if hasattr(self.tokenizer, '_initialized') else False
            }
            
        except Exception as e:
            logger.error(f"Summarizer stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
hierarchical_summarizer = HierarchicalSummarizer()