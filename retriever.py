from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime

from embedder import Embedder

logger = logging.getLogger(__name__)

@dataclass
class Citation:
    """Represents a citation with detailed source information."""
    text: str
    metadata: Dict[str, Any]
    confidence_score: float
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    page_number: Optional[int] = None
    paragraph_number: Optional[int] = None

@dataclass
class SearchResult:
    """Represents a search result with synthesis information."""
    citations: List[Citation]
    contradictions: List[Tuple[Citation, Citation]]
    cross_references: List[Tuple[Citation, Citation]]
    synthesis_notes: List[str]
    confidence_score: float

class Retriever:
    def __init__(self, embedder: Embedder):
        """Initialize the retriever with an embedder."""
        self.embedder = embedder
        logger.info("Initialized retriever")
    
    def _calculate_confidence_score(self, similarity_score: float) -> float:
        """Calculate a confidence score from similarity score."""
        # Convert cosine similarity to a more intuitive 0-100 scale
        return (similarity_score + 1) * 50
    
    def _find_contradictions(self, citations: List[Citation]) -> List[Tuple[Citation, Citation]]:
        """Find potential contradictions between citations."""
        contradictions = []
        for i, citation1 in enumerate(citations):
            for citation2 in citations[i+1:]:
                # Check if citations are from different sources
                if citation1.metadata.get('item_key') != citation2.metadata.get('item_key'):
                    # Here we could add more sophisticated contradiction detection
                    # For now, just check for basic negation patterns
                    if "not" in citation1.text and citation1.text.replace("not", "") in citation2.text:
                        contradictions.append((citation1, citation2))
        return contradictions
    
    def _find_cross_references(self, citations: List[Citation]) -> List[Tuple[Citation, Citation]]:
        """Find cross-references between documents."""
        cross_refs = []
        for i, citation1 in enumerate(citations):
            for citation2 in citations[i+1:]:
                # Check if one document cites the other
                if citation1.metadata.get('title', '') in citation2.text or \
                   citation2.metadata.get('title', '') in citation1.text:
                    cross_refs.append((citation1, citation2))
        return cross_refs
    
    def _extract_context(self, doc: Dict[str, Any], chunk_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract text before and after the matching chunk."""
        full_text = doc.get('text', '')
        if not full_text:
            return None, None
            
        # Find the chunk in the full text
        chunk_start = full_text.find(chunk_text)
        if chunk_start == -1:
            return None, None
            
        # Extract context (up to 100 chars before and after)
        context_before = full_text[max(0, chunk_start-100):chunk_start].strip()
        chunk_end = chunk_start + len(chunk_text)
        context_after = full_text[chunk_end:min(len(full_text), chunk_end+100)].strip()
        
        return context_before, context_after
    
    def search(self, query: str, k: int = 10) -> SearchResult:
        """
        Enhanced search that returns detailed results with synthesis information.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            
        Returns:
            SearchResult: Detailed search results with citations and synthesis
        """
        # Get raw search results
        # Increase k by 100% to have more candidates to filter from
        raw_results = self.embedder.search(query, k=int(k * 2))
        
        # Process each result into a citation
        citations = []
        for result in raw_results:
            # Extract context
            context_before, context_after = self._extract_context(result, result['text'])
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(result.get('score', 0))
            
            # Create citation
            citation = Citation(
                text=result['text'],
                metadata=result['metadata'],
                confidence_score=confidence,
                context_before=context_before,
                context_after=context_after,
                page_number=result['metadata'].get('page'),
                paragraph_number=result['metadata'].get('paragraph')
            )
            citations.append(citation)
        
        # Apply additional relevance filtering - remove citations below confidence threshold
        threshold = 50.0  # Min 50% confidence
        citations = [c for c in citations if c.confidence_score >= threshold]
        
        # If we still have more than k citations after filtering, keep only the top k
        if len(citations) > k:
            citations = sorted(citations, key=lambda c: c.confidence_score, reverse=True)[:k]
        
        # Find contradictions and cross-references
        contradictions = self._find_contradictions(citations)
        cross_refs = self._find_cross_references(citations)
        
        # Generate synthesis notes
        synthesis_notes = []
        if contradictions:
            synthesis_notes.append("⚠️ Found potential contradictions between sources")
        if cross_refs:
            synthesis_notes.append("ℹ️ Found cross-references between documents")
        if len(citations) == 0:
            synthesis_notes.append("⚠️ No highly relevant sources found for this query")
        
        # Calculate overall confidence based on the citations we kept
        overall_confidence = np.mean([c.confidence_score for c in citations]) if citations else 0
        
        return SearchResult(
            citations=citations,
            contradictions=contradictions,
            cross_references=cross_refs,
            synthesis_notes=synthesis_notes,
            confidence_score=overall_confidence
        )
    
    def format_for_context(self, search_result: SearchResult, max_tokens: int = 4000) -> str:
        """
        Format search results into a context string for the LLM.
        
        Args:
            search_result (SearchResult): Search results to format
            max_tokens (int): Maximum tokens for context
            
        Returns:
            str: Formatted context
        """
        if not search_result.citations:
            return "No relevant information found."
        
        context_parts = []
        total_length = 0
        char_per_token = 4  # Approximation
        
        # Add synthesis notes if any
        if search_result.synthesis_notes:
            notes = "\n".join(search_result.synthesis_notes)
            context_parts.append(f"Analysis Notes:\n{notes}\n\n")
            total_length += len(notes) / char_per_token
        
        # Add each citation with detailed information
        for i, citation in enumerate(search_result.citations):
            # Format metadata
            metadata = citation.metadata
            source = metadata.get('title', 'Unknown source')
            authors = metadata.get('creators', [])
            author_str = ", ".join(a.get('name', '') or f"{a.get('firstName', '')} {a.get('lastName', '')}" 
                                 for a in authors if a)
            date = metadata.get('date', '')
            if date:
                try:
                    date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y")
                except:
                    pass
            
            # Get additional metadata
            publisher = metadata.get('publisher', '')
            journal = metadata.get('publicationTitle', '')
            volume = metadata.get('volume', '')
            issue = metadata.get('issue', '')
            pages = metadata.get('pages', '')
            url = metadata.get('url', '')
            doi = metadata.get('DOI', '')
            item_type = metadata.get('itemType', 'document')
            
            # Create citation header with bibliographic format
            header = f"[Source {i+1}] "
            
            # Format citation based on item type
            if item_type.lower() in ('journalarticle', 'article'):
                citation_format = f"{source}"
                if author_str:
                    citation_format = f"{author_str}. {citation_format}"
                if date:
                    citation_format += f" ({date})"
                if journal:
                    citation_format += f". {journal}"
                    if volume:
                        citation_format += f", {volume}"
                        if issue:
                            citation_format += f"({issue})"
                    if pages:
                        citation_format += f", {pages}"
                if doi:
                    citation_format += f". DOI: {doi}"
                header += citation_format
            elif item_type.lower() in ('book', 'bookchapter'):
                citation_format = f"{source}"
                if author_str:
                    citation_format = f"{author_str}. {citation_format}"
                if date:
                    citation_format += f" ({date})"
                if publisher:
                    citation_format += f". {publisher}"
                if citation.page_number:
                    citation_format += f", p. {citation.page_number}"
                header += citation_format
            else:
                # Default format for other types
                header += f"{source}"
                if author_str:
                    header += f" by {author_str}"
                if date:
                    header += f" ({date})"
                if citation.page_number:
                    header += f" - Page {citation.page_number}"
            
            # Add reference ID for in-text citations
            header += f"\n[Reference ID: {i+1}]"
            
            # Add confidence score
            header += f"\nConfidence: {citation.confidence_score:.1f}%"
            
            # Format the citation text with context
            citation_text = ""
            if citation.context_before:
                citation_text += f"Context before: \"{citation.context_before}\"\n"
            citation_text += f"Content: \"{citation.text}\""
            if citation.context_after:
                citation_text += f"\nContext after: \"{citation.context_after}\""
            
            # Calculate token count
            part = f"{header}\n{citation_text}\n\n"
            part_tokens = len(part) / char_per_token
            
            if total_length + part_tokens > max_tokens:
                # Truncate to fit within token limit
                remaining_tokens = max_tokens - total_length
                truncated_length = int(remaining_tokens * char_per_token)
                part = part[:truncated_length] + "..."
            
            context_parts.append(part)
            total_length += part_tokens
            
            if total_length >= max_tokens:
                break
        
        # Add contradiction information if any
        if search_result.contradictions and total_length < max_tokens:
            contra_parts = ["\nPotential Contradictions:"]
            for c1, c2 in search_result.contradictions:
                # Get citation details for better references
                source1 = c1.metadata.get('title', 'Unknown source')
                author1 = ", ".join(a.get('name', '') or f"{a.get('firstName', '')} {a.get('lastName', '')}" 
                                  for a in c1.metadata.get('creators', []) if a)
                
                source2 = c2.metadata.get('title', 'Unknown source')
                author2 = ", ".join(a.get('name', '') or f"{a.get('firstName', '')} {a.get('lastName', '')}" 
                                  for a in c2.metadata.get('creators', []) if a)
                
                contra_parts.append(
                    f"- Contradiction between sources:\n"
                    f"  1. {source1}" + (f" by {author1}" if author1 else "") + f": \"{c1.text}\"\n"
                    f"  2. {source2}" + (f" by {author2}" if author2 else "") + f": \"{c2.text}\"\n"
                )
            contradiction_text = "\n".join(contra_parts)
            if (total_length + len(contradiction_text) / char_per_token) < max_tokens:
                context_parts.append(contradiction_text)
        
        return "\n".join(context_parts) 