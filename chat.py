from typing import List, Dict, Any, Tuple
import logging
from openai import OpenAI
import re
from dataclasses import dataclass

from config import OPENAI_API_KEY, CHAT_MODEL
from retriever import Retriever, SearchResult

logger = logging.getLogger(__name__)

@dataclass
class ReasoningStep:
    """Represents a step in the reasoning process."""
    question: str
    search_result: SearchResult
    conclusion: str
    confidence: float

class ChatHandler:
    def __init__(self, retriever: Retriever):
        """
        Initialize chat handler with retriever.
        
        Args:
            retriever (Retriever): The retriever instance
        """
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.retriever = retriever
        self.model = CHAT_MODEL
        self.chat_history = []
        self.max_tokens = 8192
    
    def _get_completion_kwargs(self) -> Dict[str, Any]:
        """Get model completion arguments."""
        return {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens
        }
        
    def _break_into_subquestions(self, query: str) -> List[str]:
        """Break a complex query into simpler sub-questions."""
        messages = [
            {"role": "system", "content": "You are an expert at breaking down complex questions into simpler sub-questions. Respond with ONLY the sub-questions, one per line."},
            {"role": "user", "content": f"Break this question into sub-questions: {query}"}
        ]
        
        response = self.client.chat.completions.create(
            messages=messages,
            **self._get_completion_kwargs()
        )
        
        # Split response into lines and clean up
        subquestions = [q.strip() for q in response.choices[0].message.content.split('\n') if q.strip()]
        return subquestions
    
    def _analyze_step(self, question: str, search_result: SearchResult) -> str:
        """Analyze search results and draw conclusions for a single step."""
        # Format context from search results
        context = self.retriever.format_for_context(search_result)
        
        messages = [
            {"role": "system", "content": """You are an expert academic researcher analyzing scholarly materials. Your analysis should:

1. Be detailed, precise, and based strictly on the provided sources
2. Extract specific arguments, theories, and frameworks mentioned in the sources
3. Use proper academic citation when referencing specific points [Use format: Source X]
4. Connect related ideas across different sources
5. Identify gaps, contradictions, or limitations in the available information
6. Avoid making claims not supported by the provided sources
7. Prioritize specificity and depth over generalization

Remember: academic rigor requires specificity and proper attribution."""},
            {"role": "user", "content": f"""Question: {question}

Available Information:
{context}

Provide a detailed academic analysis of the information available on this question. Be specific about what each source contributes to our understanding, and identify any limitations in the available information."""}
        ]
        
        response = self.client.chat.completions.create(
            messages=messages,
            **self._get_completion_kwargs()
        )
        
        return response.choices[0].message.content
    
    def _synthesize_final_response(self, steps: List[ReasoningStep], original_query: str) -> str:
        """Synthesize the final response from all reasoning steps."""
        # Format all steps and their conclusions
        steps_text = ""
        for i, step in enumerate(steps):
            steps_text += f"\nStep {i+1}: {step.question}\n"
            steps_text += f"Confidence: {step.confidence:.1f}%\n"
            steps_text += f"Analysis: {step.conclusion}\n"
        
        # Extract all unique sources for reference tracking
        all_citations = []
        seen_citations = set()
        for step in steps:
            for citation in step.search_result.citations:
                # Use title and author as unique identifier
                metadata = citation.metadata
                title = metadata.get('title', 'Unknown')
                authors = metadata.get('creators', [])
                author_str = ", ".join(a.get('name', '') or f"{a.get('firstName', '')} {a.get('lastName', '')}" 
                                    for a in authors if a)
                citation_id = f"{title}_{author_str}"
                
                if citation_id not in seen_citations:
                    seen_citations.add(citation_id)
                    all_citations.append(citation)
        
        # Format citation data to include in prompt
        citation_details = ""
        for i, citation in enumerate(all_citations):
            metadata = citation.metadata
            source = metadata.get('title', 'Unknown source')
            authors = metadata.get('creators', [])
            author_str = ", ".join(a.get('name', '') or f"{a.get('firstName', '')} {a.get('lastName', '')}" 
                                 for a in authors if a)
            date = metadata.get('date', '')
            item_type = metadata.get('itemType', 'document')
            publisher = metadata.get('publisher', '')
            journal = metadata.get('publicationTitle', '')
            
            citation_details += f"Citation {i+1}: {source}"
            if author_str:
                citation_details += f" by {author_str}"
            if date:
                citation_details += f" ({date})"
            if journal:
                citation_details += f", {journal}"
            if publisher:
                citation_details += f", {publisher}"
            citation_details += f". Type: {item_type}\n"
        
        messages = [
            {"role": "system", "content": """You are an expert academic researcher synthesizing research findings. Your output MUST follow this exact structure:

1. Start with a concise introduction to the topic
2. Present detailed analysis with in-text citations using the format [Ref: X]
3. For each major point, cite specific sources that support it
4. Clearly state when different sources provide contradictory information
5. End with a conclusion summarizing the main findings
6. ALWAYS include a "References" section at the end with numbered citations
7. Format each reference properly using academic citation style

The References section is MANDATORY. Never skip it. Format each reference using detailed bibliographic information.
Your analysis should be detailed, nuanced, and focused on the specific question asked."""},
            {"role": "user", "content": f"""Original Question: {original_query}

Reasoning Steps:{steps_text}

Available Citations:
{citation_details}

Write a comprehensive academic response that follows the required structure:
1. Introduction
2. Detailed analysis with in-text citations [Ref: X]
3. Conclusion
4. References section (MANDATORY)

Make your response as detailed and informative as possible while maintaining academic rigor."""}
        ]
        
        response = self.client.chat.completions.create(
            messages=messages,
            **self._get_completion_kwargs()
        )
        
        response_text = response.choices[0].message.content
        
        # Check if References section exists, if not, add it ourselves
        if "References" not in response_text and "REFERENCES" not in response_text:
            references_section = "\n\nReferences:\n"
            for i, citation in enumerate(all_citations):
                metadata = citation.metadata
                
                # Format citation based on type
                ref_text = ""
                source = metadata.get('title', 'Unknown source')
                authors = metadata.get('creators', [])
                author_str = ", ".join(a.get('name', '') or f"{a.get('firstName', '')} {a.get('lastName', '')}" 
                                     for a in authors if a)
                date = metadata.get('date', '')
                item_type = metadata.get('itemType', 'document')
                publisher = metadata.get('publisher', '')
                journal = metadata.get('publicationTitle', '')
                volume = metadata.get('volume', '')
                issue = metadata.get('issue', '')
                pages = metadata.get('pages', '')
                
                if item_type.lower() in ('journalarticle', 'article'):
                    ref_text = f"{i+1}. {author_str if author_str else 'Unknown Author'}. "
                    ref_text += f"({date if date else 'n.d.'}). "
                    ref_text += f"{source}. "
                    if journal:
                        ref_text += f"{journal}"
                        if volume:
                            ref_text += f", {volume}"
                            if issue:
                                ref_text += f"({issue})"
                        if pages:
                            ref_text += f", {pages}"
                    ref_text += "."
                elif item_type.lower() in ('book', 'bookchapter'):
                    ref_text = f"{i+1}. {author_str if author_str else 'Unknown Author'}. "
                    ref_text += f"({date if date else 'n.d.'}). "
                    ref_text += f"{source}. "
                    if publisher:
                        ref_text += f"{publisher}."
                else:
                    ref_text = f"{i+1}. {author_str if author_str else 'Unknown Author'}. "
                    ref_text += f"({date if date else 'n.d.'}). "
                    ref_text += f"{source}."
                    
                references_section += f"{ref_text}\n"
            
            response_text += references_section
        
        return response_text
    
    def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
        """
        Process a chat message using multi-step reasoning.
        
        Args:
            message (str): User's message
            history (List[Tuple[str, str]]): Chat history
            
        Returns:
            Tuple[List[Tuple[str, str]], str]: Updated history and response
        """
        try:
            # Create a clean copy of history to avoid modifying the original
            history_copy = list(history)
            
            # First, determine if this is a simple query or needs decomposition
            is_complex_query = self._is_complex_query(message)
            
            if is_complex_query:
                # Break query into sub-questions
                subquestions = self._break_into_subquestions(message)
                
                # Process each sub-question
                reasoning_steps = []
                for question in subquestions:
                    # Search for relevant information
                    search_result = self.retriever.search(question, k=5)
                    
                    # Analyze results and draw conclusions
                    conclusion = self._analyze_step(question, search_result)
                    
                    # Record reasoning step
                    step = ReasoningStep(
                        question=question,
                        search_result=search_result,
                        conclusion=conclusion,
                        confidence=search_result.confidence_score
                    )
                    reasoning_steps.append(step)
                
                # Synthesize final response
                response = self._synthesize_final_response(reasoning_steps, message)
            else:
                # For simple queries, just do a direct search and analysis
                search_result = self.retriever.search(message, k=5)
                conclusion = self._analyze_step(message, search_result)
                
                # Create a single reasoning step
                step = ReasoningStep(
                    question=message,
                    search_result=search_result,
                    conclusion=conclusion,
                    confidence=search_result.confidence_score
                )
                
                # Use the direct conclusion as the response
                response = conclusion
            
            # Update history copy
            history_copy.append((message, response))
            
            return history_copy, response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            error_response = "I apologize, but I encountered an error while processing your question. Please try rephrasing or ask another question."
            # Create a clean copy of history to avoid modifying the original
            history_copy = list(history)
            history_copy.append((message, error_response))
            return history_copy, error_response

    def _is_complex_query(self, query: str) -> bool:
        """Always treat all queries as complex for deeper analysis."""
        return True

    def reset_chat(self):
        """Reset the chat history."""
        self.chat_history = [] 