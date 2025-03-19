from typing import List, Dict, Any, Tuple
import logging
from openai import OpenAI
import re
from dataclasses import dataclass

from config import OPENAI_API_KEY, CHAT_MODEL, REASONING_EFFORT
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
        self.reasoning_effort = REASONING_EFFORT
        self.chat_history = []
        self.max_tokens = 8192
        self.in_clarification_stage = False
        self.original_query = None
    
    def _get_completion_kwargs(self) -> Dict[str, Any]:
        """Get model completion arguments."""
        return {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": self.max_tokens
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
2. Organize your analysis with relevant section headers that reflect key themes (use ## for headers)
3. Extract specific arguments, theories, and frameworks discussed in the scholarly literature
4. Use **bold** for key concepts and *italics* for emphasis where appropriate (but sparingly)
5. Use proper academic citation when referencing specific points using the format [Author Year]
6. Identify connections and tensions between different scholarly perspectives
7. Acknowledge limitations in the available literature when appropriate

Section headers should be meaningful and reflect the content they introduce rather than generic structural labels. Aim for 2-4 headers that organize the content by topic for better readability.

Remember: scholarly rigor requires specificity, proper attribution, and well-organized presentation."""},
            {"role": "user", "content": f"""Question: {question}

Available Information:
{context}

Provide a detailed scholarly analysis of the information available on this question. Organize your response with appropriate section headers to group related concepts. Discuss how different sources contribute to our understanding of the topic, and identify any significant limitations or gaps in the literature."""}
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
            {"role": "system", "content": """You are an expert academic researcher synthesizing scholarly findings. Your output must maintain high standards of academic rigor while being well-organized and readable. Follow these guidelines:

1. Organize your analysis with meaningful section headers that reflect the content's themes and topics
2. Use headers to logically group related concepts and findings (e.g., "## Current Frameworks in AI Alignment" or "## Technical Approaches and Methodologies")
3. Use limited formatting like **bold** for key terms and *italics* for emphasis where appropriate
4. Incorporate in-text citations using the format [Author Year] consistently
5. When discussing key concepts, cite specific sources that contribute to the understanding
6. Present contrasting views when sources provide differing perspectives
7. Include a "## References" section at the end with properly formatted citations

Your headers should reflect the natural structure of the content rather than rigid academic sections. Use headers throughout your analysis to enhance readability while maintaining scholarly coherence. Make your headers descriptive of the specific content they introduce rather than generic structural labels.

Your analysis should be detailed, evidence-based, and focused specifically on addressing the research question."""},
            {"role": "user", "content": f"""Research Question: {original_query}

Reasoning Steps:{steps_text}

Available Citations:
{citation_details}

Synthesize a comprehensive scholarly analysis that addresses this research question with academic rigor and proper citations. Use appropriate section headers to organize the content by topic, and limited formatting (bold/italic) to improve readability."""}
        ]
        
        response = self.client.chat.completions.create(
            messages=messages,
            **self._get_completion_kwargs()
        )
        
        response_text = response.choices[0].message.content
        
        # Check if References section exists, if not, add it ourselves
        if "## References" not in response_text and "## REFERENCES" not in response_text:
            references_section = "\n\n## References\n"
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
    
    def _generate_clarification_questions(self, message: str) -> str:
        """
        Generate clarifying questions for the user's initial query.
        
        Args:
            message (str): User's initial message
            
        Returns:
            str: Clarifying questions to ask the user
        """
        messages = [
            {"role": "system", "content": """You are a professional researcher at a top-tier academic institution. Your role is to ask focused clarifying questions before providing a comprehensive analysis.

Your goal is to gather precise information about the user's research objectives to provide the most relevant scholarly analysis.

Based on their initial query, ask 2-3 specific questions that help clarify:
1. The specific focus of their research question or scholarly interest
2. Relevant theoretical frameworks, key authors, methodological approaches, or time periods they wish to examine
3. The scope and depth of analysis they require (e.g., comprehensive literature review, critical analysis of competing perspectives, etc.)

Maintain a professional scholarly tone. Briefly acknowledge their research area, then pose your questions concisely and precisely."""},
            {"role": "user", "content": f"I'm conducting research on the following topic or question: {message}"}
        ]
        
        response = self.client.chat.completions.create(
            messages=messages,
            **self._get_completion_kwargs()
        )
        
        return response.choices[0].message.content
    
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
            
            # Check if this is the first message in a conversation
            if len(history_copy) == 0:
                # This is the first message, enter clarification stage
                self.in_clarification_stage = True
                self.original_query = message
                
                # Generate clarifying questions
                response = self._generate_clarification_questions(message)
                
                # Update history copy
                history_copy.append((message, response))
                
                return history_copy, response
            elif self.in_clarification_stage:
                # This is the response to our clarification questions
                # Combine the original query with the clarification info, but keep it hidden from the user
                combined_query = f"""INITIAL RESEARCH QUERY: {self.original_query}

CLARIFICATIONS PROVIDED BY USER: {message}

Please provide a comprehensive scholarly analysis that addresses both the original query and incorporates the clarifications. Focus on academic rigor, proper citations, and providing nuanced analysis. Use appropriate section headers (## format) to organize content by topic rather than by document structure. Use limited formatting (**bold** for key terms, *italics* for emphasis) to enhance readability. Aim for meaningful section headers that reflect the specific themes and topics discussed."""
                
                # Exit clarification stage
                self.in_clarification_stage = False
                
                # Process the combined query internally, but return only the user's clarification message
                # to be displayed in the chat interface
                history_copy.append((message, None))  # Add user message but no response yet
                
                # Process the combined query
                internal_history = list(history_copy[:-1])  # Exclude the last message we just added
                _, response = self._process_query(combined_query, internal_history)
                
                # Update history with just the user's visible message and the response
                history_copy[-1] = (message, response)
                
                return history_copy, ""
            
            # Regular query processing for non-first messages
            return self._process_query(message, history_copy)
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            error_response = "I apologize, but I encountered an error while processing your question. Please try rephrasing or ask another question."
            # Create a clean copy of history to avoid modifying the original
            history_copy = list(history)
            history_copy.append((message, error_response))
            return history_copy, error_response

    def _process_query(self, message: str, history_copy: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
        """Process a query and generate a response."""
        try:
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
            # Add error response to history
            history_copy.append((message, error_response))
            return history_copy, error_response

    def _is_complex_query(self, query: str) -> bool:
        """Always treat all queries as complex for deeper analysis."""
        return True

    def reset_chat(self):
        """Reset the chat history."""
        self.chat_history = []
        self.in_clarification_stage = False
        self.original_query = None 