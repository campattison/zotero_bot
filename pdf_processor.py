import os
from typing import List, Dict, Any
import logging
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """Initialize PDF processor with chunking parameters."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # More granular separation
        )
        logger.info(f"Initialized PDF processor with chunk size {chunk_size} and overlap {chunk_overlap}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove excessive spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Fix hyphenated words at line breaks
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Fix common PDF extraction artifacts
        text = re.sub(r'([a-z])- ([a-z])', r'\1\2', text, flags=re.IGNORECASE)  # Fix "word- breaks"
        text = re.sub(r'(?<=[a-z])\.(?=[A-Z])', '. ', text)  # Add space after period if missing
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Add space between words if missing
        
        # Normalize whitespace around punctuation
        text = re.sub(r'\s*([.,;:!?])\s*', r'\1 ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure paragraphs are properly separated
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    # Clean each page's text individually
                    page_text = self._clean_text(page_text)
                    text += page_text + "\n\n"  # Add two newlines between pages
            
            # Final cleaning pass on the entire document
            text = self._clean_text(text)
            
            logger.info(f"Extracted and cleaned {len(text)} characters from {pdf_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text (str): Text to chunk
            metadata (Dict[str, Any], optional): Metadata to include with each chunk
            
        Returns:
            List[Dict[str, Any]]: List of chunks with metadata
        """
        # Clean the text one final time before chunking
        text = self._clean_text(text)
        
        chunks = self.text_splitter.create_documents([text], [metadata or {}])
        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        
        # Convert LangChain documents to dictionaries
        result = []
        for i, chunk in enumerate(chunks):
            # Clean each chunk one final time
            cleaned_chunk = self._clean_text(chunk.page_content)
            result.append({
                "text": cleaned_chunk,
                "metadata": {**chunk.metadata, "chunk_id": i}
            })
        
        return result 