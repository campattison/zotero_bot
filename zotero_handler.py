import os
import glob
from typing import List, Dict, Any
import logging

from config import ZOTERO_STORAGE_PATH

logger = logging.getLogger(__name__)

class ZoteroHandler:
    def __init__(self, storage_path: str = ZOTERO_STORAGE_PATH):
        """Initialize Zotero handler with the path to local storage."""
        self.storage_path = storage_path
        if not os.path.exists(storage_path):
            raise ValueError(f"Zotero storage path not found: {storage_path}")
        
        logger.info(f"Initialized Zotero handler with storage path: {storage_path}")
    
    def get_all_pdfs(self) -> List[str]:
        """
        Get paths to all PDF files in the Zotero storage.
        
        Returns:
            List[str]: List of full paths to PDF files
        """
        pdf_pattern = os.path.join(self.storage_path, "**", "*.pdf")
        pdf_files = glob.glob(pdf_pattern, recursive=True)
        logger.info(f"Found {len(pdf_files)} PDF files in Zotero storage")
        return pdf_files
    
    def get_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract basic metadata about a PDF from its path.
        This is a simplified version - in a real implementation,
        you might want to parse Zotero's SQLite database for proper metadata.
        
        Args:
            pdf_path (str): Path to the PDF file
        
        Returns:
            Dict[str, Any]: Basic metadata like filename and last modified date
        """
        filename = os.path.basename(pdf_path)
        last_modified = os.path.getmtime(pdf_path)
        size_bytes = os.path.getsize(pdf_path)
        
        return {
            "filename": filename,
            "path": pdf_path,
            "last_modified": last_modified,
            "size_bytes": size_bytes
        } 