import os
import glob
import re
import sqlite3
import logging
import shutil
import tempfile
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ZoteroHandler:
    """Handler for interacting with Zotero storage."""
    
    def __init__(self):
        """Initialize Zotero handler."""
        # Default location of Zotero storage
        home_dir = os.path.expanduser("~")
        self.storage_path = os.path.join(home_dir, "Zotero", "storage")
        self.data_directory = os.path.join(home_dir, "Zotero")
        self.sqlite_path = os.path.join(self.data_directory, "zotero.sqlite")
        self.temp_db_path = None
        
        # Log storage path
        logger.info(f"Initialized Zotero handler with storage path: {self.storage_path}")
        logger.info(f"Zotero database path: {self.sqlite_path}")
        
        # Check if paths exist
        if not os.path.exists(self.storage_path):
            logger.warning(f"Zotero storage path not found: {self.storage_path}")
        
        if not os.path.exists(self.sqlite_path):
            logger.warning(f"Zotero database not found: {self.sqlite_path}")
    
    def _get_db_connection(self) -> Optional[sqlite3.Connection]:
        """
        Get a connection to the Zotero database.
        If the database is locked, create a temporary copy and connect to that.
        
        Returns:
            sqlite3.Connection: Connection to the database
        """
        if not os.path.exists(self.sqlite_path):
            logger.error(f"Zotero database not found: {self.sqlite_path}")
            return None
            
        try:
            # Try to connect to the original database
            conn = sqlite3.connect(self.sqlite_path)
            return conn
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                logger.warning("Zotero database is locked. Creating temporary copy...")
                
                # Create a temporary directory and copy the database
                temp_dir = tempfile.mkdtemp(prefix="zotero_bot_")
                self.temp_db_path = os.path.join(temp_dir, "zotero_copy.sqlite")
                
                try:
                    # Copy the database file to our temporary location
                    shutil.copy2(self.sqlite_path, self.temp_db_path)
                    logger.info(f"Created temporary database copy at {self.temp_db_path}")
                    
                    # Connect to the copy
                    conn = sqlite3.connect(self.temp_db_path)
                    return conn
                except Exception as copy_err:
                    logger.error(f"Error creating database copy: {copy_err}")
                    return None
            else:
                logger.error(f"Error connecting to database: {e}")
                return None
    
    def get_storage_subfolders(self) -> List[Dict[str, str]]:
        """
        Get a list of subfolders in the Zotero storage directory.
        
        Returns:
            List[Dict[str, str]]: List of dictionaries with folder info including path and name
        """
        try:
            subfolders = []
            # List direct children of the storage folder
            for item in os.listdir(self.storage_path):
                full_path = os.path.join(self.storage_path, item)
                if os.path.isdir(full_path):
                    # Check if there are any PDFs in this folder or its subfolders
                    pdf_count = len(glob.glob(os.path.join(full_path, "**", "*.pdf"), recursive=True))
                    if pdf_count > 0:
                        subfolders.append({
                            "name": item,
                            "path": full_path,
                            "pdf_count": pdf_count
                        })
            
            # Sort by name
            subfolders.sort(key=lambda x: x["name"])
            logger.info(f"Found {len(subfolders)} subfolders with PDFs in Zotero storage")
            return subfolders
        except Exception as e:
            logger.error(f"Error listing Zotero subfolders: {e}")
            return []
    
    def get_all_pdfs(self, subfolder_path: Optional[str] = None) -> List[str]:
        """
        Get paths to all PDF files in the Zotero storage or specified subfolder.
        
        Args:
            subfolder_path (Optional[str]): If provided, only search within this subfolder
        
        Returns:
            List[str]: List of full paths to PDF files
        """
        base_path = subfolder_path if subfolder_path else self.storage_path
        pdf_pattern = os.path.join(base_path, "**", "*.pdf")
        pdf_files = glob.glob(pdf_pattern, recursive=True)
        logger.info(f"Found {len(pdf_files)} PDF files in {os.path.basename(base_path) if subfolder_path else 'Zotero storage'}")
        return pdf_files
    
    def get_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata about a PDF from its path and filename.
        This enhanced version attempts to extract more bibliographic information
        from the Zotero storage structure and filename patterns.
        
        Args:
            pdf_path (str): Path to the PDF file
        
        Returns:
            Dict[str, Any]: Metadata including title, authors, and other citation info
        """
        # Basic file metadata
        filename = os.path.basename(pdf_path)
        last_modified = os.path.getmtime(pdf_path)
        size_bytes = os.path.getsize(pdf_path)
        
        # Get parent folder name which often contains the item key or citation info
        parent_folder = os.path.basename(os.path.dirname(pdf_path))
        
        # Default metadata
        metadata = {
            "filename": filename,
            "path": pdf_path,
            "last_modified": last_modified,
            "size_bytes": size_bytes,
            "item_key": parent_folder,  # Zotero item key is usually the folder name
            "title": self._extract_title_from_filename(filename),
            "itemType": "document",
            "creators": self._extract_authors_from_filename(filename),
            "date": self._extract_year_from_filename(filename)
        }
        
        # Try to extract more structured metadata if available
        if self._is_structured_zotero_path(pdf_path):
            additional_metadata = self._extract_structured_metadata(pdf_path)
            metadata.update(additional_metadata)
        
        return metadata
    
    def _extract_title_from_filename(self, filename: str) -> str:
        """Extract a probable title from the filename."""
        # Remove file extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # Remove common Zotero attachment indicators
        name_without_ext = re.sub(r'_?(attachments?|supplement(al)?s?|appendix|appendices)_?', '', name_without_ext, flags=re.IGNORECASE)
        
        # Remove year pattern if present
        name_without_ext = re.sub(r'[\(\[]?(19|20)\d{2}[\)\]]?', '', name_without_ext)
        
        # Remove author pattern if it appears to be at the beginning (Last, First)
        name_without_ext = re.sub(r'^[A-Z][a-z]+,\s+[A-Z][a-z]+\s+-\s+', '', name_without_ext)
        
        # Clean up extra spaces and punctuation
        name_without_ext = re.sub(r'\s+', ' ', name_without_ext).strip()
        name_without_ext = re.sub(r'_+', ' ', name_without_ext).strip()
        
        return name_without_ext if name_without_ext else "Unknown Title"
    
    def _extract_authors_from_filename(self, filename: str) -> List[Dict[str, str]]:
        """Extract probable authors from the filename."""
        # Look for author pattern: Last, First or Last
        authors = []
        
        # Try to find author pattern at the start: Last, First - 
        author_match = re.search(r'^([A-Z][a-z]+),\s+([A-Z][a-z]+)\s+-\s+', filename)
        if author_match:
            authors.append({
                "lastName": author_match.group(1),
                "firstName": author_match.group(2)
            })
            return authors
        
        # Try to find multiple authors pattern: LastName et al
        author_match = re.search(r'^([A-Z][a-z]+)\s+et\s+al', filename, re.IGNORECASE)
        if author_match:
            authors.append({
                "lastName": author_match.group(1),
                "firstName": ""
            })
            return authors
        
        # Try to find a single author's last name at the beginning
        author_match = re.search(r'^([A-Z][a-z]+)\s+[^\s,]+', filename)
        if author_match:
            authors.append({
                "lastName": author_match.group(1),
                "firstName": ""
            })
            return authors
        
        # If no authors found, provide a default
        if not authors:
            authors.append({
                "name": "Unknown Author"
            })
        
        return authors
    
    def _extract_year_from_filename(self, filename: str) -> str:
        """Extract publication year from the filename if present."""
        # Look for a year pattern: 1900-2099
        year_match = re.search(r'(19|20)\d{2}', filename)
        if year_match:
            return year_match.group(0)
        return ""
    
    def _is_structured_zotero_path(self, pdf_path: str) -> bool:
        """Check if this path follows Zotero's structured storage format."""
        # Zotero usually stores files in a format like: storage/ITEMKEY/filename.pdf
        # or in nested folders by collection
        parts = pdf_path.split(os.path.sep)
        return len(parts) >= 3 and "storage" in parts
    
    def _extract_structured_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a structured Zotero path.
        This is a placeholder for more advanced metadata extraction.
        In a complete implementation, this might parse Zotero's SQLite database
        or metadata.json files that accompany PDFs.
        """
        # This is a placeholder for more advanced extraction
        # For now, we'll just use more sophisticated filename patterns
        
        parent_folder = os.path.basename(os.path.dirname(pdf_path))
        filename = os.path.basename(pdf_path)
        
        metadata = {}
        
        # If the parent folder looks like a Zotero item key (8 characters, alphanumeric)
        if re.match(r'^[A-Z0-9]{8}$', parent_folder):
            metadata["item_key"] = parent_folder
            
            # Check for common journal article filename patterns:
            # Pattern: Journal Name_Volume_Issue_Pages
            journal_match = re.search(r'([A-Za-z\s]+)_(\d+)_(\d+)_(\d+-\d+)', filename)
            if journal_match:
                metadata["publicationTitle"] = journal_match.group(1).replace('_', ' ')
                metadata["volume"] = journal_match.group(2)
                metadata["issue"] = journal_match.group(3)
                metadata["pages"] = journal_match.group(4)
                metadata["itemType"] = "journalArticle"
        
        return metadata
    
    def get_collections(self) -> List[Dict[str, Any]]:
        """Get all Zotero collections with their hierarchy.
        
        Returns:
            List of collection dictionaries with name, key, parent, and PDF count.
        """
        # Get database connection
        conn = self._get_db_connection()
        if not conn:
            logger.error("Could not establish connection to Zotero database")
            return []
            
        try:
            logger.info(f"Connected to Zotero database")
            cursor = conn.cursor()
            
            # Get all collections
            logger.info("Executing query to get all collections")
            cursor.execute("""
                SELECT c.collectionID, c.collectionName, c.parentCollectionID, c.key
                FROM collections c
                ORDER BY c.collectionName
            """)
            
            raw_collections = cursor.fetchall()
            logger.info(f"Found {len(raw_collections)} raw collections in Zotero database")
            
            collections = []
            for col_id, name, parent_id, key in raw_collections:
                # Count PDFs in this collection
                try:
                    cursor.execute("""
                        SELECT COUNT(DISTINCT ia.path) 
                        FROM itemAttachments ia
                        JOIN collectionItems ci ON ci.itemID = ia.parentItemID
                        WHERE ci.collectionID = ? AND ia.contentType = 'application/pdf'
                    """, (col_id,))
                    
                    pdf_count = cursor.fetchone()[0]
                    logger.info(f"Collection {name} has {pdf_count} PDFs")
                except Exception as e:
                    logger.warning(f"Error counting PDFs in collection {name}: {e}")
                    pdf_count = 0
                
                collections.append({
                    "id": col_id,
                    "name": name,
                    "parent_id": parent_id,
                    "key": key,
                    "pdf_count": pdf_count,
                    "path": None,  # Will be set after building hierarchy
                    "full_name": name  # Will be updated with full path
                })
            
            # Build the full path names
            collection_dict = {c["id"]: c for c in collections}
            
            for collection in collections:
                # Build the full path name (My Library > Parent > Child)
                path_parts = [collection["name"]]
                parent_id = collection["parent_id"]
                
                while parent_id and parent_id in collection_dict:
                    parent = collection_dict[parent_id]
                    path_parts.insert(0, parent["name"])
                    parent_id = parent["parent_id"]
                
                collection["full_name"] = " > ".join(path_parts)
            
            conn.close()
            logger.info(f"Successfully processed {len(collections)} collections from Zotero database")
            
            # Sort by full name
            collections.sort(key=lambda x: x["full_name"])
            
            # Clean up temporary database if needed
            if self.temp_db_path and os.path.exists(os.path.dirname(self.temp_db_path)):
                try:
                    os.remove(self.temp_db_path)
                    os.rmdir(os.path.dirname(self.temp_db_path))
                    logger.info(f"Cleaned up temporary database copy")
                except Exception as e:
                    logger.warning(f"Could not clean up temporary database: {e}")
            
            return collections
            
        except Exception as e:
            logger.error(f"Error getting collections: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Close connection
            if conn:
                conn.close()
                
            return []
    
    def get_pdfs_in_collection(self, collection_id: int) -> List[str]:
        """Get all PDF files in a specific Zotero collection.
        
        Args:
            collection_id: ID of the collection to get PDFs from.
            
        Returns:
            List of PDF file paths.
        """
        # Get database connection
        conn = self._get_db_connection()
        if not conn:
            logger.error("Could not establish connection to Zotero database")
            return []
            
        try:
            cursor = conn.cursor()
            
            # First, get collection info for logging
            cursor.execute("SELECT collectionName FROM collections WHERE collectionID = ?", (collection_id,))
            collection_name = cursor.fetchone()
            if collection_name:
                collection_name = collection_name[0]
                logger.info(f"Getting PDFs for collection: {collection_name} (ID: {collection_id})")
            
            # Get paths from attachments - modified query to only use existing columns
            cursor.execute("""
                -- Direct PDF attachments in the collection
                SELECT ia.path, ia.contentType, i.key
                FROM itemAttachments ia
                JOIN items i ON i.itemID = ia.itemID
                JOIN collectionItems ci ON ci.itemID = ia.itemID
                WHERE ci.collectionID = ? AND ia.contentType = 'application/pdf'
                
                UNION
                
                -- PDF attachments of parent items in the collection
                SELECT ia.path, ia.contentType, i.key
                FROM itemAttachments ia
                JOIN items i ON i.itemID = ia.itemID
                JOIN collectionItems ci ON ci.itemID = ia.parentItemID
                WHERE ci.collectionID = ? AND ia.contentType = 'application/pdf'
            """, (collection_id, collection_id))
            
            results = cursor.fetchall()
            logger.info(f"Found {len(results)} potential PDF entries in database for collection ID {collection_id}")
            
            # Process the paths and check if files exist
            pdf_paths = []
            for path, content_type, item_key in results:
                if not path:
                    logger.warning("Skipping empty path entry")
                    continue
                    
                try:
                    # Try multiple possible path constructions
                    possible_paths = []
                    
                    # Extract filename from path if present
                    filename = None
                    if ':' in path:
                        # Format: storage_key:filename
                        storage_key, file_part = path.split(':', 1)
                        filename = file_part
                        possible_paths.append(os.path.join(self.data_directory, "storage", storage_key, file_part))
                    else:
                        filename = os.path.basename(path)
                        if path.startswith("storage/"):
                            possible_paths.append(os.path.join(self.data_directory, path))
                        else:
                            possible_paths.append(os.path.join(self.storage_path, path))
                    
                    # Try with path-derived filename if available
                    if filename:
                        if ':' in path:
                            storage_key = path.split(':', 1)[0]
                            possible_paths.append(os.path.join(self.data_directory, "storage", storage_key, filename))
                        
                    # Try more direct paths by searching
                    if filename:
                        # Get the base filename without path
                        base_filename = os.path.basename(filename)
                        # Search for this file in storage directory
                        search_pattern = os.path.join(self.data_directory, "storage", "**", base_filename)
                        matching_files = glob.glob(search_pattern, recursive=True)
                        possible_paths.extend(matching_files)
                    
                    # 4. Remove doubles from possible paths
                    possible_paths = [p for p in possible_paths if "storage/storage" not in p]
                    
                    # Log path attempts
                    logger.info(f"Trying {len(possible_paths)} possible paths for {filename or path}")
                    for i, p in enumerate(possible_paths):
                        logger.info(f"  Path {i+1}: {p}")
                    
                    # Try all possible paths
                    found = False
                    for pdf_path in possible_paths:
                        if os.path.exists(pdf_path) and os.path.isfile(pdf_path):
                            pdf_paths.append(pdf_path)
                            logger.info(f"Found PDF at: {pdf_path}")
                            found = True
                            break
                    
                    if not found:
                        # Search entire directory for filenames that might match
                        if filename:
                            base_filename = os.path.basename(filename)
                            # More aggressive search pattern
                            search_pattern = os.path.join(self.data_directory, "**", base_filename)
                            matching_files = glob.glob(search_pattern, recursive=True)
                            
                            if matching_files:
                                pdf_paths.append(matching_files[0])
                                logger.info(f"Found PDF through wider search: {matching_files[0]}")
                            else:
                                logger.warning(f"Could not find PDF for {filename or path} after trying multiple paths")
                        else:
                            logger.warning(f"Could not find PDF for {path} after trying multiple paths")
                
                except Exception as e:
                    logger.error(f"Error processing path {path}: {e}")
            
            conn.close()
            logger.info(f"Found {len(pdf_paths)} accessible PDFs in collection {collection_id}")
            return pdf_paths
            
        except Exception as e:
            logger.error(f"Error getting PDFs in collection: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Close connection
            if conn:
                conn.close()
                
            return []
    
    def _get_item_metadata_from_database(self, storage_key: str) -> Optional[Dict[str, Any]]:
        """Get item metadata from Zotero database using storage key."""
        # Get database connection
        conn = self._get_db_connection()
        if not conn:
            return None
            
        try:
            cursor = conn.cursor()
            
            # Query to get item data based on storage key
            cursor.execute("""
                SELECT i.itemID, i.key, it.typeName, 
                       (SELECT value FROM itemData id JOIN fields f ON id.fieldID = f.fieldID 
                        WHERE id.itemID = i.itemID AND f.fieldName = 'title') as title,
                       (SELECT value FROM itemData id JOIN fields f ON id.fieldID = f.fieldID 
                        WHERE id.itemID = i.itemID AND f.fieldName = 'date') as date,
                       (SELECT value FROM itemData id JOIN fields f ON id.fieldID = f.fieldID 
                        WHERE id.itemID = i.itemID AND f.fieldName = 'publicationTitle') as publication
                FROM items i
                JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
                JOIN itemAttachments ia ON ia.parentItemID = i.itemID
                WHERE ia.key = ?
            """, (storage_key,))
            
            item_data = cursor.fetchone()
            
            if item_data:
                item_id, key, item_type, title, date, publication = item_data
                
                # Get creators (authors)
                cursor.execute("""
                    SELECT ct.typeName, c.firstName, c.lastName
                    FROM itemCreators ic
                    JOIN creators c ON ic.creatorID = c.creatorID
                    JOIN creatorTypes ct ON ic.creatorTypeID = ct.creatorTypeID
                    WHERE ic.itemID = ?
                    ORDER BY ic.orderIndex
                """, (item_id,))
                
                creators = []
                for creator_type, first_name, last_name in cursor.fetchall():
                    creators.append({
                        "creatorType": creator_type,
                        "firstName": first_name,
                        "lastName": last_name
                    })
                
                # Get tags
                cursor.execute("""
                    SELECT t.name
                    FROM itemTags it
                    JOIN tags t ON it.tagID = t.tagID
                    WHERE it.itemID = ?
                """, (item_id,))
                
                tags = [tag[0] for tag in cursor.fetchall()]
                
                conn.close()
                
                return {
                    "title": title if title else "Unknown Title",
                    "creators": creators if creators else [{"firstName": "", "lastName": "Unknown Author"}],
                    "date": date if date else "",
                    "itemType": item_type,
                    "publicationTitle": publication if publication else "",
                    "tags": tags
                }
                
            conn.close()
            
        except Exception as e:
            logger.error(f"Error getting item metadata from database: {e}")
            
            # Close connection
            if conn:
                conn.close()
                
        return None
    
    def debug_collection_pdfs(self, collection_id: int) -> None:
        """
        Debug helper to print detailed information about PDFs in a collection.
        
        Args:
            collection_id: ID of the collection to debug
        """
        # Get database connection
        conn = self._get_db_connection()
        if not conn:
            logger.error("Could not establish connection to Zotero database for debugging")
            return
        
        try:
            cursor = conn.cursor()
            
            # Get collection info
            cursor.execute("""
                SELECT collectionName, key
                FROM collections
                WHERE collectionID = ?
            """, (collection_id,))
            
            collection_info = cursor.fetchone()
            if not collection_info:
                logger.error(f"Collection ID {collection_id} not found in database")
                conn.close()
                return
            
            collection_name, collection_key = collection_info
            logger.info(f"Debugging collection: {collection_name} (ID: {collection_id}, Key: {collection_key})")
            
            # Get all storage folder keys
            all_storage_folders = []
            for item in os.listdir(os.path.join(self.data_directory, "storage")):
                if os.path.isdir(os.path.join(self.data_directory, "storage", item)) and len(item) == 8:
                    all_storage_folders.append(item)
            
            logger.info(f"Found {len(all_storage_folders)} storage folders in Zotero")
            
            # Get all items in this collection
            cursor.execute("""
                SELECT i.itemID, i.key, it.typeName
                FROM items i
                JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
                JOIN collectionItems ci ON ci.itemID = i.itemID
                WHERE ci.collectionID = ?
            """, (collection_id,))
            
            items = cursor.fetchall()
            logger.info(f"Found {len(items)} items in collection")
            
            # Get attachment information for these items
            pdf_count = 0
            found_files = []
            
            # First check attachments in the database
            for item_id, item_key, item_type in items:
                cursor.execute("""
                    SELECT ia.itemID, i.key, ia.path, i.itemID, ia.contentType
                    FROM itemAttachments ia
                    JOIN items i ON i.itemID = ia.itemID
                    WHERE ia.parentItemID = ?
                """, (item_id,))
                
                attachments = cursor.fetchall()
                for att_id, att_key, path, att_item_id, content_type in attachments:
                    if content_type == 'application/pdf':
                        pdf_count += 1
                        
                        # Extract filename from path
                        filename = None
                        if path and ':' in path:
                            filename = path.split(':', 1)[1]
                        elif path:
                            filename = os.path.basename(path)
                        
                        # Log the database information
                        logger.info(f"PDF {pdf_count} in database:")
                        logger.info(f"  Parent Item: {item_id} ({item_type}) - Key: {item_key}")
                        logger.info(f"  Attachment: {att_id} - Key: {att_key}")
                        logger.info(f"  DB Path: {path}")
                        logger.info(f"  Filename (extracted): {filename}")
                        
                        # Try all possible path constructions
                        possible_paths = []
                        
                        # Use storage key if available
                        if path and ':' in path:
                            storage_key, file_part = path.split(':', 1)
                            possible_paths.append(os.path.join(self.data_directory, "storage", storage_key, file_part))
                        
                        # Try with att_key folder
                        if filename:
                            possible_paths.append(os.path.join(self.data_directory, "storage", att_key, filename))
                        
                        # Try with parent item key folder
                        if filename:
                            possible_paths.append(os.path.join(self.data_directory, "storage", item_key, filename))
                        
                        # Try direct filename in storage subfolders
                        if filename:
                            for folder in all_storage_folders:
                                possible_paths.append(os.path.join(self.data_directory, "storage", folder, filename))
                        
                        # Remove any paths with double storage
                        possible_paths = [p for p in possible_paths if "storage/storage" not in p]
                        
                        # Try all possible paths
                        found = False
                        for p in possible_paths:
                            if os.path.exists(p) and os.path.isfile(p):
                                logger.info(f"  FOUND at: {p}")
                                found_files.append(p)
                                found = True
                                break
                        
                        if not found:
                            logger.warning(f"  NOT FOUND after trying {len(possible_paths)} possible paths")
            
            # Try direct search
            if len(found_files) == 0:
                logger.info("No PDFs found in the database. Trying direct file search...")
                
                # Try to find PDFs by searching for filenames related to this collection
                cursor.execute("""
                    SELECT value
                    FROM itemData id
                    JOIN fields f ON id.fieldID = f.fieldID
                    JOIN collectionItems ci ON ci.itemID = id.itemID
                    WHERE ci.collectionID = ? AND f.fieldName = 'title'
                """, (collection_id,))
                
                titles = [row[0] for row in cursor.fetchall()]
                logger.info(f"Found {len(titles)} titles in collection to search for")
                
                # Search for PDFs with these titles
                for title in titles[:10]:  # Limit to first 10 to avoid too much searching
                    if not title:
                        continue
                    
                    # Clean title for filename search
                    clean_title = title.replace(':', ' ').replace('/', ' ').replace('\\', ' ')
                    clean_title = ''.join(c for c in clean_title if c.isalnum() or c.isspace())
                    clean_title = clean_title.strip()
                    
                    if len(clean_title) < 5:  # Too short to be useful
                        continue
                    
                    logger.info(f"Searching for PDFs with title: {clean_title}")
                    
                    # Search for PDFs with this title
                    for folder in all_storage_folders:
                        folder_path = os.path.join(self.data_directory, "storage", folder)
                        for root, dirs, files in os.walk(folder_path):
                            for file in files:
                                if file.lower().endswith('.pdf') and clean_title.lower() in file.lower():
                                    pdf_path = os.path.join(root, file)
                                    logger.info(f"  Found potential match: {pdf_path}")
                                    found_files.append(pdf_path)
            
            conn.close()
            logger.info(f"Debug complete. Confirmed {len(found_files)} out of {pdf_count} PDFs in collection {collection_name}")
            
        except Exception as e:
            logger.error(f"Error debugging collection PDFs: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Close connection
            if conn:
                conn.close() 