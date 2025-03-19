import os
import pickle
from typing import List, Dict, Any
import logging
import faiss
import numpy as np
import time
from openai import OpenAI
import glob

from config import OPENAI_API_KEY, EMBEDDING_MODEL, VECTOR_DB_PATH

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model: str = EMBEDDING_MODEL, vector_db_path: str = VECTOR_DB_PATH):
        """Initialize the embedder with OpenAI client and vector database path."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.vector_db_path = vector_db_path
        self.index_path = os.path.join(vector_db_path, "faiss_index")
        self.documents_path = os.path.join(vector_db_path, "documents.pkl")
        self.registry_path = os.path.join(vector_db_path, "document_registry.pkl")
        
        # Add backup paths
        self.backup_dir = os.path.join(vector_db_path, "backups")
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Create directory if it doesn't exist
        os.makedirs(vector_db_path, exist_ok=True)
        
        # Initialize or load index and documents
        self.index = None
        self.documents = []
        self.document_registry = {}  # Maps file paths to last modified timestamps
        self._load_or_create_index()
        
        # Track embedding metrics
        self.metrics = {
            "api_calls": 0,
            "total_chunks": 0,
            "processed_chunks": 0,
            "start_time": None,
            "estimated_cost": 0.0,  # Based on OpenAI's pricing
            "rate_limit_delays": 0,
            "last_checkpoint": None,  # Track when we last saved a checkpoint
            "checkpoint_interval": 300  # Save checkpoint every 5 minutes
        }
        
        # Add query embedding cache
        self.embedding_cache = {}
        self.cache_size_limit = 1000  # Maximum number of cached embeddings
        
        logger.info(f"Initialized embedder with model {model}")
    
    def _load_or_create_index(self):
        """Load existing index and documents or create new ones."""
        if os.path.exists(self.index_path) and os.path.exists(self.documents_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.documents_path, 'rb') as f:
                    self.documents = pickle.load(f)
                
                # Load document registry if it exists
                if os.path.exists(self.registry_path):
                    with open(self.registry_path, 'rb') as f:
                        self.document_registry = pickle.load(f)
                    logger.info(f"Loaded document registry with {len(self.document_registry)} tracked files")
                
                logger.info(f"Loaded existing index with {len(self.documents)} documents")
            except Exception as e:
                logger.error(f"Error loading index: {e}. Attempting to load from backup...")
                if not self._load_from_backup():
                    logger.error("Could not load from backup. Creating new index.")
                    self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        self.documents = []
        self.document_registry = {}
        # Create a new index - dimension 3072 for text-embedding-3-large
        dimension = 3072 if "3" in self.model else 1536
        self.index = faiss.IndexFlatL2(dimension)
        logger.info(f"Created new index with dimension {dimension}")
    
    def _save(self):
        """Save the index and documents to disk with periodic backups."""
        try:
            # Regular save
            faiss.write_index(self.index, self.index_path)
            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
            with open(self.registry_path, 'wb') as f:
                pickle.dump(self.document_registry, f)
            
            # Check if we need to create a backup
            current_time = time.time()
            if (self.metrics["last_checkpoint"] is None or 
                current_time - self.metrics["last_checkpoint"] >= self.metrics["checkpoint_interval"]):
                if self._create_backup():
                    self.metrics["last_checkpoint"] = current_time
            
            logger.info(f"Saved index with {len(self.documents)} documents and {len(self.document_registry)} tracked files")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False
    
    def _create_backup(self):
        """Create a backup of the current index and documents."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_index = os.path.join(self.backup_dir, f"faiss_index_{timestamp}")
        backup_docs = os.path.join(self.backup_dir, f"documents_{timestamp}.pkl")
        backup_registry = os.path.join(self.backup_dir, f"document_registry_{timestamp}.pkl")
        
        try:
            # Save current state to backup
            faiss.write_index(self.index, backup_index)
            with open(backup_docs, 'wb') as f:
                pickle.dump(self.documents, f)
            with open(backup_registry, 'wb') as f:
                pickle.dump(self.document_registry, f)
            
            # Clean up old backups (keep last 5)
            self._cleanup_old_backups()
            
            logger.info(f"Created backup at {timestamp}")
            return True
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    def _cleanup_old_backups(self):
        """Keep only the 5 most recent backups."""
        try:
            # Get all backup files
            index_backups = sorted(glob.glob(os.path.join(self.backup_dir, "faiss_index_*")))
            doc_backups = sorted(glob.glob(os.path.join(self.backup_dir, "documents_*.pkl")))
            registry_backups = sorted(glob.glob(os.path.join(self.backup_dir, "document_registry_*.pkl")))
            
            # Remove all but the 5 most recent backups
            for backups in [index_backups[:-5], doc_backups[:-5], registry_backups[:-5]]:
                for backup in backups:
                    try:
                        os.remove(backup)
                    except Exception as e:
                        logger.error(f"Error removing old backup {backup}: {e}")
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
    
    def _load_from_backup(self):
        """Attempt to load the most recent backup."""
        try:
            # Find most recent backup
            index_backups = sorted(glob.glob(os.path.join(self.backup_dir, "faiss_index_*")))
            doc_backups = sorted(glob.glob(os.path.join(self.backup_dir, "documents_*.pkl")))
            registry_backups = sorted(glob.glob(os.path.join(self.backup_dir, "document_registry_*.pkl")))
            
            if not (index_backups and doc_backups and registry_backups):
                return False
            
            # Load most recent backup
            self.index = faiss.read_index(index_backups[-1])
            with open(doc_backups[-1], 'rb') as f:
                self.documents = pickle.load(f)
            with open(registry_backups[-1], 'rb') as f:
                self.document_registry = pickle.load(f)
            
            # Save loaded backup as current state
            self._save()
            
            logger.info(f"Successfully loaded from backup with {len(self.documents)} documents")
            return True
        except Exception as e:
            logger.error(f"Error loading from backup: {e}")
            return False
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for the given text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        # Check cache first
        if text in self.embedding_cache:
            logger.info(f"Using cached embedding for text of length {len(text)}")
            return self.embedding_cache[text]
            
        try:
            # Track API call
            self.metrics["api_calls"] += 1
            
            # Calculate approximate cost (based on OpenAI's pricing)
            # Assuming ~$0.0001 per 1K tokens and avg 4 chars per token
            tokens = len(text) / 4
            self.metrics["estimated_cost"] += (tokens / 1000) * 0.0001
            
            # Make the API call
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embedding = response.data[0].embedding
            
            # Add to cache
            if len(self.embedding_cache) >= self.cache_size_limit:
                # Remove a random item if cache is full
                self.embedding_cache.pop(next(iter(self.embedding_cache)))
            self.embedding_cache[text] = embedding
            
            return embedding
        except Exception as e:
            if "rate limit" in str(e).lower():
                self.metrics["rate_limit_delays"] += 1
                logger.warning(f"Rate limit hit. Waiting before retry: {e}")
                time.sleep(5)  # Wait 5 seconds before retry
                return self.create_embedding(text)  # Retry
            logger.error(f"Error creating embedding: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the vector database.
        
        Args:
            documents (List[Dict[str, Any]]): List of documents with text and metadata
        """
        try:
            # Initialize start time if this is the first batch
            if self.metrics["start_time"] is None:
                self.metrics["start_time"] = time.time()
            
            # Update total chunks
            self.metrics["total_chunks"] += len(documents)
            
            # Create embeddings for each document
            embeddings = []
            for i, doc in enumerate(documents):
                embedding = self.create_embedding(doc["text"])
                embeddings.append(embedding)
                
                # Update processed chunks
                self.metrics["processed_chunks"] += 1
                
                # Log progress every 10 chunks
                if self.metrics["processed_chunks"] % 10 == 0:
                    self._log_embedding_progress()
            
            # Add to index
            embeddings_array = np.array(embeddings).astype('float32')
            self.index.add(embeddings_array)
            
            # Update document store - save document index position
            start_idx = len(self.documents)
            for i, doc in enumerate(documents):
                doc["index"] = start_idx + i
                self.documents.append(doc)
            
            # Save to disk
            self._save()
            
            logger.info(f"Added {len(documents)} documents to index")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def _log_embedding_progress(self):
        """Log embedding progress metrics."""
        if self.metrics["start_time"] is None or self.metrics["processed_chunks"] == 0:
            return
            
        elapsed = time.time() - self.metrics["start_time"]
        chunks_per_sec = self.metrics["processed_chunks"] / elapsed if elapsed > 0 else 0
        
        # Only calculate if we have processed some chunks
        if self.metrics["processed_chunks"] > 0 and self.metrics["total_chunks"] > 0:
            percent_done = self.metrics["processed_chunks"] / self.metrics["total_chunks"] * 100
            
            # Estimate time remaining
            if chunks_per_sec > 0:
                chunks_remaining = self.metrics["total_chunks"] - self.metrics["processed_chunks"]
                seconds_remaining = chunks_remaining / chunks_per_sec
                
                minutes, seconds = divmod(seconds_remaining, 60)
                time_remaining = f"{int(minutes)}m {int(seconds)}s"
            else:
                time_remaining = "calculating..."
                
            logger.info(
                f"Embedding progress: {self.metrics['processed_chunks']}/{self.metrics['total_chunks']} chunks "
                f"({percent_done:.2f}%) at {chunks_per_sec:.2f} chunks/sec. "
                f"Est. time remaining: {time_remaining}. "
                f"API calls: {self.metrics['api_calls']}, Est. cost: ${self.metrics['estimated_cost']:.4f}"
            )
    
    def get_embedding_progress(self) -> Dict[str, Any]:
        """
        Get current embedding progress metrics.
        
        Returns:
            Dict[str, Any]: Dictionary with progress metrics
        """
        # Calculate additional metrics
        metrics = self.metrics.copy()
        
        # Add percentage complete
        if metrics["total_chunks"] > 0:
            metrics["percent_complete"] = (metrics["processed_chunks"] / metrics["total_chunks"]) * 100
        else:
            metrics["percent_complete"] = 0
            
        # Add elapsed time
        if metrics["start_time"] is not None:
            metrics["elapsed_seconds"] = time.time() - metrics["start_time"]
            
            # Calculate processing rate
            if metrics["elapsed_seconds"] > 0:
                metrics["chunks_per_second"] = metrics["processed_chunks"] / metrics["elapsed_seconds"]
                
                # Estimate time remaining
                chunks_remaining = metrics["total_chunks"] - metrics["processed_chunks"]
                if metrics["chunks_per_second"] > 0:
                    metrics["estimated_seconds_remaining"] = chunks_remaining / metrics["chunks_per_second"]
                else:
                    metrics["estimated_seconds_remaining"] = None
            else:
                metrics["chunks_per_second"] = 0
                metrics["estimated_seconds_remaining"] = None
        else:
            metrics["elapsed_seconds"] = 0
            metrics["chunks_per_second"] = 0
            metrics["estimated_seconds_remaining"] = None
            
        return metrics
    
    def reset_metrics(self):
        """Reset embedding metrics."""
        self.metrics = {
            "api_calls": 0,
            "total_chunks": 0,
            "processed_chunks": 0,
            "start_time": None,
            "estimated_cost": 0.0,
            "rate_limit_delays": 0,
            "last_checkpoint": None,
            "checkpoint_interval": 300
        }
        logger.info("Embedding metrics reset")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query (str): Query string
            k (int, optional): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of relevant documents with scores
        """
        try:
            # Create embedding for query
            query_embedding = self.create_embedding(query)
            query_embedding_array = np.array([query_embedding]).astype('float32')
            
            # Search
            scores, indices = self.index.search(query_embedding_array, k)
            
            # Return results
            results = []
            for i, doc_idx in enumerate(indices[0]):
                if doc_idx < len(self.documents) and doc_idx >= 0:  # Valid index
                    doc = self.documents[doc_idx].copy()
                    doc["score"] = float(scores[0][i])
                    results.append(doc)
            
            logger.info(f"Found {len(results)} relevant documents for query")
            return results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def needs_processing(self, file_path: str) -> bool:
        """
        Check if a document needs to be processed based on its path and modification time.
        
        Args:
            file_path (str): Path to the document
            
        Returns:
            bool: True if the document needs processing, False otherwise
        """
        # If not in registry, it needs processing
        if file_path not in self.document_registry:
            return True
        
        # Check if file has been modified since last processing
        last_modified = os.path.getmtime(file_path)
        last_processed = self.document_registry.get(file_path, 0)
        
        # If file has been modified, it needs processing
        if last_modified > last_processed:
            return True
        
        logger.info(f"Skipping {os.path.basename(file_path)} - already processed and unchanged")
        return False
    
    def register_document(self, file_path: str):
        """
        Register a document as processed.
        
        Args:
            file_path (str): Path to the document
        """
        self.document_registry[file_path] = os.path.getmtime(file_path)
    
    def get_processed_document_count(self) -> int:
        """
        Get the number of documents that have been processed.
        
        Returns:
            int: Count of processed documents
        """
        # Simply return the count of documents in the registry
        return len(self.document_registry) 