import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "o3-mini"  # Using o3-mini for enhanced reasoning capabilities
REASONING_EFFORT = "high"  # Can be "low", "medium", or "high"

# Zotero settings
ZOTERO_STORAGE_PATH = os.getenv("ZOTERO_STORAGE_PATH", "")

# Vector DB settings
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "vector_db")

# Embedding settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# UI settings
THEME = "default"
APP_TITLE = "Zotero PDF Chat"
APP_DESCRIPTION = "Chat with your Zotero PDF library using OpenAI's o3-mini reasoning model" 