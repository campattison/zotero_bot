# Vector Database Directory

This directory stores the vector embeddings, document data, and index files used by the application.

## Contents

When the application is running, this directory will contain:

- `document_registry.pkl`: Registry mapping document IDs to metadata
- `documents.pkl`: Stored document text chunks and metadata
- `faiss_index`: FAISS vector index file for semantic search
- `backups/`: Backup files created during database updates

## Setup

These files are automatically created when you first process your PDFs. You don't need to manually create anything in this directory.

**Important**: All data in this directory is specific to your personal documents and should **not** be committed to a public repository. The `.gitignore` file is configured to exclude these files from git.

## Recreating the Database

If you need to recreate the vector database:

1. Delete all files in this directory (except for `.gitkeep` and this `README.md`)
2. Run the PDF processing function in the application

## Backups

Automatic backups are created in the `backups/` subdirectory. You can use these to restore previous versions of your vector database if needed. 