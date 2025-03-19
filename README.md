# Zotero PDF Chat

Chat with your Zotero PDF library using OpenAI's models. This tool creates embeddings of your PDF documents and uses them for retrieval-augmented generation to provide accurate, contextual responses to your questions.

## Features

- 📚 Seamless integration with your local Zotero library
- 🔍 Automatic indexing of PDF documents
- 💬 Natural language chat interface
- 🧠 Retrieval-augmented generation for accurate responses
- 📊 Citation of sources in responses
- 🚀 Built with Gradio for a clean, responsive UI

## Requirements

- Python 3.8+
- OpenAI API key
- Local Zotero library with PDF files

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/zotero-pdf-chat.git
   cd zotero-pdf-chat
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file based on the example:
   ```bash
   cp .env.example .env
   ```

4. Edit the `.env` file and add your OpenAI API key and Zotero storage path:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ZOTERO_STORAGE_PATH=path_to_your_zotero_storage
   ```

## Configuration

You can customize various settings in `config.py`:

- `EMBEDDING_MODEL`: The OpenAI embedding model to use (default: "text-embedding-3-large")
- `CHAT_MODEL`: The OpenAI model for chat (default: "gpt-4o")
- `CHUNK_SIZE`: Size of text chunks for embeddings (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `THEME`: Gradio theme (default: "default")

## Usage

1. Start the application:
   ```bash
   python app.py
   ```
   Sometimes users have to use: 
   ```bash
   python3 app.py
   ```

2. Open your browser and navigate to the listed localhost (e.g. http://localhost:7860)

3. The first time you run the app, you will have to index your PDFs in the index page. This may take some time depending on the size of your library.

4. Once indexing is complete, you can start chatting with your documents!

## How It Works

1. **PDF Processing**: The tool extracts text from your Zotero PDFs using PyPDF.
2. **Text Chunking**: The extracted text is split into manageable chunks with appropriate overlap.
3. **Embedding Generation**: Each chunk is embedded using OpenAI's embedding model.
4. **Vector Database**: Embeddings are stored in a FAISS vector database for efficient similarity search.
5. **Query Processing**: When you ask a question, it's embedded and used to retrieve relevant document chunks.
6. **Response Generation**: The OpenAI chat model generates a response based on the retrieved context.

## Project Structure

- `app.py`: Main Gradio application and UI
- `zotero_handler.py`: Integration with Zotero library
- `pdf_processor.py`: PDF text extraction and preprocessing
- `embedder.py`: Document embedding and vector database management
- `retriever.py`: Semantic search functionality
- `chat.py`: Chat interface and response generation
- `config.py`: Configuration settings

## Privacy and Security

This application processes your personal documents locally and makes API calls to OpenAI for embeddings and chat completions:

- Your PDF files never leave your local machine
- Only the text extracted from PDFs is sent to OpenAI's API
- The application stores processed document data in the `vector_db` directory
- API keys and personal paths are stored in the `.env` file (never commit this file to public repositories)

### Data Storage

- All processed data is stored locally in the `vector_db` directory
- This directory is excluded from git via `.gitignore`
- Your OpenAI API key is stored in the `.env` file, which is also excluded from git

### Setting Up Securely

Please follow our [Environment Setup Guide](ENV_SETUP.md) to configure your environment securely and prevent accidental exposure of sensitive information.

## Limitations

- Currently only supports local Zotero libraries, not Zotero web
- Requires an OpenAI API key (usage will incur costs)
- Processing large libraries may take significant time and API calls

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 