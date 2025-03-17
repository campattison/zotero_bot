# Quick Start Guide

This guide will help you get Zotero PDF Chat up and running quickly.

## 1. Prerequisites

Before you begin, make sure you have:
- Python 3.8 or higher installed
- A Zotero library with PDF files
- An OpenAI API key (get one at https://platform.openai.com/api-keys)

## 2. Installation

### Option A: Install from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/zotero-pdf-chat.git
   cd zotero-pdf-chat
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option B: Install with pip

Coming soon!

## 3. Configuration

1. Create your `.env` file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your information:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ZOTERO_STORAGE_PATH=/path/to/your/zotero/storage
   ```

   **Finding your Zotero storage path:**
   - In Zotero, go to Edit → Preferences → Advanced → Files and Folders
   - Look for "Data Directory Location" and note the path
   - The storage folder is usually inside this directory at `/storage`

## 4. Start the Application

Run the application:
```bash
python app.py
```

The web interface will be available at http://localhost:7860

## 5. First Steps

1. When you first run the application, go to the "Index PDFs" tab and click "Start Indexing"
2. Wait for the indexing process to complete (this may take a while for large libraries)
3. Once indexing is complete, go to the "Chat" tab and start asking questions about your PDFs!

## 6. Examples of Questions You Can Ask

- "What does [author] say about [specific topic]?"
- "Summarize the key findings from my papers on [topic]"
- "Find contradictory views on [topic] across my papers"
- "What methodologies are used to study [phenomenon] in my library?"
- "What are the limitations mentioned in papers about [topic]?"

## 7. Troubleshooting

- **Indexing is slow:** This is normal for large libraries. Each PDF needs to be processed and sent to OpenAI for embedding.
- **API Key issues:** Make sure your OpenAI API key is valid and has sufficient credit.
- **PDFs not found:** Double-check your Zotero storage path in the .env file.

## 8. Cost Considerations

Using this tool will incur costs on your OpenAI account:
- Embedding generation uses the text-embedding-3-large model
- Chat responses use the gpt-4o model by default

Monitor your usage on the OpenAI dashboard to avoid unexpected charges. 