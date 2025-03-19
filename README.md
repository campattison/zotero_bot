# Zotero PDF Chat

A simple tool to chat with your Zotero PDF library using AI.

## What This Does

- Connects to your local Zotero library
- Lets you ask questions about your PDFs
- Gives answers based on the content of your documents
- Provides citations from the source material

## Requirements

- Python 3.8 or higher
- OpenAI API key (requires payment)
- Local Zotero library with PDFs

## Setup Guide

### 1. Get the Code

```bash
git clone https://github.com/yourusername/zotero-pdf-chat.git
cd zotero-pdf-chat
```

### 2. Install Required Software

```bash
pip install -r requirements.txt
```

### 3. Set Up Your Configuration

Create a configuration file:
```bash
cp .env.example .env
```

Edit the `.env` file with your information:
```
OPENAI_API_KEY=your_openai_api_key_here
ZOTERO_STORAGE_PATH=/path/to/your/zotero/storage
```

**Finding your Zotero storage path:**
- Open Zotero
- Go to Edit → Preferences → Advanced → Files and Folders
- Look for "Data Directory Location"
- The storage folder is inside this directory at `/storage`

### 4. Start the Program

```bash
python app.py
```
If that doesn't work, try:
```bash
python3 app.py
```

The program will be available at http://localhost:7860 in your web browser.

## Using the Program

### First Run: Index Your PDFs

1. When you first start the app, go to the "Index PDFs" tab
2. Click "Start Indexing" (this may take time for large libraries)
3. Wait for indexing to complete

### Asking Questions

1. Go to the "Chat" tab
2. Type your question about your PDFs
3. The system will search your documents and provide an answer

## Example Questions

- "What are the major approaches to value alignment in the most recent technical safety papers in my library?"
- "Synthesize the key arguments for and against digital minds having moral status across my collection"
- "Compare formal verification methods applied to AI systems in papers published in the last two years"
- "What theoretical frameworks are emerging to address the interpretability problem in deep learning models?"
- "Trace the evolution of arguments about AI governance structures in my papers from 2020-2023"
- "Identify the methodological approaches used to evaluate safety benchmarks across my technical papers"

## Common Issues

- **Slow indexing:** Normal for large libraries - each PDF needs processing
- **API Key errors:** Check that your OpenAI API key is valid and has available credit
- **PDFs not found:** Verify your Zotero storage path is correct in the .env file

## Cost Information

This tool uses OpenAI's API which charges based on usage:
- Creating embeddings uses the text-embedding-3-large model
- Chat responses use the o3-mini model by default

Monitor your OpenAI dashboard to track costs.

## How It Works (Simple Version)

1. The program extracts text from your PDFs
2. It creates AI-readable versions (embeddings) of your documents
3. When you ask a question, it finds the most relevant parts of your PDFs
4. It uses AI to create an answer based on those relevant sections 