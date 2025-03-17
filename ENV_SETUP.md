# Environment Setup Guide

This guide explains how to set up your environment variables for the Zotero PDF Chat application.

## Creating your .env file

Create a file named `.env` in the root directory of the project with the following content:

```
# OpenAI API Key (Required)
OPENAI_API_KEY=your_openai_api_key_here

# Path to your Zotero storage folder (Required)
# This is typically located at:
# - macOS: ~/Zotero/storage
# - Windows: C:\Users\{username}\Zotero\storage
# - Linux: ~/Zotero/storage
ZOTERO_STORAGE_PATH=/path/to/your/zotero/storage
```

## Obtaining API Keys

### OpenAI API Key

1. Go to [OpenAI's API platform](https://platform.openai.com/)
2. Sign up for an account or log in
3. Navigate to the API keys section
4. Create a new API key
5. Copy the key and paste it in your `.env` file

## Finding Your Zotero Storage Path

### macOS

The default Zotero storage path is: `~/Zotero/storage`

To find the exact path:
1. Open Zotero
2. Go to Preferences (⌘,)
3. Click on Advanced
4. Click on "Show Data Directory"
5. Navigate to the "storage" folder within the opened directory
6. Copy the full path to this folder

### Windows

The default Zotero storage path is: `C:\Users\{username}\Zotero\storage`

To find the exact path:
1. Open Zotero
2. Go to Edit > Preferences
3. Click on Advanced
4. Click on "Show Data Directory"
5. Navigate to the "storage" folder within the opened directory
6. Copy the full path to this folder

### Linux

The default Zotero storage path is: `~/Zotero/storage`

To find the exact path:
1. Open Zotero
2. Go to Edit > Preferences
3. Click on Advanced
4. Click on "Show Data Directory"
5. Navigate to the "storage" folder within the opened directory
6. Copy the full path to this folder

## Testing Your Configuration

After setting up your `.env` file, you can verify your configuration by running:

```bash
python -c "from config import OPENAI_API_KEY, ZOTERO_STORAGE_PATH; print(f'OpenAI API Key set: {bool(OPENAI_API_KEY)}\\nZotero storage path: {ZOTERO_STORAGE_PATH}')"
```

If successful, you should see:
```
OpenAI API Key set: True
Zotero storage path: /your/path/to/zotero/storage
``` 