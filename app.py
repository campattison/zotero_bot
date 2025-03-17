import os
import logging
import gradio as gr
import threading
import time
from typing import List, Dict, Any, Tuple

from config import APP_TITLE, APP_DESCRIPTION, THEME
from zotero_handler import ZoteroHandler
from pdf_processor import PDFProcessor
from embedder import Embedder
from retriever import Retriever
from chat import ChatHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
zotero = ZoteroHandler()
pdf_processor = PDFProcessor()
embedder = Embedder()
retriever = Retriever(embedder)
chat_handler = ChatHandler(retriever)

# Global variables
indexing_status = {"running": False, "processed": 0, "total": 0, "current_file": ""}
TEST_MODE = True  # Set to False to process all documents
TEST_DOC_LIMIT = 20

# UI settings
THEME = "default"
APP_TITLE = "Zotero PDF Chat"
APP_DESCRIPTION = "Chat with your Zotero PDF library using OpenAI's models"
if TEST_MODE:
    APP_DESCRIPTION += f" (TEST MODE - Limited to {TEST_DOC_LIMIT} documents)"

def index_pdfs_thread(progress=gr.Progress()):
    """Function to run in a thread to index PDFs."""
    global indexing_status
    
    try:
        # Reset embedding metrics
        embedder.reset_metrics()
        
        # Get all PDFs
        pdf_files = zotero.get_all_pdfs()
        logger.info(f"Found {len(pdf_files)} PDF files in Zotero storage")
        
        # Limit number of documents in test mode
        if TEST_MODE:
            total_available = len(pdf_files)
            pdf_files = pdf_files[:TEST_DOC_LIMIT]
            logger.info(f"TEST MODE: Limited processing to {len(pdf_files)} documents out of {total_available}")
        
        indexing_status["total"] = len(pdf_files)
        indexing_status["processed"] = 0
        indexing_status["running"] = True
        indexing_status["total_chunks"] = 0
        indexing_status["processed_chunks"] = 0
        indexing_status["start_time"] = time.time()
        
        # Process each PDF
        for i, pdf_file in enumerate(pdf_files):
            if not indexing_status["running"]:
                logger.info("Indexing process stopped by user")
                break  # Allow for cancellation
            
            logger.info(f"Document {i+1}/{len(pdf_files)}: {os.path.basename(pdf_file)}")
            indexing_status["current_file"] = os.path.basename(pdf_file)
            
            # Check if document needs processing
            if not embedder.needs_processing(pdf_file):
                indexing_status["processed"] += 1
                progress(indexing_status["processed"] / indexing_status["total"], desc=f"Skipping {indexing_status['current_file']} (already processed)")
                continue
            
            # Get metadata
            metadata = zotero.get_pdf_metadata(pdf_file)
            
            # Extract text
            text = pdf_processor.extract_text(pdf_file)
            if not text:
                logger.warning(f"No text extracted from {pdf_file}")
                continue
                
            # Chunk text
            chunks = pdf_processor.chunk_text(text, metadata)
            indexing_status["total_chunks"] += len(chunks)
            
            # Add to embeddings
            embedder.add_documents(chunks)
            
            # Register document as processed
            embedder.register_document(pdf_file)
            
            # Update progress with detailed metrics
            indexing_status["processed"] += 1
            indexing_status["processed_chunks"] = embedder.metrics["processed_chunks"]
            
            # Calculate time remaining
            metrics = embedder.get_embedding_progress()
            time_remaining = "calculating..."
            if metrics.get("estimated_seconds_remaining"):
                minutes, seconds = divmod(metrics["estimated_seconds_remaining"], 60)
                time_remaining = f"{int(minutes)}m {int(seconds)}s"
            
            # Update progress bar with detailed info
            progress_desc = (
                f"Processing {indexing_status['current_file']} | "
                f"Files: {indexing_status['processed']}/{indexing_status['total']} | "
                f"Chunks: {metrics['processed_chunks']}/{metrics['total_chunks']} | "
                f"API Calls: {metrics['api_calls']} | "
                f"Est. Cost: ${metrics['estimated_cost']:.4f} | "
                f"Time Remaining: {time_remaining}"
            )
            progress(indexing_status["processed"] / indexing_status["total"], desc=progress_desc)
            
        indexing_status["running"] = False
        
        # Get final metrics
        final_metrics = embedder.get_embedding_progress()
        elapsed = time.time() - indexing_status["start_time"]
        minutes, seconds = divmod(elapsed, 60)
        
        return (
            f"✅ Indexing complete! Processed {indexing_status['processed']} of {indexing_status['total']} PDFs.\n\n"
            f"📊 Metrics:\n"
            f"- Total chunks: {final_metrics['processed_chunks']}\n"
            f"- Total API calls: {final_metrics['api_calls']}\n"
            f"- Estimated cost: ${final_metrics['estimated_cost']:.4f}\n"
            f"- Processing time: {int(minutes)}m {int(seconds)}s\n"
            f"- Processing rate: {final_metrics['chunks_per_second']:.2f} chunks/second"
        )
        
    except Exception as e:
        indexing_status["running"] = False
        error_msg = f"❌ Error during indexing: {str(e)}"
        logger.error(error_msg)
        return error_msg

def index_pdfs_wrapper(progress=gr.Progress()):
    """Wrapper to start the indexing thread."""
    if indexing_status["running"]:
        return "⚠️ Indexing is already running."
    
    # Start indexing in a thread
    thread = threading.Thread(target=lambda: index_pdfs_thread(progress))
    thread.daemon = True
    thread.start()
    
    # Return initial status
    return "🔄 Indexing started..."

def stop_indexing():
    """Stop the indexing process."""
    global indexing_status
    if indexing_status["running"]:
        logger.info("Stopping indexing process...")
        indexing_status["running"] = False
        return "⏹️ Indexing stopped."
    else:
        return "ℹ️ No indexing process running."

def get_indexing_status():
    """Get the current indexing status."""
    global indexing_status
    test_mode_prefix = "[TEST MODE] " if TEST_MODE else ""
    
    if indexing_status["running"]:
        return f"{test_mode_prefix}🔄 Indexing in progress: {indexing_status['processed']}/{indexing_status['total']} PDFs processed. Current: {indexing_status['current_file']}"
    elif indexing_status["processed"] > 0:
        return f"{test_mode_prefix}✅ Indexing complete: {indexing_status['processed']}/{indexing_status['total']} PDFs processed."
    else:
        return f"{test_mode_prefix}ℹ️ No indexing has been performed yet."

def chat(message, history):
    """Handle chat messages and return the response."""
    if not message.strip():
        return history
    
    # Check if any documents are indexed
    if len(embedder.documents) == 0:
        logger.warning("Attempted chat but no documents indexed yet")
        history.append((message, "No documents have been indexed yet. Please go to the 'Index PDFs' tab and index some documents first."))
        return history
    
    # Log the query
    logger.info(f"Chat query: {message[:50]}..." if len(message) > 50 else f"Chat query: {message}")
    logger.info(f"Searching among {len(embedder.documents)} indexed documents")
    
    try:
        # Get response from the chat handler
        updated_history, _ = chat_handler.chat(message, history)
        
        # Return the updated history
        return updated_history
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)
        history.append((message, f"I encountered an error: {str(e)}. Please try again with a different question."))
        return history

def reset_chat_history():
    """Reset the chat history."""
    chat_handler.reset_chat()
    return []

def check_setup():
    """Check if the setup is correct."""
    issues = []
    
    if not os.path.exists(zotero.storage_path):
        issues.append(f"❌ Zotero storage path not found: {zotero.storage_path}")
    
    try:
        # Quick test of OpenAI API
        test_embedding = embedder.create_embedding("Test")
        if len(test_embedding) == 0:
            issues.append("❌ Failed to create test embedding with OpenAI API")
    except Exception as e:
        issues.append(f"❌ OpenAI API error: {str(e)}")
    
    if issues:
        return "\n".join(issues)
    else:
        return "✅ Setup looks good! You can now index your PDFs and start chatting."

# Add function to format embedding metrics for display
def format_embedding_metrics() -> str:
    """Format embedding metrics for display."""
    if len(embedder.documents) == 0:
        return "No documents indexed yet."
        
    metrics = embedder.get_embedding_progress()
    
    # If we have active indexing
    if indexing_status["running"]:
        elapsed = time.time() - metrics.get("start_time", time.time())
        minutes, seconds = divmod(elapsed, 60)
        
        time_remaining = "calculating..."
        if metrics.get("estimated_seconds_remaining"):
            rem_minutes, rem_seconds = divmod(metrics["estimated_seconds_remaining"], 60)
            time_remaining = f"{int(rem_minutes)}m {int(rem_seconds)}s"
            
        return (
            f"📊 Live Embedding Metrics:\n\n"
            f"📂 Files: {indexing_status['processed']}/{indexing_status['total']}\n"
            f"📄 Chunks: {metrics['processed_chunks']}/{metrics['total_chunks']} ({metrics.get('percent_complete', 0):.1f}%)\n"
            f"🔄 Processing rate: {metrics.get('chunks_per_second', 0):.2f} chunks/second\n"
            f"⏱️ Elapsed time: {int(minutes)}m {int(seconds)}s\n"
            f"⏳ Estimated time remaining: {time_remaining}\n"
            f"🔌 API calls: {metrics['api_calls']}\n"
            f"💰 Estimated cost: ${metrics['estimated_cost']:.4f}"
        )
    
    # If we have indexed documents but not currently indexing
    if len(embedder.documents) > 0:
        return (
            f"📊 Embedding Database Stats:\n\n"
            f"📄 Total chunks indexed: {len(embedder.documents)}\n"
            f"📂 Files in registry: {len(embedder.document_registry)}\n"
            f"💾 Vector database size: {len(embedder.documents) * (3072 if '3' in embedder.model else 1536) * 4 / (1024*1024):.2f} MB"
        )
    
    return "No metrics available."

def create_ui(chat_handler):
    """Create the Gradio interface."""
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(height=500)
        
        with gr.Row():
            msg = gr.Textbox(
                label="Chat Message",
                placeholder="Type your message here...",
                lines=3
            )
            
        with gr.Row():
            submit = gr.Button("Submit")
            clear = gr.Button("Clear")
        
        def handle_chat(message, history):
            """Handle chat messages."""
            updated_history, _ = chat_handler.chat(message, history)
            return "", updated_history  # Return empty string to clear input
        
        # Set up event handlers
        submit.click(
            handle_chat,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],  # Now also returning to msg to clear it
        )
        
        clear.click(lambda: None, None, chatbot, queue=False)
        
        msg.submit(
            handle_chat,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],  # Now also returning to msg to clear it
        )
    
    return demo

# Create Gradio interface
with gr.Blocks(title=APP_TITLE, theme=THEME) as app:
    gr.Markdown(f"# {APP_TITLE}")
    
    with gr.Tab("Chat"):
        chat_handler = ChatHandler(retriever)
        chat_interface = create_ui(chat_handler)
    
    with gr.Tab("Index PDFs"):
        # PDF indexing interface
        with gr.Row():
            with gr.Column():
                index_button = gr.Button("Start Indexing")
                stop_button = gr.Button("Stop Indexing")
                metrics_output = gr.Markdown("No metrics available.")
                
        index_button.click(
            index_pdfs_wrapper,
            outputs=[metrics_output]
        )
        
        stop_button.click(
            stop_indexing,
            outputs=[metrics_output]
        )

def main():
    # Launch the app
    app.queue()
    app.launch(server_name="0.0.0.0")

if __name__ == "__main__":
    main() 