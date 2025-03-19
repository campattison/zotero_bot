import os
import logging
import gradio as gr
import threading
import time
from typing import List, Dict, Any, Tuple, Optional
import glob
import signal
import subprocess

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
TEST_MODE = False  # Set to False to process all documents
TEST_DOC_LIMIT = 20

# UI settings
THEME = "default"
APP_TITLE = "Zotero PDF Chat"
APP_DESCRIPTION = "Chat with your Zotero PDF library using OpenAI's models"
if TEST_MODE:
    APP_DESCRIPTION += f" (TEST MODE - Limited to {TEST_DOC_LIMIT} documents)"

def index_pdfs_thread(collection_id: Optional[int] = None, progress=gr.Progress()):
    """Function to run in a thread to index PDFs."""
    global indexing_status
    
    try:
        # Reset embedding metrics
        embedder.reset_metrics()
        
        # Get PDFs from selected collection or all PDFs
        if collection_id is not None:
            pdf_files = zotero.get_pdfs_in_collection(collection_id)
            collection_name = next((c["full_name"] for c in zotero.get_collections() if c["id"] == collection_id), "Unknown")
            logger.info(f"Found {len(pdf_files)} PDF files in collection: {collection_name}")
        else:
            pdf_files = zotero.get_all_pdfs()
            logger.info(f"Found {len(pdf_files)} PDF files in all Zotero storage")
        
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
        
        # Ensure everything is saved
        embedder._save()
        
        indexing_status["running"] = False
        logger.info(f"Indexing completed: {indexing_status['processed']} documents processed")
        
        # Return metrics for display
        return get_indexing_metrics()
        
    except Exception as e:
        logger.error(f"Error in indexing thread: {e}")
        indexing_status["running"] = False
        return f"Error: {str(e)}"

def get_folder_choices():
    """Get collection choices for the dropdown menu."""
    # Add "All Collections" option
    choices = ["All Collections"]
    
    # Get collections from Zotero
    logger.info("Getting collections from Zotero for dropdown")
    collections = zotero.get_collections()
    logger.info(f"Found {len(collections)} collections in Zotero database for dropdown")
    
    # Track collection names to handle duplicates
    collection_names = {}
    
    for collection in collections:
        full_path = collection["full_name"]
        pdf_count = collection["pdf_count"]
        
        if pdf_count > 0:
            # Check if this is a duplicate name
            if full_path in collection_names:
                # Already have this name, so we need to disambiguate
                logger.warning(f"Duplicate collection name found: {full_path}")
            else:
                # Add to tracking and choices
                collection_names[full_path] = pdf_count
                choices.append(f"{full_path} ({pdf_count} PDFs)")
    
    logger.info(f"Final dropdown choices: {len(choices)} items")
    return choices

def folder_choice_to_collection_id(choice: str) -> Optional[int]:
    """Convert a collection choice from the dropdown to a collection ID."""
    if choice == "All Collections":
        return None
        
    # Extract collection name and PDF count from the choice
    # Format is typically "Collection Name (X PDFs)"
    parts = choice.split(" (")
    if len(parts) < 2:
        logger.error(f"Invalid choice format: {choice}")
        return None
        
    collection_name = parts[0]
    pdf_count_str = parts[1].rstrip(" PDFs)")
    try:
        pdf_count = int(pdf_count_str)
    except ValueError:
        logger.error(f"Invalid PDF count in choice: {choice}")
        return None
    
    # Find the matching collection by name AND pdf count to handle duplicates
    collections = zotero.get_collections()
    matching_collections = []
    
    for collection in collections:
        if collection["full_name"] == collection_name and collection["pdf_count"] == pdf_count:
            matching_collections.append(collection)
    
    if len(matching_collections) == 1:
        # Single exact match found
        return matching_collections[0]["id"]
    elif len(matching_collections) > 1:
        # Multiple matches with same name and PDF count
        # Log the ambiguity but return the first match
        logger.warning(f"Multiple collections match '{collection_name}' with {pdf_count} PDFs. Using the first match.")
        return matching_collections[0]["id"]
    else:
        # Try a more lenient match - just by the end of the path
        for collection in collections:
            # Check if the collection name ends with our search term and has the right PDF count
            if collection["full_name"].endswith(collection_name) and collection["pdf_count"] == pdf_count:
                logger.info(f"Found collection {collection['full_name']} matching end path '{collection_name}' with {pdf_count} PDFs")
                return collection["id"]
    
    logger.error(f"No collection found matching '{collection_name}' with {pdf_count} PDFs")
    return None

def start_indexing(folder_choice: str, progress=gr.Progress()):
    """Start the indexing process in a thread."""
    if indexing_status["running"]:
        return "Indexing is already in progress. Please wait for it to complete or stop it."
    
    # Convert folder choice to collection ID
    collection_id = folder_choice_to_collection_id(folder_choice)
    
    # Start indexing in a thread
    threading.Thread(target=index_pdfs_thread, args=(collection_id, progress)).start()
    
    # Return initial status
    return "Indexing started. Please wait..."

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

def format_embedding_metrics():
    """Format embedding metrics for display."""
    # Get final metrics
    final_metrics = embedder.get_embedding_progress()
    
    # Check if we have a start time
    if indexing_status.get("start_time"):
        elapsed = time.time() - indexing_status["start_time"]
        minutes, seconds = divmod(elapsed, 60)
        time_str = f"{int(minutes)}m {int(seconds)}s"
    else:
        time_str = "N/A"
    
    if indexing_status.get("running"):
        status = "✅ Indexing in progress..."
    else:
        status = "✅ Indexing complete!"
    
    chunks_per_second = final_metrics.get("chunks_per_second", 0)
    
    return (
        f"{status}\n\n"
        f"📊 Metrics:\n"
        f"- Processed {indexing_status.get('processed', 0)} of {indexing_status.get('total', 0)} PDFs\n"
        f"- Total chunks: {final_metrics.get('processed_chunks', 0)}\n"
        f"- Total API calls: {final_metrics.get('api_calls', 0)}\n"
        f"- Estimated cost: ${final_metrics.get('estimated_cost', 0):.4f}\n"
        f"- Processing time: {time_str}\n"
        f"- Processing rate: {chunks_per_second:.2f} chunks/second"
    )

def get_indexing_metrics():
    """Get the current indexing metrics for display."""
    if not indexing_status.get("running", False):
        return format_embedding_metrics()
    
    # Calculate time remaining
    metrics = embedder.get_embedding_progress()
    time_remaining = "calculating..."
    if metrics.get("estimated_seconds_remaining"):
        minutes, seconds = divmod(metrics["estimated_seconds_remaining"], 60)
        time_remaining = f"{int(minutes)}m {int(seconds)}s"
    
    # Format progress message
    progress_desc = (
        f"Processing {indexing_status.get('current_file', 'files')} | "
        f"Files: {indexing_status.get('processed', 0)}/{indexing_status.get('total', 0)} | "
        f"Chunks: {metrics.get('processed_chunks', 0)}/{metrics.get('total_chunks', 0)} | "
        f"API Calls: {metrics.get('api_calls', 0)} | "
        f"Est. Cost: ${metrics.get('estimated_cost', 0):.4f} | "
        f"Time Remaining: {time_remaining}"
    )
    
    return progress_desc

# New function for simplified metrics display
def get_simple_metrics_display():
    """Generate a simplified, easy-to-read metrics display."""
    metrics = embedder.get_embedding_progress()
    
    # Get elapsed time
    if indexing_status.get("start_time"):
        elapsed = time.time() - indexing_status["start_time"]
        elapsed_min, elapsed_sec = divmod(int(elapsed), 60)
        elapsed_str = f"{elapsed_min:02d}:{elapsed_sec:02d}"
    else:
        elapsed_str = "00:00"
    
    # Calculate progress percentage
    if metrics.get("total_chunks", 0) > 0:
        progress_pct = (metrics.get("processed_chunks", 0) / metrics.get("total_chunks", 1)) * 100
    else:
        progress_pct = 0
    
    # Calculate time remaining
    if metrics.get("estimated_seconds_remaining"):
        remaining_min, remaining_sec = divmod(int(metrics.get("estimated_seconds_remaining", 0)), 60)
        remaining_str = f"{remaining_min:02d}:{remaining_sec:02d}"
    else:
        remaining_str = "--:--"
    
    # Create progress bar
    bar_width = 20
    filled = int(bar_width * progress_pct / 100)
    progress_bar = "█" * filled + "░" * (bar_width - filled)
    
    # Format the display
    status = "IN PROGRESS" if indexing_status.get("running", False) else "COMPLETE"
    
    display = f"""
## Embedding Progress: {status}

[{progress_bar}] {progress_pct:.1f}%

### Time
⏱️ Elapsed: {elapsed_str}
⏳ Remaining: {remaining_str}

### Cost
💰 Current: ${metrics.get('estimated_cost', 0):.4f}
📈 Estimated Total: ${(metrics.get('estimated_cost', 0) / max(progress_pct, 1) * 100) if progress_pct > 0 else 0:.4f}

### Progress
📁 Files: {indexing_status.get('processed', 0)}/{indexing_status.get('total', 0)}
📄 Chunks: {metrics.get('processed_chunks', 0)}/{metrics.get('total_chunks', 0)}
🔄 API Calls: {metrics.get('api_calls', 0)}
    """
    
    return display

def stop_app_servers():
    """Force stop any running Gradio servers on ports 7860 and 7861."""
    try:
        # Find processes using ports 7860 and 7861
        cmd = "lsof -t -i:7860,7861"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        
        if output:
            pids = output.split('\n')
            logger.info(f"Found {len(pids)} processes using app ports: {', '.join(pids)}")
            
            # Kill each process
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    logger.info(f"Killed process {pid}")
                except Exception as e:
                    logger.error(f"Failed to kill process {pid}: {e}")
        else:
            logger.info("No processes found using app ports")
            
        return "App server ports cleared"
    except subprocess.CalledProcessError:
        logger.info("No processes found using app ports")
        return "No processes to clean up"
    except Exception as e:
        logger.error(f"Error stopping app servers: {e}")
        return f"Error: {str(e)}"

def debug_selected_collection(folder_choice: str):
    """Debug the selected collection to help diagnose issues."""
    if not folder_choice or folder_choice == "All Collections":
        return "Please select a specific collection to debug"
    
    # Convert folder choice to collection ID
    collection_id = folder_choice_to_collection_id(folder_choice)
    if not collection_id:
        return f"Error: Could not find collection ID for {folder_choice}"
    
    # Get details about this collection
    collections = zotero.get_collections()
    collection_info = next((c for c in collections if c["id"] == collection_id), None)
    
    if not collection_info:
        return f"Error: Could not find collection information for ID {collection_id}"
    
    # Run debug function on the collection
    zotero.debug_collection_pdfs(collection_id)
    
    # Get PDFs that would be indexed
    pdf_files = zotero.get_pdfs_in_collection(collection_id)
    
    # Format debug information
    result = [
        f"## Debug Information for Collection",
        f"Collection: {collection_info['full_name']}",
        f"Collection ID: {collection_id}",
        f"Expected PDF Count: {collection_info['pdf_count']}",
        f"Actual PDFs Found: {len(pdf_files)}",
        f"Zotero Database: {zotero.sqlite_path}",
        f"Zotero Storage Path: {zotero.storage_path}",
        "",
        "### First 5 PDF files found (if any):"
    ]
    
    for i, pdf in enumerate(pdf_files[:5]):
        result.append(f"{i+1}. {pdf}")
        
    if len(pdf_files) > 5:
        result.append(f"...and {len(pdf_files) - 5} more")
    
    if len(pdf_files) == 0:
        result.append("No PDF files found in this collection!")
        result.append("")
        result.append("Possible reasons:")
        result.append("1. The PDFs aren't properly linked to items in this collection")
        result.append("2. The PDFs don't exist at the expected storage locations")
        result.append("3. There might be a mismatch between collection hierarchies")
        result.append("")
        result.append("Try searching for PDFs directly:")
        
        # Try searching for PDFs in storage
        storage_pattern = os.path.join(zotero.data_directory, "storage", "**", "*.pdf")
        storage_pdfs = glob.glob(storage_pattern, recursive=True)
        result.append(f"Found {len(storage_pdfs)} PDFs anywhere in the Zotero storage directory")
        if storage_pdfs:
            result.append("Sample paths:")
            for i, pdf in enumerate(storage_pdfs[:3]):
                result.append(f"- {pdf}")
            if len(storage_pdfs) > 3:
                result.append(f"...and {len(storage_pdfs) - 3} more")
    
    return "\n".join(result)

def estimate_collection_processing(folder_choice: str):
    """Estimate the time and cost to process an entire collection."""
    # Convert folder choice to collection ID
    collection_id = folder_choice_to_collection_id(folder_choice)
    
    # Get PDFs from selected collection or all PDFs
    if collection_id is not None:
        pdf_files = zotero.get_pdfs_in_collection(collection_id)
        collection_name = next((c["full_name"] for c in zotero.get_collections() if c["id"] == collection_id), "Unknown")
        logger.info(f"Estimating for {len(pdf_files)} PDF files in collection: {collection_name}")
    else:
        pdf_files = zotero.get_all_pdfs()
        logger.info(f"Estimating for {len(pdf_files)} PDF files in all Zotero storage")
    
    # Limit number of documents in test mode
    if TEST_MODE:
        total_available = len(pdf_files)
        pdf_files = pdf_files[:TEST_DOC_LIMIT]
        logger.info(f"TEST MODE: Limiting estimate to {len(pdf_files)} documents out of {total_available}")
    
    # Count documents that need processing vs. already processed
    to_process = []
    already_processed = []
    
    for pdf_file in pdf_files:
        if embedder.needs_processing(pdf_file):
            to_process.append(pdf_file)
        else:
            already_processed.append(pdf_file)
    
    # Get historical metrics if available
    avg_chunks_per_doc = 0
    avg_time_per_chunk = 0
    avg_cost_per_chunk = 0
    
    # Check if we have previous metrics to use
    processed_docs = embedder.get_processed_document_count()
    total_chunks = len(embedder.documents)
    
    if processed_docs > 0 and total_chunks > 0:
        # Use historical data for estimates
        avg_chunks_per_doc = total_chunks / processed_docs
        
        # Assume approximately 3 seconds per chunk on average as a baseline
        # (This can be adjusted based on system performance)
        avg_time_per_chunk = 3
        
        # Average cost based on approximately 1000 tokens per chunk at $0.0001 per 1K tokens
        avg_cost_per_chunk = 0.0001
    else:
        # No historical data, use conservative defaults
        avg_chunks_per_doc = 20  # Guess: average academic PDF might have ~20 chunks
        avg_time_per_chunk = 4   # Be conservative and assume 4 seconds per chunk
        avg_cost_per_chunk = 0.00015  # Be conservative on cost too
    
    # Calculate estimates
    est_chunks = len(to_process) * avg_chunks_per_doc
    est_total_time_seconds = est_chunks * avg_time_per_chunk
    est_total_cost = est_chunks * avg_cost_per_chunk
    
    # Format time estimate
    hours, remainder = divmod(est_total_time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_estimate = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    if hours == 0:
        time_estimate = f"{int(minutes)}m {int(seconds)}s"
    
    # Create estimate display
    if len(to_process) == 0:
        result = f"""
## Collection Estimate

✅ **All {len(pdf_files)} documents in this collection are already processed!**

No further processing needed. You can chat with these documents immediately.
        """
    else:
        result = f"""
## Collection Estimate

📚 **Collection Overview:**
- Total PDFs: {len(pdf_files)}
- Already processed: {len(already_processed)}
- Need processing: {len(to_process)}

⏱️ **Estimated Processing Time:**
- Approximately {time_estimate}

💰 **Estimated Cost:**
- Approximately ${est_total_cost:.2f}

These estimates are based on {processed_docs} previously processed documents.
Actual time and cost may vary based on document length and content.
        """
    
    return result

def create_ui():
    """Create the Gradio UI."""
    with gr.Blocks(theme=THEME) as demo:
        gr.Markdown(f"# {APP_TITLE}\n\n{APP_DESCRIPTION}")
        
        with gr.Tabs():
            # Chat Tab
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=4):
                        chat_history = gr.Chatbot(
                            [],
                            elem_id="chatbot",
                            bubble_full_width=False,
                            avatar_images=None,  # Use text avatars instead of images
                            height=600,
                            show_label=False,
                        )
                        with gr.Row():
                            with gr.Column(scale=8):
                                query = gr.Textbox(
                                    show_label=False,
                                    placeholder="Ask a question about your PDFs...",
                                    container=False
                                )
                            with gr.Column(scale=1):
                                with gr.Row():
                                    submit = gr.Button("Send", variant="primary")
                                    reset_chat = gr.Button("New Chat")
                
                # Define a function to handle message submission that immediately shows the user's message
                def user_message_handler(message, history):
                    # Add user message to history immediately
                    history = history + [(message, None)]
                    return history, ""
                
                # Define a function to process the bot's response
                def bot_response_handler(history):
                    if history and history[-1][1] is None:
                        user_message = history[-1][0]
                        # Process the actual response using chat_handler
                        updated_history, _ = chat_handler.chat(user_message, history[:-1])
                        # Replace the last history item with the complete one
                        history = history[:-1] + [(user_message, updated_history[-1][1])]
                    return history
                
                # Connect the UI elements to the functions
                submit.click(
                    fn=user_message_handler,
                    inputs=[query, chat_history],
                    outputs=[chat_history, query],
                ).then(
                    fn=bot_response_handler,
                    inputs=[chat_history],
                    outputs=[chat_history]
                )
                
                query.submit(
                    fn=user_message_handler,
                    inputs=[query, chat_history],
                    outputs=[chat_history, query],
                ).then(
                    fn=bot_response_handler,
                    inputs=[chat_history],
                    outputs=[chat_history]
                )
                
                def reset_chat_ui():
                    """Reset the chat UI and backend chat state."""
                    chat_handler.reset_chat()
                    return [], ""
                
                reset_chat.click(
                    reset_chat_ui,
                    inputs=[],
                    outputs=[chat_history, query]
                )
            
            # Index PDFs Tab
            with gr.Tab("Index PDFs"):
                with gr.Row():
                    with gr.Column():
                        folder_choice = gr.Dropdown(
                            label="Select Folder",
                            choices=get_folder_choices()
                        )
                        refresh_folders = gr.Button("Refresh Folder List")
                        
                        # Add estimate button
                        estimate_button = gr.Button("Estimate Processing")
                        
                        with gr.Row():
                            index_button = gr.Button("Start Indexing")
                            debug_button = gr.Button("Debug Selected Collection")
                            
                        with gr.Row():
                            clear_ports_button = gr.Button("Fix Port Issues")
                        
                        # Add note about stopping indexing
                        gr.Markdown("""
                        > **Note:** To stop indexing, press `Ctrl+C` in the terminal/command prompt where you started the app. 
                        > On Mac, you can also use `Command+C` or `Control+C`.
                        """)
                            
                        metrics_output = gr.Markdown("No metrics available.")
                
                # Add refresh functionality
                refresh_folders.click(
                    lambda: gr.Dropdown(choices=get_folder_choices()),
                    outputs=[folder_choice]
                )
                
                # Add estimate functionality
                estimate_button.click(
                    estimate_collection_processing,
                    inputs=[folder_choice],
                    outputs=[metrics_output]
                )
                
                # Update indexing functionality
                index_button.click(
                    start_indexing,
                    inputs=[folder_choice],
                    outputs=[metrics_output]
                )
                
                # Add debug functionality
                debug_button.click(
                    debug_selected_collection,
                    inputs=[folder_choice],
                    outputs=[metrics_output]
                )
                
                # Add port fix functionality
                clear_ports_button.click(
                    stop_app_servers,
                    inputs=[],
                    outputs=[metrics_output]
                )
                
                # Update to use simple metrics display with frequent updates
                gr.on(
                    triggers=[index_button.click],
                    fn=get_simple_metrics_display,
                    outputs=[metrics_output],
                    every=1  # Update every second
                )
    
    return demo

def main():
    # Launch the app
    app = create_ui()
    app.queue()
    app.launch(server_name="0.0.0.0")

if __name__ == "__main__":
    main() 