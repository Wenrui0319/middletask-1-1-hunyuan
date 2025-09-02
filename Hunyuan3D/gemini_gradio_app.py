import gradio as gr
import subprocess
import os
import re
import uuid
import atexit
import shutil
from pathlib import Path
import json

# --- Constants and Configuration ---
GOOGLE_CLOUD_PROJECT = "enduring-sweep-465314-i1"
os.environ["GOOGLE_CLOUD_PROJECT"] = GOOGLE_CLOUD_PROJECT

TEMP_DIR = Path("temp_file_workspace")
TEMP_DIR.mkdir(exist_ok=True)

# --- Cleanup Function ---
def cleanup():
    """Remove the temporary directory upon script exit."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        print("Temporary workspace cleaned up.")

atexit.register(cleanup)

# --- Core Gemini CLI Interaction ---
def gemini_cli_interface(text_prompt, file_paths, history):
    """
    Calls the Gemini CLI, handling text, multiple files, and history.
    """
    # Prepare accessible paths for all uploaded files and get their basenames
    relative_file_paths = []
    for file_path in file_paths:
        try:
            # We need the original filename to copy TO the temp dir
            original_file = Path(file_path)
            # The destination path inside the temp workspace
            destination_path = TEMP_DIR / original_file.name
            shutil.copy(original_file, destination_path)
            # The path to be used in the prompt should be relative to the cwd
            relative_file_paths.append(original_file.name)
        except Exception as e:
            yield f"Error copying file '{file_path}' to workspace: {e}"
            return

    # System instruction for the model
    system_instruction = (
        "IMPORTANT: When you provide content that is meant to be copied, such as code snippets, "
        "configuration examples, or prompts, you MUST enclose it in a standard markdown code block "
        "with a language specifier. For example:\n"
        "```text\n"
        "This is a prompt that can be copied.\n"
        "```\n"
        "For code, use the appropriate language, like ```python or ```javascript."
    )
    
    # Build the full prompt with history
    full_prompt = f"System: {system_instruction}\n\n"
    if history:
        for message in history:
            role = "User" if message["role"] == "user" else "Model"
            # Clean content for the prompt history
            clean_content = re.sub('<[^<]+?>', '', message.get("content", ""))
            full_prompt += f"{role}: {clean_content}\n"
    
    full_prompt += f"User: {text_prompt}"

    # If there are files, their relative paths are mentioned in the prompt
    if relative_file_paths:
        full_prompt += " " + " ".join(relative_file_paths)

    # Construct the command
    command = ["gemini", "-p", full_prompt]

    try:
        # Run the command with the CWD set to the temporary workspace
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=TEMP_DIR
        )
        
        # Stream stdout
        for line in iter(process.stdout.readline, ''):
            yield line
        
        process.stdout.close()
        
        stderr_output = process.stderr.read()
        process.stderr.close()
        
        if process.wait() != 0:
            yield f"Error executing Gemini CLI:\n{stderr_output}"

    except FileNotFoundError:
        yield "Error: 'gemini' command not found. Make sure the Gemini CLI is installed and in your PATH."
    except Exception as e:
        yield f"An unexpected error occurred: {str(e)}"

# --- Gradio UI and Backend Logic ---

# JavaScript to handle file uploads and update the textbox
# This is a key part of the new UI interaction.
js_script = """
function(files_state, text_input) {
    // files_state is a JSON string of a list of file paths. We parse it.
    let current_files = files_state ? JSON.parse(files_state) : [];
    
    // Create a new array with the paths of the newly uploaded files
    const new_files = Array.from(arguments[0]).map(f => f.name);

    // Add new file paths to our state
    current_files.push(...new_files);
    
    // Create the @-mentions for the new files
    const mentions = new_files.map(f => `@'${f}'`).join(' ');
    
    // Update the textbox
    const updated_text = text_input ? text_input + ' ' + mentions : mentions;

    // Return the updated file state (as a JSON string) and the new textbox content.
    // The 'null' is for the gr.UploadButton, to clear it after upload.
    return [JSON.stringify(current_files), updated_text, null];
}
"""

# Add custom CSS to ensure code blocks wrap text correctly.
css = """
pre, code {
    white-space: pre-wrap !important;
    word-wrap: break-word;
}
"""


def submit_message(text_input, files_state, history):
    """
    Handles message submission, processes files, and streams the response.
    """
    history = history or []
    uploaded_files = json.loads(files_state) if files_state else []

    if not text_input and not uploaded_files:
        yield history, None, ""
        return
        
    # Append user message to history
    user_message_content = text_input
    # Find all file paths associated with @-mentions in the text
    mentioned_files = re.findall(r"@'([^']+)'", text_input)
    
    # We will pass all uploaded files to the backend for now.
    # A more advanced version could filter based on mentions.
    file_paths_to_backend = uploaded_files # The state already holds the list of filenames (strings)

    # Display the user message in the chat
    history.append({"role": "user", "content": user_message_content})
    
    # Add a placeholder for the assistant's response
    history.append({"role": "assistant", "content": "..."})
    yield history, "[]", "" # Clear files_state and text_input

    # Stream the response from the backend
    response_generator = gemini_cli_interface(text_input, file_paths_to_backend, history[:-2]) # Exclude the current exchange
    
    full_response = ""
    for line in response_generator:
        full_response += line
        # Update the last message in history with the streaming content
        history[-1]["content"] = full_response
        
        # Yield the history directly. Gradio will render the markdown.
        yield history, "[]", ""


def create_gemini_chat_ui():
    with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
        # This state will hold a JSON string of the uploaded file paths
        uploaded_files_state = gr.State(value="[]")

        chatbot = gr.Chatbot(label="Chat History", height=600, type="messages")
        
        with gr.Row(elem_classes="input-row"):
            with gr.Column(scale=10):
                text_input = gr.Textbox(
                    label="Message",
                    placeholder="Type your message or upload files...",
                    lines=1,
                    max_lines=5,
                    show_label=False,
                )
            with gr.Column(scale=1, min_width=80):
                upload_btn = gr.UploadButton("📁", file_count="multiple")

        submit_btn = gr.Button("Send", variant="primary")

        # --- Event Handlers ---
        
        # Handles file uploads
        upload_btn.upload(
            fn=lambda files, state, text: (json.dumps([f.name for f in files] + (json.loads(state) if state else [])), text + " " + " ".join([f"@{Path(f.name).name}" for f in files]), None),
            inputs=[upload_btn, uploaded_files_state, text_input],
            outputs=[uploaded_files_state, text_input, upload_btn]
        )
        
        # Handles message submission (click or enter)
        submit_btn.click(
            fn=submit_message,
            inputs=[text_input, uploaded_files_state, chatbot],
            outputs=[chatbot, uploaded_files_state, text_input],
        )
        text_input.submit(
            fn=submit_message,
            inputs=[text_input, uploaded_files_state, chatbot],
            outputs=[chatbot, uploaded_files_state, text_input],
        )

    return uploaded_files_state, text_input