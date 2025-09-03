# file_operations.py
"""
File operations for the Hunyuan3D application.
Handles file explorer operations including delete, upload, and download functionality.
"""

import os
import shutil
import zipfile
import tempfile
import time
from PIL import Image
import gradio as gr


def handle_file_selection(file_paths):
    """Handle file selection changes from FileExplorer"""
    return file_paths or []


def delete_selected_files(file_paths):
    """delete selected files from the data folder"""

    if file_paths:
        for file_path in file_paths:
            try:
                # security check: ensure the file is within the data directory
                full_path = os.path.abspath(file_path)
                data_dir = os.path.abspath("data/")
                if not full_path.startswith(data_dir):
                    continue

                if os.path.exists(full_path):
                    if os.path.isfile(full_path):
                        os.remove(full_path)
                    elif os.path.isdir(full_path):
                        shutil.rmtree(full_path)
            except Exception as e:
                print(f"error deleting file: {e}")
    
    # Return a new FileExplorer with a dynamic key to force a full re-render
    return gr.FileExplorer(root_dir="data/", file_count="multiple", label="File Explorer", show_label=True, height=600, value=[], key=str(time.time()))

def upload_selected_image(file_paths):
    """Upload selected image using existing image upload logic"""
    if not file_paths:
        raise gr.Error("No file selected for upload")

    if len(file_paths) > 1:
        raise gr.Error("Please select only one image for upload")

    file_path = file_paths[0]

    # Security check: ensure the file is within the data directory
    full_path = os.path.abspath(file_path)
    data_dir = os.path.abspath("data/")
    if not full_path.startswith(data_dir):
        raise gr.Error(f"Unauthorized file access: {file_path}")

    # Check if it's an image file
    supported_formats = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]
    if not any(full_path.lower().endswith(ext) for ext in supported_formats):
        raise gr.Error("Selected file is not a supported image format")

    try:
        # Load the image using PIL
        image = Image.open(full_path)
        return image
    except Exception as e:
        raise gr.Error(f"Error loading image: {str(e)}")


def download_selected_files(file_paths):
    """Create a zip file of selected files for download"""
    if not file_paths:
        raise gr.Error("No files selected for download")

    try:
        # Create a temporary zip file
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "selected_files.zip")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in file_paths:
                try:
                    # Security check
                    full_path = os.path.abspath(file_path)
                    data_dir = os.path.abspath("data/")
                    if not full_path.startswith(data_dir):
                        continue

                    if os.path.exists(full_path):
                        if os.path.isfile(full_path):
                            # Add file to zip
                            arcname = os.path.relpath(full_path, data_dir)
                            zipf.write(full_path, arcname)
                        elif os.path.isdir(full_path):
                            # Add directory and all its contents
                            for root, dirs, files in os.walk(full_path):
                                for file in files:
                                    file_full_path = os.path.join(root, file)
                                    arcname = os.path.relpath(file_full_path, data_dir)
                                    zipf.write(file_full_path, arcname)
                except Exception as e:
                    # Continue with other files if one fails
                    pass

        return zip_path
    except Exception as e:
        raise gr.Error(f"Error creating download archive: {str(e)}")


def upload_from_local(files):
    """Handle file uploads from the user's local machine to the data directory."""
    if files:
        # The 'data' directory is the root for the file explorer
        save_path_root = "data/"
        
        for temp_file in files:
            # The object from UploadButton can be a NamedString, where `name` holds the
            # original filename and the string itself is the temporary path.
            original_filename = os.path.basename(temp_file.name)

            # Construct the destination path
            destination_path = os.path.join(save_path_root, original_filename)

            # The temp_file object itself is the path to the temporary path.
            shutil.move(temp_file, destination_path)
            
    # Return a new FileExplorer with a dynamic key to force a full re-render
    return gr.FileExplorer(root_dir="data/", file_count="multiple", label="File Explorer", show_label=True, height=600, value=[], key=str(time.time()))


# Export the functions for use in the main app
def dispatch_image(files, active_tab_name):
    """
    Loads a single image and returns it, along with update signals for other components.
    The position of the loaded image in the returned tuple depends on the active_tab_name.
    """
    if not files:
        raise gr.Error("No file selected.")

    if len(files) > 1:
        raise gr.Error("Please select only one image to edit.")

    file_path = files[0]

    # Security check
    full_path = os.path.abspath(file_path)
    data_dir = os.path.abspath("data/")
    if not full_path.startswith(data_dir):
        raise gr.Error(f"Unauthorized file access: {file_path}")

    # Check if it's an image file
    supported_formats = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]
    if not any(full_path.lower().endswith(ext) for ext in supported_formats):
        raise gr.Error("Selected file is not a supported image format.")

    try:
        loaded_image = Image.open(full_path)
    except Exception as e:
        raise gr.Error(f"Error loading image: {str(e)}")

    # Define the order of tabs/outputs as they appear in app.py
    # This order MUST be kept in sync with the `outputs` list of the click event.
    # The order of tabs/outputs as they appear in app.py
    # This order MUST be kept in sync with the `outputs` list of the click event.
    tab_order = ["SAM", "Qwen Edit", "Qwen Inpainting", "Gemini Chat", "Hunyuan3D"]
    
    # Initialize a list of "no update" signals for all 6 output components
    outputs = [gr.update()] * 6

    try:
        # Find the index of the active tab
        if active_tab_name in tab_order:
            target_index = tab_order.index(active_tab_name)

            if active_tab_name == "SAM":
                outputs[target_index] = full_path

            elif active_tab_name == "Gemini Chat":
                import json
                from pathlib import Path
                # For Gemini Chat, update the state and textbox, which are at index 3 and 4
                outputs[3] = gr.update(value=json.dumps([full_path]))
                outputs[4] = gr.update(value=f"@{Path(full_path).name}")
            elif active_tab_name == "Qwen Inpainting":
                from qwen_inpainting_logic import upload_masked_image
                # Create a temporary file object for upload_masked_image
                class TempFile:
                    def __init__(self, name):
                        self.name = name
                
                temp_file_obj = TempFile(full_path)
                inpainting_data = upload_masked_image(temp_file_obj)
                outputs[target_index] = inpainting_data
            elif active_tab_name == "Hunyuan3D":
                 # Hunyuan3D is the last tab, so its output is at index 5
                 outputs[5] = loaded_image
            else:
                # For other tabs, just update the corresponding image input
                outputs[target_index] = loaded_image
    except ValueError:
        # This block might not be strictly necessary anymore but is good for safety
        gr.Warning(f"Unknown editor tab: {active_tab_name}. Cannot dispatch image.")

    return tuple(outputs)


def preview_image(file_paths):
    """Preview the selected image file."""
    if not file_paths or len(file_paths) > 1:
        return None

    file_path = file_paths[0]
    supported_formats = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]

    if any(file_path.lower().endswith(ext) for ext in supported_formats):
        try:
            # Security check
            full_path = os.path.abspath(file_path)
            data_dir = os.path.abspath("data/")
            if not full_path.startswith(data_dir):
                return None
            
            return gr.Image(value=full_path, visible=True)
        except Exception:
            return None
    return None


__all__ = [
    "handle_file_selection",
    "delete_selected_files",
    "upload_selected_image",
    "download_selected_files",
    "upload_from_local",
    "dispatch_image",
    "preview_image",
]
