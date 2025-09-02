# file_operations.py
"""
File operations for the Hunyuan3D application.
Handles file explorer operations including delete, upload, and download functionality.
"""

import os
import shutil
import zipfile
import tempfile
from PIL import Image
import gradio as gr


def handle_file_selection(file_paths):
    """Handle file selection changes from FileExplorer"""
    return file_paths or []


def delete_selected_files(file_paths):
    """delete selected files from the data folder"""

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
        except exception as e:
            print(f"error deleting file: {e}")
    return []

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


# Export the functions for use in the main app
__all__ = [
    "handle_file_selection",
    "delete_selected_files",
    "upload_selected_image",
    "download_selected_files",
]
