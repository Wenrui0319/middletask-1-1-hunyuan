import gradio as gr
from PIL import Image
from io import BytesIO
import os
import uuid
import traceback
import google.generativeai as genai
from google.generativeai import types

try:
    # Make sure to configure your API key securely
    genai.configure(api_key="AIzaSyCqVsEkGfQ3rqdote_eMkJHZNLdWKgaRjI")
    # The client is implicitly created and managed by the library after configuration.
    # We can directly use the functions from the genai module.
    client = genai
except Exception as e:
    print(f"Could not initialize Gemini client: {e}")
    client = None

# --- Mock Gemini API Interaction for UI development ---
def mock_gemini_image_edit(text_prompt, input_image):
    """
    Mock function to simulate the Gemini Image Edit API call.
    It currently just returns the input image.
    """
    print(f"Mocking Gemini Image Edit with prompt: '{text_prompt}'")
    if input_image:
        # Create a dummy response
        output_image = input_image.copy()
        # In a real scenario, this would be the image returned from the API
        return output_image, "Successfully generated mock image."
    return None, "Error: Input image is missing."

# --- Real Gemini API Interaction ---
def gemini_image_edit(text_prompt, input_image):
    """
    Calls the Gemini Image Edit API.
    """
    if not client:
        return None, "Gemini client is not initialized. Check your API key and configuration."

    try:
        # The user's prompt
        prompt = text_prompt

        # The image to edit
        image = input_image

        # This is where the actual API call would be made
        model = genai.GenerativeModel('gemini-2.5-flash-image-preview')
        response = model.generate_content([prompt, image])

        generated_image = None
        response_text = ""
        # The response object might contain the result in `candidates`
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    response_text += part.text
                elif part.inline_data and part.inline_data.data:
                    generated_image = Image.open(BytesIO(part.inline_data.data))
                    # Stop after finding the first image
                    break

        if generated_image:
            return generated_image, response_text or "Image generated successfully."
        else:
            return None, response_text or "Failed to generate image from response."

    except Exception as e:
        traceback.print_exc()
        return None, f"An unexpected error occurred: {str(e)}"

# --- Gradio UI and Backend Logic ---
def create_gemini_image_edit_ui():
    """
    Creates the Gradio UI for the Gemini Image Edit feature.
    """
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        generated_image_state = gr.State()

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil", image_mode="RGBA")
                prompt_input = gr.Textbox(label="Prompt", placeholder="Describe the edit you want to make...")
                submit_btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="Edited Image", type="pil", image_mode="RGBA")
                status_text = gr.Textbox(label="Status", interactive=False)
                with gr.Row():
                    file_name_input = gr.Textbox(label="Filename", placeholder="Enter filename (e.g., my_image.png)", scale=3)
                    save_btn = gr.Button("Save to Workspace", scale=1)
                save_status_text = gr.Textbox(label="Save Status", interactive=False)

        def on_submit(prompt, image):
            if not prompt or image is None:
                return None, "Please provide both an image and a prompt.", None
            
            yield None, "Generating, please wait...", None
            
            edited_image, message = gemini_image_edit(prompt, image)
            
            yield edited_image, message, edited_image

        def save_to_workspace(image_to_save, filename):
            if image_to_save is None:
                return "No image to save."

            subfolder = "gemini_edited_images"
            save_dir = os.path.join("data", subfolder)
            os.makedirs(save_dir, exist_ok=True)

            if not filename:
                timestamp = int(time.time() * 1000)
                filename = f"gemini_edit_{timestamp}.png"
            
            if not any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                filename += '.png'

            if ".." in filename or "/" in filename:
                return "Invalid filename. Do not use '..' or '/'."

            try:
                save_path = os.path.join(save_dir, filename)
                image_to_save.save(save_path)
                return f"Image saved successfully to {save_path}"
            except Exception as e:
                return f"Error saving image: {str(e)}"

        submit_btn.click(
            fn=on_submit,
            inputs=[prompt_input, input_image],
            outputs=[output_image, status_text, generated_image_state]
        )

        save_btn.click(
            fn=save_to_workspace,
            inputs=[generated_image_state, file_name_input],
            outputs=[save_status_text]
        )

    return input_image