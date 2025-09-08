# app.py

import os
import torch
import shutil
import gradio as gr
import uvicorn
import time
from fastapi import FastAPI, HTTPException, UploadFile, File
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse  # Import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json

import hunyuan_logic
import sam_logic
from sam_logic import SAM_AVAILABLE
import file_operations
import qwen_edit_logic
import qwen_inpainting_logic
import gemini_gradio_app

try:
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradio_tmp")
    os.makedirs(temp_dir, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = temp_dir
    print(f"Gradio temporary directory set to: {temp_dir}")
except Exception as e:
    print(f"Warning: Could not create or set local Gradio temp directory: {e}")

os.makedirs("data/sam", exist_ok=True)


def get_file_structure(directory):
    """
    Recursively builds a file structure for a given directory.
    """
    tree = []
    for item in sorted(Path(directory).iterdir()):
        node = {"name": item.name, "path": str(item)}
        if item.is_dir():
            node["type"] = "folder"
            node["children"] = get_file_structure(item)
        else:
            node["type"] = "file"
        tree.append(node)
    return tree


def build_app(args):
    title = "Hunyuan3D-2: High Resolution Textured 3D Assets Generation"
    if hunyuan_logic.MV_MODE:
        title = "Hunyuan3D-2mv: Image to 3D Generation with 1-4 Views"
    if "mini" in hunyuan_logic.args.subfolder:
        title = "Hunyuan3D-2mini: Strong 0.6B Image to Shape Generator"
    if hunyuan_logic.TURBO_MODE:
        title = title.replace(":", "-Turbo: Fast ")

    with gr.Blocks(
        theme=gr.themes.Base(),
        title="Hunyuan-3D-2.0",
        analytics_enabled=False,
        css=".gradio-container { max-width: unset !important; padding-left: 20px; padding-right: 20px; }",
    ) as demo:
        # Central state for the selected file path
        selected_file_state = gr.State(None)
        # Hidden textbox to receive file path from HTML
        file_path_input = gr.Textbox(
            label="File Path", visible=True, elem_id="file_path_input"
        )
        # State to determine the action (preview or edit)
        file_action_state = gr.State("preview")

        # 主界面标题
        # 主界面标题
        gr.HTML("""
        <div style="text-align: center;">
            <h1 style="font-size: 2em; margin: 0; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                Selective 3D Generation Engine
            </h1>
            <h2 style="font-size: 1em; margin: 10px 0 0 0; font-weight: 300; opacity: 0.9;">
                Middle Task 1 - Team 1 
            </h2>
        </div>
        """)
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("File Explorer"):
                        image = gr.Image(
                            label="Image", type="pil", image_mode="RGBA", height=290
                        )
                        caption = gr.Textbox(visible=False)
                        mv_image_front = gr.Image(visible=False)
                        mv_image_back = gr.Image(visible=False)
                        mv_image_left = gr.Image(visible=False)
                        mv_image_right = gr.Image(visible=False)
                        with gr.Group():
                            file_out = gr.File(label="File", visible=False)
                            file_out2 = gr.File(label="File", visible=False)
                        gr.HTML(
                            f"<iframe src='/static/file_explorer.html' width='100%' height='600px' frameborder='0'></iframe>"
                        )

            with gr.Column(scale=9):
                active_tab_state = gr.State("SAM")
                with gr.Tabs() as tabs_output:
                    with gr.Tab("SAM", visible=SAM_AVAILABLE, id="SAM"):
                        sam_input_image = sam_logic.create_sam_ui(
                            sam_predictor_global=sam_logic.sam_predictor_global,
                            device=args.sam_device,
                        )
                    with gr.Tab("Qwen Edit", id="Qwen Edit"):
                        qwen_edit_input_image = qwen_edit_logic.create_qwen_edit_ui()
                    with gr.Tab("Qwen Inpainting", id="Qwen Inpainting"):
                        qwen_inpainting_input_image = (
                            qwen_inpainting_logic.create_qwen_inpainting_ui()
                        )
                    with gr.Tab("Gemini Chat", id="Gemini Chat"):
                        gemini_uploaded_files_state, gemini_text_input = (
                            gemini_gradio_app.create_gemini_chat_ui()
                        )
                    with gr.Tab("Hunyuan3D", id="Hunyuan3D"):
                        hunyuan_input_image = hunyuan_logic.create_hunyuan_ui(
                            hunyuan_logic.SUPPORTED_FORMATS,
                            hunyuan_logic.HTML_OUTPUT_PLACEHOLDER,
                            tabs_output,
                            caption,
                            mv_image_front,
                            mv_image_back,
                            mv_image_left,
                            mv_image_right,
                            file_out,
                            file_out2,
                        )

        def on_tab_select(evt: gr.SelectData):
            return evt.value

        tabs_output.select(fn=on_tab_select, inputs=None, outputs=[active_tab_state])

        def update_selected_file(file_path, active_tab):
            """
            Update the selected_file_state, preview the image, and dispatch it
            to the active tab.
            """
            print(
                f"update_selected_file called with file_path: {file_path}, active_tab: {active_tab}"
            )
            if not file_path:
                # Return updates to clear all fields if file_path is empty
                return (
                    None,
                    None,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

            # Ensure the path has the correct 'data/' prefix for backend operations
            if not file_path.startswith("data/"):
                prefixed_path = os.path.join("data", file_path)
            else:
                prefixed_path = file_path

            preview_update = file_operations.preview_image([prefixed_path])

            dispatched_updates = file_operations.dispatch_image(
                [prefixed_path], active_tab
            )

            (
                sam_update,
                qwen_edit_update,
                qwen_inp_update,
                gemini_state_update,
                gemini_text_update,
                hunyuan_update,
            ) = dispatched_updates

            return (
                prefixed_path,
                preview_update,
                sam_update,
                qwen_edit_update,
                qwen_inp_update,
                gemini_state_update,
                gemini_text_update,
                hunyuan_update,
            )

        file_path_input.change(
            fn=update_selected_file,
            inputs=[file_path_input, active_tab_state],
            outputs=[
                selected_file_state,
                image,
                sam_input_image,
                qwen_edit_input_image,
                qwen_inpainting_input_image,
                gemini_uploaded_files_state,
                gemini_text_input,
                hunyuan_input_image,
            ],
        )

        demo.load(fn=sam_logic.check_sam_model_on_load)

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2")
    parser.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-0")
    parser.add_argument("--texgen_model_path", type=str, default="tencent/Hunyuan3D-2")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for main models like Hunyuan3D.",
    )
    parser.add_argument(
        "--sam_device",
        type=str,
        default=None,
        help="Device for SAM model. Defaults to --device if not set.",
    )
    parser.add_argument("--mc_algo", type=str, default="mc")
    parser.add_argument("--cache-path", type=str, default="gradio_cache")
    parser.add_argument("--enable_t23d", action="store_true")
    parser.add_argument("--disable_tex", action="store_true")
    parser.add_argument("--enable_flashvdm", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--low_vram_mode", action="store_true")
    args = parser.parse_args()

    if args.sam_device is None:
        args.sam_device = args.device

    project_root = os.path.dirname(os.path.abspath(__file__))

    hunyuan_logic.initialize_hunyuan(args, project_root)

    from argparse import Namespace

    sam_args = Namespace(device=args.sam_device)
    sam_logic.initialize_sam(sam_args)

    hunyuan_logic.args = args
    hunyuan_logic.SAVE_DIR = temp_dir

    app = FastAPI()

    @app.get("/api/files")
    async def read_files():
        return get_file_structure("data")

    @app.get("/api/download")
    async def download_file(path: str):
        # Security check: Ensure the path is within the allowed directory.
        safe_base_dir = os.path.abspath("data")
        requested_path = os.path.abspath(os.path.join(safe_base_dir, path.strip("/\\")))

        if not os.path.exists(requested_path):
            raise HTTPException(status_code=404, detail="File not found.")

        # Use the existing logic to zip files/directories.
        # The function expects a list of paths.
        zip_path = file_operations.download_selected_files([requested_path])

        # Return the zip file as a response.
        return FileResponse(
            zip_path, media_type="application/zip", filename=os.path.basename(zip_path)
        )

    @app.delete("/api/delete")
    async def delete_file(path: str):
        # Security check: Ensure the path is within the allowed directory.
        safe_base_dir = os.path.abspath("data")
        # Just use the path directly, it will be joined to safe_base_dir
        requested_path = os.path.abspath(os.path.join(safe_base_dir, path.strip("/\\")))

        if not os.path.exists(requested_path):
            raise HTTPException(status_code=404, detail="File not found.")

        try:
            # Reuse the logic from file_operations
            file_operations.delete_selected_files([requested_path])
            return {"message": "File deleted successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/upload")
    async def upload_files(files: list[UploadFile] = File(...)):
        save_path_root = "data/"
        os.makedirs(save_path_root, exist_ok=True)

        for file in files:
            destination_path = os.path.join(save_path_root, file.filename)
            try:
                with open(destination_path, "wb+") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not upload file: {file.filename} - {str(e)}",
                )
            finally:
                file.file.close()

        return {"message": f"{len(files)} files uploaded successfully."}

    static_dir = Path(hunyuan_logic.SAVE_DIR).absolute()
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/static", StaticFiles(directory=static_dir, html=False), name="static"
    )  # html=False is safer

    env_maps_src = os.path.join(project_root, "assets", "env_maps")
    env_maps_dest = os.path.join(static_dir, "env_maps")
    if os.path.exists(env_maps_src):
        shutil.copytree(env_maps_src, env_maps_dest, dirs_exist_ok=True)

    file_explorer_src = os.path.join(project_root, "file_explorer.html")
    file_explorer_dest = os.path.join(static_dir, "file_explorer.html")
    if os.path.exists(file_explorer_src):
        shutil.copy(file_explorer_src, file_explorer_dest)

    demo = build_app(args)
    app = gr.mount_gradio_app(app, demo, path="/")

    if args.low_vram_mode:
        torch.cuda.empty_cache()

    print(f"\n>>> Gradio 服务即将启动！请在浏览器中打开 http://{args.host}:{args.port}")
    print(f"    - 主模型设备: {args.device}")
    print(f"    - SAM 模型设备: {args.sam_device}")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
