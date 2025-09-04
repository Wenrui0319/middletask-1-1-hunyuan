import gradio as gr
import json
from workflow_manager import WorkflowManager
import os
import time

# 1. 初始化工作流管理器
manager = WorkflowManager()

# 2. 读取HTML模板
# 确保index.html和app.py在同一目录下，或者提供正确路径
with open("index.html", "r", encoding="utf-8") as f:
    html_template = f.read()

# --- 后端逻辑函数 ---

def create_root(image_temp_path):
    """处理上传，创建根节点. 只返回新的树数据."""
    if not image_temp_path:
        return None
    tree_data = manager.create_root_node(image_temp_path.name)
    return tree_data

def update_preview(selected_node_id):
    """根据选中的节点ID更新预览图像"""
    if not selected_node_id:
        return None
    
    node = manager.find_node(selected_node_id)
    if node and "file" in node:
        return node["file"]["path"]
    return None

def save_new_node(parent_node_id, current_tab_name, current_tree):
    """模拟保存一个操作结果，并创建新节点"""
    # 同步管理器的状态
    manager.tree = current_tree

    if not parent_node_id:
        gr.Warning("No parent node selected!")
        return current_tree

    parent_node = manager.find_node(parent_node_id)
    if not parent_node:
        gr.Warning("Parent node not found!")
        return current_tree
    
    from PIL import Image
    img = Image.open(parent_node["file"]["path"])
    processed_img = img.convert("L")
    
    temp_output_path = os.path.join(manager.workspace_dir, f"temp_{time.time()}.png")
    processed_img.save(temp_output_path)
    
    operation_details = {
        "name": current_tab_name,
        "params": {"detail": "Converted to grayscale"}
    }
    
    tree_data = manager.add_child_node(parent_node_id, temp_output_path, operation_details)
    return tree_data

def delete_node(node_id_to_delete, current_tree):
    """删除节点及其子树"""
    # 同步管理器的状态
    manager.tree = current_tree

    if not node_id_to_delete:
        gr.Warning("No node selected for deletion!")
        return current_tree

    tree_data = manager.delete_subtree(node_id_to_delete)
    return tree_data


# --- Gradio UI 构建 ---

with gr.Blocks(theme=gr.themes.Soft(), css="static/css/style.css") as demo:
    gr.Markdown("# Gradio 工作流可视化 DEMO")

    # 存储状态和执行JS的隐藏组件
    with gr.Row(visible=False):
        workflow_json_state = gr.JSON(value=None)
        selected_node_id_state = gr.Textbox(elem_id="selected-node-id-input")
        preview_trigger_button = gr.Button(elem_id="preview-button")
        # 新增一个专门用来执行JS的HTML组件
        js_runner = gr.HTML()

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 工作流历史")
            # 主HTML视图只加载一次，不再动态更新
            workflow_html = gr.HTML(html_template)
            gr.Markdown("## 节点预览")
            preview_image = gr.Image(label="选中节点图像预览", interactive=False)

        # 右侧：功能区
        with gr.Column(scale=2):
            with gr.Tabs() as tabs:
                with gr.TabItem("功能 A: 灰度化", id="tab_grayscale"):
                    gr.Markdown("这是功能A的界面。点击下面的按钮将对选中节点图像进行灰度化并保存为新节点。")
                    # 此处的输入图像可以由“编辑”按钮填充
                    input_image_a = gr.Image(label="输入图像") 
                    save_button_a = gr.Button("保存至工作区")

                with gr.TabItem("功能 B: Inpainting", id="tab_inpainting"):
                    gr.Markdown("这是功能B的界面。")
                    input_image_b = gr.Image(label="输入图像")
                    save_button_b = gr.Button("保存至工作区")
    
    with gr.Row():
        upload_button = gr.UploadButton("上传初始图像", file_types=["image"])
        # "编辑"按钮需要知道当前在哪个Tab，Gradio暂不支持直接获取，但我们可以为每个Tab设置一个保存按钮
        # edit_button = gr.Button("编辑选中节点")
        delete_button = gr.Button("删除选中节点")

    # --- 事件绑定 ---

    def create_js_call(tree_data):
        """根据树数据生成JS调用脚本"""
        if tree_data is None:
            return "", tree_data
        js_code = f"<script>renderWorkflowTree({json.dumps(tree_data)});</script>"
        return js_code, tree_data

    # 1. 上传图像 -> 更新JSON状态，并返回JS脚本
    @upload_button.upload(inputs=[upload_button], outputs=[js_runner, workflow_json_state])
    def handle_upload(image_temp_path):
        tree_data = create_root(image_temp_path)
        return create_js_call(tree_data)

    # 2. 预览功能不变
    preview_trigger_button.click(
        fn=update_preview,
        inputs=[selected_node_id_state],
        outputs=[preview_image]
    )
    
    # 3. 保存新节点 -> 更新JSON状态，并返回JS脚本
    @save_button_a.click(inputs=[selected_node_id_state, workflow_json_state], outputs=[js_runner, workflow_json_state])
    def handle_save_a(node_id, tree):
        tree_data = save_new_node(node_id, "Grayscale", tree)
        return create_js_call(tree_data)

    @save_button_b.click(inputs=[selected_node_id_state, workflow_json_state], outputs=[js_runner, workflow_json_state])
    def handle_save_b(node_id, tree):
        tree_data = save_new_node(node_id, "Inpainting", tree)
        return create_js_call(tree_data)

    # 4. 删除节点 -> 更新JSON状态，并返回JS脚本
    @delete_button.click(inputs=[selected_node_id_state, workflow_json_state], outputs=[js_runner, workflow_json_state])
    def handle_delete(node_id, tree):
        tree_data = delete_node(node_id, tree)
        return create_js_call(tree_data)

if __name__ == "__main__":
    # Gradio需要能访问到static目录，需按如下方式启动
    demo.launch(allowed_paths=["static", "workspace"])
