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
    """处理上传，创建根节点"""
    if not image_temp_path:
        return None, None
    
    tree_data = manager.create_root_node(image_temp_path.name)
    
    # 返回要在 gr.HTML 中执行的JS代码
    js_call = f"<script>renderWorkflowTree({json.dumps(tree_data)});</script>"
    return html_template + js_call, tree_data

def update_preview(selected_node_id_json):
    """根据选中的节点ID更新预览图像"""
    if not selected_node_id_json:
        return None
    
    # The input from gr.Textbox is a JSON string, e.g., '"node-id-string"'
    # We need to parse it correctly. Sometimes it might just be the string.
    try:
        selected_node_id = json.loads(selected_node_id_json)
    except (json.JSONDecodeError, TypeError):
        selected_node_id = selected_node_id_json

    if not selected_node_id:
        return None
        
    node = manager.find_node(selected_node_id)
    if node and "file" in node:
        return node["file"]["path"]
    return None

def save_new_node(parent_node_id_json, current_tab_name):
    """模拟保存一个操作结果，并创建新节点"""
    try:
        parent_node_id = json.loads(parent_node_id_json)
    except (json.JSONDecodeError, TypeError):
        parent_node_id = parent_node_id_json

    if not parent_node_id:
        gr.Warning("No parent node selected!")
        # Return current state without changes
        return html_template + f"<script>renderWorkflowTree({json.dumps(manager.get_tree())});</script>", manager.get_tree()

    # --- 模拟一个编辑操作 ---
    # 1. 找到父节点的图像
    parent_node = manager.find_node(parent_node_id)
    if not parent_node:
        gr.Warning("Parent node not found!")
        return html_template + f"<script>renderWorkflowTree({json.dumps(manager.get_tree())});</script>", manager.get_tree()
    
    # 2. 假设我们对图像做了一个操作，并生成了一个新图像
    # 在真实应用中，这里会调用你的图像处理函数
    # 这里我们只是复制父节点的图像来模拟
    from PIL import Image
    
    img = Image.open(parent_node["file"]["path"])
    # 模拟处理：比如转为灰度
    processed_img = img.convert("L")
    
    # 3. 保存新图像到临时位置
    temp_output_path = os.path.join(manager.workspace_dir, f"temp_{time.time()}.png")
    processed_img.save(temp_output_path)
    
    # 4. 定义操作属性
    operation_details = {
        "name": current_tab_name, # e.g., "Inpainting", "Grayscale"
        "params": {"detail": "Converted to grayscale"}
    }
    
    # 5. 在工作流中添加新节点
    tree_data = manager.add_child_node(parent_node_id, temp_output_path, operation_details)
    
    js_call = f"<script>renderWorkflowTree({json.dumps(tree_data)});</script>"
    return html_template + js_call, tree_data


def delete_node(node_id_to_delete_json):
    """删除节点及其子树"""
    try:
        node_id_to_delete = json.loads(node_id_to_delete_json)
    except (json.JSONDecodeError, TypeError):
        node_id_to_delete = node_id_to_delete_json

    if not node_id_to_delete:
        gr.Warning("No node selected for deletion!")
        return html_template, manager.get_tree()

    tree_data = manager.delete_subtree(node_id_to_delete)
    
    js_call = f"<script>renderWorkflowTree({json.dumps(tree_data)});</script>"
    return html_template + js_call, tree_data

# --- Gradio UI 构建 ---

with gr.Blocks(theme=gr.themes.Soft(), css="static/css/style.css") as demo:
    gr.Markdown("# Gradio 工作流可视化 DEMO")
    
    # 存储状态的隐藏组件
    with gr.Row(visible=False):
        # 用于在前后端传递整个树的JSON数据
        workflow_json_state = gr.JSON(value=None)
        # 用于从JS接收当前选中的节点ID
        selected_node_id_state = gr.Textbox(elem_id="selected-node-id-input")
        # 用于从JS触发预览更新的隐藏按钮
        preview_trigger_button = gr.Button(elem_id="preview-button")
    
    with gr.Row():
        # 左侧：工作流和预览
        with gr.Column(scale=1):
            gr.Markdown("## 工作流历史")
            # gr.HTML 用于渲染D3树状图
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
    
    # 1. 上传图像 -> 创建根节点，更新HTML和JSON状态
    upload_button.upload(
        fn=create_root,
        inputs=[upload_button],
        outputs=[workflow_html, workflow_json_state]
    )

    # 2. JS中点击节点 -> 更新selected_node_id_state -> 触发preview_trigger_button
    #    -> 更新预览图
    #    (这是一个链式反应: JS -> Textbox -> Button.click -> Python fn)
    preview_trigger_button.click(
        fn=update_preview,
        inputs=[selected_node_id_state],
        outputs=[preview_image]
    )
    
    # 3. 点击"保存至工作区"按钮 -> 创建新节点
    def create_save_handler(tab_name):
        return lambda node_id: save_new_node(node_id, tab_name)

    save_button_a.click(
        fn=create_save_handler("Grayscale"),
        inputs=[selected_node_id_state],
        outputs=[workflow_html, workflow_json_state]
    )
    save_button_b.click(
        fn=create_save_handler("Inpainting"),
        inputs=[selected_node_id_state],
        outputs=[workflow_html, workflow_json_state]
    )

    # 4. 点击"删除"按钮 -> 删除节点
    delete_button.click(
        fn=delete_node,
        inputs=[selected_node_id_state],
        outputs=[workflow_html, workflow_json_state]
    )

if __name__ == "__main__":
    # Gradio需要能访问到static目录，需按如下方式启动
    demo.launch(allowed_paths=["static", "workspace"])
