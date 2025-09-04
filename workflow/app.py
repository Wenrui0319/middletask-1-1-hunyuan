import gradio as gr
import os
from workflow_manager import WorkflowManager

# 初始化工作流管理器
wm = WorkflowManager(filepath="workflow.json", workspace_dir="workspace")

# --- JavaScript and CSS ---
# --- JavaScript and CSS ---

# 1. 导入 Mermaid.js 库
# 2. 定义 handleNodeClick 用于节点点击时与 Gradio 后端通信
# 3. 定义 renderMermaid 函数，用于接收 Mermaid 语法并渲染 SVG
# 4. 使用 MutationObserver 监听 Mermaid HTML 组件的变化，一旦内容更新，自动重新渲染
js_script = """
// 使用动态导入，确保 Mermaid 库在使用前加载
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';

// 将 handleNodeClick 暴露到 window 对象，以便 Mermaid 的 click 指令可以调用
window.handleNodeClick = function(nodeId) {
    const hidden_textbox_elem = document.querySelector('#selected_node_id_input textarea');
    if (hidden_textbox_elem) {
        hidden_textbox_elem.value = nodeId;
        const inputEvent = new Event('input', { bubbles: true });
        const changeEvent = new Event('change', { bubbles: true });
        hidden_textbox_elem.dispatchEvent(inputEvent);
        hidden_textbox_elem.dispatchEvent(changeEvent);
    } else {
        console.error("Could not find the hidden textbox for node selection.");
    }
}

// 渲染 Mermaid 图表的函数
async function renderMermaid(mermaidContainer) {
    // 初始时 mermaidContainer 是 Gradio 组件的根 div, 我们要找的是里面的 span
    const targetSpan = mermaidContainer.querySelector('span');
    const container = targetSpan || mermaidContainer;
    
    const mermaidSyntax = container.textContent || '';
    
    // 如果是初始提示信息或已渲染的SVG，则不重复操作
    if (mermaidSyntax.trim().startsWith('<p') || mermaidContainer.querySelector('svg')) {
        container.style.textAlign = 'center';
        return;
    }
    
    container.style.textAlign = 'left';

    try {
        // 为渲染创建一个唯一的 ID
        const svgId = 'mermaid-svg-' + Date.now();
        const { svg } = await mermaid.render(svgId, mermaidSyntax);
        container.innerHTML = svg;
    } catch (e) {
        console.error("Mermaid rendering error:", e);
        container.innerHTML = `<p style='color:red;'>Error rendering workflow: ${e.message}</p>`;
    }
}

// Gradio 加载完成后执行的函数
function onGradioAppLoaded() {
    // 初始化 Mermaid
    mermaid.initialize({ startOnLoad: false, theme: 'base', 'fontFamily': 'monospace' });

    const targetNode = document.getElementById('workflow_tree_container');
    if (!targetNode) {
        console.error("Mermaid container not found.");
        return;
    }

    // 首次加载时渲染一次
    renderMermaid(targetNode);

    // 监听 targetNode 的子节点变化
    const observer = new MutationObserver((mutationsList, observer) => {
        for(const mutation of mutationsList) {
            // 当Gradio更新HTML组件时，通常是替换子节点
            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                 // 内容发生变化，重新渲染
                renderMermaid(targetNode);
                break;
            }
        }
    });

    // 配置观察器
    observer.observe(targetNode, { childList: true, subtree: true });
}

// 等待 Gradio 应用完全加载
// 使用 DOMContentLoaded 作为备用，但主要依赖于 gradio_config 的出现
function setupObserver() {
    // Gradio 通常会把 Blocks app 放在一个 <gradio-app> 标签里
    // 我们轮询检查 gradio_config 是否加载完毕
    const interval = setInterval(() => {
        const gradioApp = document.querySelector('gradio-app');
        if (gradioApp && window.gradio_config) {
            clearInterval(interval);
            onGradioAppLoaded();
        }
    }, 100);
}

document.addEventListener('DOMContentLoaded', setupObserver);

"""

# Mermaid 容器的样式
css_style = """
#workflow_tree_container {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px;
    background-color: #f9f9f9;
    min-height: 300px; /* 确保在加载时有高度 */
    display: flex;
    justify-content: center;
    align-items: center;
}
.mermaid {
    width: 100%;
}
.mermaid svg {
    display: block;
    margin: auto;
}
"""

# --- Gradio UI 定义 ---
with gr.Blocks(js=js_script, css=css_style) as demo:
    gr.Markdown("## 图像编辑工作流")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 节点预览与操作")
            preview_image = gr.Image(label="节点预览", interactive=False, height=300)
            edit_button = gr.Button("载入此图进行编辑")
            delete_button = gr.Button("删除选中节点及后续")
            
            # 隐藏的组件，用于JS和Python通信
            selected_node_id = gr.Textbox(
                label="Selected Node ID",
                visible=False,
                elem_id="selected_node_id_input"
            )

        with gr.Column(scale=2):
            gr.Markdown("#### 工作流历史")
            # 使用一个稳定的 elem_id 以便 JS 观察
            workflow_tree_html = gr.HTML(
                wm.to_mermaid(), # 初始加载时直接获取Mermaid字符串
                elem_id="workflow_tree_container"
            )
            
            upload_button = gr.UploadButton(
                "上传新图像 (创建新工作流)",
                file_types=["image"]
            )

    # --- 后端逻辑绑定 (将在后续步骤中实现) ---
    
    # --- Backend Logic Binding ---
    
    # 应用加载时，渲染初始的工作流树
    # demo.load() is no longer needed as the initial state is set directly in gr.HTML
    # and the JS MutationObserver will handle the rendering.

    # 绑定上传按钮的事件
    def on_upload(file_obj):
        if file_obj is None:
            return wm.to_mermaid()
        
        # Gradio 的 UploadButton 返回一个包含临时文件信息的对象
        # .name 是临时文件的路径
        # Gradio v4+ file object can be a list.
        # This code is defensive to handle both single object and list.
        if isinstance(file_obj, list):
            if not file_obj: return wm.to_mermaid() # Empty list
            file_obj = file_obj[0]
        
        # In some OS, file.type might be None.
        file_type = getattr(file_obj, 'type', 'application/octet-stream')
        wm.add_root_node(file_obj.name, file_type if file_type else 'unknown')
        wm.save()
        return wm.to_mermaid()

    upload_button.upload(fn=on_upload, inputs=upload_button, outputs=workflow_tree_html)

    # 绑定节点选择事件
    def on_select_node(node_id):
        if not node_id:
            return gr.update(value=None)
        
        node = wm.get_node(node_id)
        if node:
            image_path = node['data']['file_path']
            # 确保文件存在
            if os.path.exists(image_path):
                return gr.update(value=image_path)
        
        # 如果找不到节点或文件，则清空预览
        return gr.update(value=None)

    selected_node_id.change(
        fn=on_select_node,
        inputs=selected_node_id,
        outputs=preview_image
    )

    # 绑定“编辑”按钮的事件
    def on_edit_click(current_node_id):
        if not current_node_id:
            gr.Warning("请先在工作流中选择一个节点！")
            return

        node = wm.get_node(current_node_id)
        if not node:
            gr.Warning("找不到所选节点。")
            return

        # 检查节点类型，禁止编辑非叶子节点
        # A node can be edited if it's a leaf.
        # The logic was correct, just adding a comment for clarity.
        if node['type'] != 'leaf':
            gr.Warning(f"此节点为操作节点 ({node['label']})，其结果不可再次编辑。请选择一个叶子节点。")
            return
        
        # 模拟加载到编辑页
        gr.Info(f"已将节点 {node['label']} ({node['id'][:8]}...) 的图像加载到编辑页面。")
        print(f"EDITING: Loading image from {node['data']['file_path']} for node {node['id']}")

    edit_button.click(
        fn=on_edit_click,
        inputs=[selected_node_id],
        outputs=None # 此操作不直接更新任何组件
    )

    # 绑定“删除”按钮的事件
    def on_delete_click(current_node_id):
        if not current_node_id:
            gr.Warning("请先在工作流中选择一个要删除的节点！")
            return wm.to_mermaid(), None # 返回更新，但不清空预览

        wm.delete_subtree(current_node_id)
        wm.save()
        
        gr.Info(f"已删除节点 {current_node_id[:8]}... 及其所有后续节点。")
        
        # 刷新树，并清空预览和选中ID
        return wm.to_mermaid(), None, ""

    delete_button.click(
        fn=on_delete_click,
        inputs=[selected_node_id],
        outputs=[workflow_tree_html, preview_image, selected_node_id]
    )

    # 备注：实际的“保存”按钮应位于各个编辑子页面中。
    # 此处不再保留“模拟保存”按钮，以保持主界面的整洁。


if __name__ == "__main__":
    demo.launch()