# workflow_manager.py

import json
import os
import uuid
from datetime import datetime

class WorkflowManager:
    """
    管理图像编辑工作流的状态，包括节点的创建、删除、查询，
    以及将工作流数据持久化为 JSON 文件。
    """
    def __init__(self, filepath="workflow.json", workspace_dir="workspace"):
        """
        初始化管理器。

        Args:
            filepath (str): 工作流JSON文件的路径。
            workspace_dir (str): 用于存放上传图片、中间结果和遮罩的目录。
        """
        self.filepath = filepath
        self.workspace_dir = workspace_dir
        self.nodes = []
        
        # 确保工作区目录存在
        os.makedirs(os.path.join(self.workspace_dir, "uploads"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_dir, "masks"), exist_ok=True)

        self.load()

    def load(self):
        """
        从 JSON 文件加载节点数据。如果文件不存在，则初始化一个空的工作流。
        """
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.nodes = data.get("nodes", [])
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading workflow file: {e}. Starting with a new workflow.")
                self.nodes = []
        else:
            self.nodes = []

    def save(self):
        """
        将当前节点数据和元数据保存到 JSON 文件。
        """
        data_to_save = {
            "metadata": {
                "version": "1.0",
                "last_saved": datetime.now().isoformat()
            },
            "nodes": self.nodes
        }
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)

    def get_node(self, node_id):
        """
        根据 ID 查找并返回节点对象。

        Args:
            node_id (str): 要查找的节点 ID。

        Returns:
            dict | None: 找到的节点对象，如果未找到则返回 None。
        """
        for node in self.nodes:
            if node['id'] == node_id:
                return node
        return None

    def to_mermaid(self):
        """
        将当前工作流数据转换为 Mermaid.js 格式的字符串，并包含用于渲染的 HTML。

        Returns:
            str: 包含 Mermaid 图表定义和 script 标签的完整 HTML 字符串。
        """
        if not self.nodes:
            return "<div class='mermaid-container'><p>工作流为空，请上传一张图片以开始。</p></div>"

        # 替换在 ID 中可能影响 Mermaid 语法的字符
        def sanitize_id(node_id):
            return node_id.replace('-', '_')

        mermaid_string = "graph TD;\n"
        
        # 定义节点
        for node in self.nodes:
            s_id = sanitize_id(node['id'])
            label = node['label'].replace('"', '') # 移除双引号
            mermaid_string += f'    {s_id}["{label}"];\n'
        
        mermaid_string += "\n"

        # 定义连接
        for node in self.nodes:
            if node['parent_id']:
                s_parent_id = sanitize_id(node['parent_id'])
                s_id = sanitize_id(node['id'])
                mermaid_string += f'    {s_parent_id} --> {s_id};\n'

        mermaid_string += "\n"

        # 定义样式
        mermaid_string += "    classDef leafNode fill:#3498db,color:#fff,stroke:#fff,stroke-width:2px;\n"
        mermaid_string += "    classDef opNode fill:#2980b9,color:#fff,stroke:#fff,stroke-width:2px;\n\n"

        # 应用样式和点击事件
        for node in self.nodes:
            s_id = sanitize_id(node['id'])
            node_class = "leafNode" if node['type'] == 'leaf' else "opNode"
            mermaid_string += f"    class {s_id} {node_class};\n"
            # 注意：Mermaid 的 click 事件需要原始 ID
            mermaid_string += f'    click {s_id} call handleNodeClick("{node["id"]}");\n'

        # 完整 HTML 输出
        # 我们需要一个唯一的 div ID 来确保 mermaid.js 每次都能重新渲染
        container_id = f"mermaid-container-{uuid.uuid4().hex}"
        
        html_output = f"""
        <div id="{container_id}" class="mermaid">
            {mermaid_string}
        </div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: false }});
            async function renderMermaid() {{
                const element = document.getElementById('{container_id}');
                if (element) {{
                    const {{ svg }} = await mermaid.render('graph-svg', element.textContent);
                    element.innerHTML = svg;
                }}
            }}
            // 使用 setTimeout 确保 DOM 更新后再执行渲染
            setTimeout(renderMermaid, 100);
        </script>
        """
        return html_output
