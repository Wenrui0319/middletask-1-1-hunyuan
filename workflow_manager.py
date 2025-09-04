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
