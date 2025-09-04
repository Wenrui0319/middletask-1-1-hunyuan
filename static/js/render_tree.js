// 将渲染逻辑封装在一个全局对象中，以避免污染全局命名空间
window.WorkflowRenderer = (function() {
    // --- 私有变量和配置 ---
    const svg = d3.select(".workflow-svg");
    const container = document.getElementById("workflow-container");
    const linksGroup = svg.select(".workflow-links");
    const nodesGroup = svg.select(".workflow-nodes");

    const nodeRadius = 50; // 节点圆形背景的半径
    let selectedNodeId = null; // 当前选中的节点ID

    // D3的树布局生成器
    const treeLayout = d3.tree().nodeSize([nodeRadius * 2.5, 150]);

    // D3的缩放和平移行为
    const zoom = d3.zoom()
        .scaleExtent([0.1, 2]) // 缩放范围
        .on("zoom", (event) => {
            linksGroup.attr("transform", event.transform);
            nodesGroup.attr("transform", event.transform);
        });
    
    svg.call(zoom);

    // --- 私有函数 ---

    /**
     * 更新节点的视觉表现
     * @param {d3.Selection} nodeSelection - D3的节点选择集
     */
    function updateNodes(nodeSelection) {
        // 为每个新节点创建一个分组
        const nodeEnter = nodeSelection.enter()
            .append("g")
            .attr("class", "node")
            .attr("transform", d => `translate(${d.x}, ${d.y})`)
            .on("click", function(event, d) {
                // --- 节点点击事件 ---
                selectedNodeId = d.data.id;
                
                // 更新视觉选中效果
                d3.selectAll('.node').classed('selected', false);
                d3.select(this).classed('selected', true);
                
                // 调用在HTML中定义的函数，与Gradio后端通信
                if (window.selectNodeInGradio) {
                    window.selectNodeInGradio(d.data.id, d.data.file.path);
                }
            });

        // 添加圆形背景
        nodeEnter.append("circle")
            .attr("class", "node-background")
            .attr("r", nodeRadius);

        // 添加节点文本
        nodeEnter.append("text")
            .attr("dy", ".35em") // 垂直居中
            .text(d => d.data.name);
        
        // 更新现有节点的位置
        nodeSelection.transition()
            .duration(500)
            .attr("transform", d => `translate(${d.x}, ${d.y})`);

        // 移除旧节点
        nodeSelection.exit().remove();
    }

    /**
     * 更新连线的视觉表现
     * @param {d3.Selection} linkSelection - D3的连线选择集
     */
    function updateLinks(linkSelection) {
        // D3对角线生成器，用于创建平滑的曲线
        const diagonal = d3.linkVertical()
            .x(d => d.x)
            .y(d => d.y);

        // 绘制连线
        linkSelection.enter()
            .insert("path", "g")
            .attr("class", "link")
            .attr("d", diagonal)
            .merge(linkSelection)
            .transition()
            .duration(500)
            .attr("d", diagonal);
        
        linkSelection.exit().remove();
    }


    // --- 公开方法 ---
    return {
        /**
         * 渲染整个工作流树
         * @param {Object} treeData - 后端传递的JSON树数据
         */
        render: function(treeData) {
            if (!container || !svg.node()) {
                console.error("SVG container not found.");
                return;
            }

            // 使用D3的层次结构数据
            const root = d3.hierarchy(treeData);
            
            // 计算树的布局
            treeLayout(root);

            const nodes = root.descendants();
            const links = root.links();
            
            // 将树的中心对准容器中心
            const initialTransform = d3.zoomIdentity
                .translate(container.offsetWidth / 2, 80); // 80是顶部的偏移
            svg.call(zoom.transform, initialTransform);


            // --- 数据绑定和更新 ---
            const nodeSelection = nodesGroup.selectAll("g.node")
                .data(nodes, d => d.data.id); // 使用唯一ID作为key
            
            const linkSelection = linksGroup.selectAll("path.link")
                .data(links, d => d.target.data.id); // 连线也用目标ID作为key

            updateNodes(nodeSelection);
            updateLinks(linkSelection);
        },

        /**
         * 获取当前选中的节点ID
         */
        getSelectedNodeId: function() {
            return selectedNodeId;
        }
    };
})();