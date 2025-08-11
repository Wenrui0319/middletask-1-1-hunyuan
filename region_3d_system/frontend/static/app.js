/**
 * SAM Segmentation Demo - Interactive Frontend
 */

class SAMDemo {
    constructor() {
        this.imageCanvas = document.getElementById('imageCanvas');
        this.overlayCanvas = document.getElementById('overlayCanvas');
        this.imageCtx = this.imageCanvas.getContext('2d');
        this.overlayCtx = this.overlayCanvas.getContext('2d');
        
        this.currentImage = null;
        this.currentMode = 'point';
        this.isDrawing = false;
        this.points = [];
        this.box = null;
        this.currentMask = null;
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // File upload
        const imageUpload = document.getElementById('imageUpload');
        imageUpload.addEventListener('change', (e) => this.handleImageUpload(e));
        
        // Drag and drop
        const canvasWrapper = document.getElementById('canvasWrapper');
        canvasWrapper.addEventListener('dragover', (e) => {
            e.preventDefault();
            canvasWrapper.style.background = '#f0f0f0';
        });
        
        canvasWrapper.addEventListener('dragleave', () => {
            canvasWrapper.style.background = '';
        });
        
        canvasWrapper.addEventListener('drop', (e) => {
            e.preventDefault();
            canvasWrapper.style.background = '';
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                this.loadImage(file);
            }
        });
        
        // Mode selection
        document.querySelectorAll('.mode-button').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.mode-button').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentMode = btn.dataset.mode;
                this.clearOverlay();
            });
        });
        
        // Clear button
        document.getElementById('clearBtn').addEventListener('click', () => {
            this.clearAll();
        });
        
        // Canvas interaction
        this.imageCanvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.imageCanvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.imageCanvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.imageCanvas.addEventListener('click', (e) => this.handleClick(e));
        
        // Download buttons
        document.getElementById('downloadMask').addEventListener('click', () => {
            this.downloadMask();
        });
        
        document.getElementById('downloadImage').addEventListener('click', () => {
            this.downloadResult();
        });
    }
    
    handleImageUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.loadImage(file);
        }
    }
    
    loadImage(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.currentImage = img;
                this.displayImage();
                document.getElementById('placeholder').style.display = 'none';
                
                // Enable download buttons
                document.getElementById('downloadMask').disabled = false;
                document.getElementById('downloadImage').disabled = false;
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    displayImage() {
        if (!this.currentImage) return;
        
        // Calculate scale to fit canvas
        const maxWidth = this.imageCanvas.parentElement.clientWidth;
        const maxHeight = 600;
        
        let scale = Math.min(
            maxWidth / this.currentImage.width,
            maxHeight / this.currentImage.height,
            1
        );
        
        const width = this.currentImage.width * scale;
        const height = this.currentImage.height * scale;
        
        // Set canvas sizes
        this.imageCanvas.width = width;
        this.imageCanvas.height = height;
        this.overlayCanvas.width = width;
        this.overlayCanvas.height = height;
        
        // Draw image
        this.imageCtx.drawImage(this.currentImage, 0, 0, width, height);
    }
    
    handleClick(event) {
        if (this.currentMode !== 'point') return;
        
        const rect = this.imageCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // Add positive point
        this.points.push({ x, y, label: 1 });
        this.drawPoint(x, y, true);
        
        // Perform segmentation
        this.performPointSegmentation(x, y, 1);
    }
    
    handleMouseDown(event) {
        if (this.currentMode === 'box') {
            const rect = this.imageCanvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            
            this.isDrawing = true;
            this.box = { x1: x, y1: y, x2: x, y2: y };
        } else if (this.currentMode === 'polygon') {
            const rect = this.imageCanvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            
            this.points.push({ x, y });
            this.drawPolygon();
        }
    }
    
    handleMouseMove(event) {
        if (this.currentMode === 'box' && this.isDrawing && this.box) {
            const rect = this.imageCanvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            
            this.box.x2 = x;
            this.box.y2 = y;
            this.drawBox();
        }
    }
    
    handleMouseUp(event) {
        if (this.currentMode === 'box' && this.isDrawing && this.box) {
            this.isDrawing = false;
            
            // Perform box segmentation
            this.performBoxSegmentation(
                Math.min(this.box.x1, this.box.x2),
                Math.min(this.box.y1, this.box.y2),
                Math.max(this.box.x1, this.box.x2),
                Math.max(this.box.y1, this.box.y2)
            );
        }
    }
    
    drawPoint(x, y, positive = true) {
        this.overlayCtx.fillStyle = positive ? '#00ff00' : '#ff0000';
        this.overlayCtx.beginPath();
        this.overlayCtx.arc(x, y, 5, 0, 2 * Math.PI);
        this.overlayCtx.fill();
        
        this.overlayCtx.strokeStyle = '#ffffff';
        this.overlayCtx.lineWidth = 2;
        this.overlayCtx.stroke();
    }
    
    drawBox() {
        if (!this.box) return;
        
        this.clearOverlay();
        
        this.overlayCtx.strokeStyle = '#00ff00';
        this.overlayCtx.lineWidth = 2;
        this.overlayCtx.strokeRect(
            this.box.x1,
            this.box.y1,
            this.box.x2 - this.box.x1,
            this.box.y2 - this.box.y1
        );
    }
    
    drawPolygon() {
        if (this.points.length === 0) return;
        
        this.clearOverlay();
        
        this.overlayCtx.strokeStyle = '#00ff00';
        this.overlayCtx.lineWidth = 2;
        this.overlayCtx.beginPath();
        
        this.overlayCtx.moveTo(this.points[0].x, this.points[0].y);
        for (let i = 1; i < this.points.length; i++) {
            this.overlayCtx.lineTo(this.points[i].x, this.points[i].y);
        }
        
        if (this.points.length > 2) {
            this.overlayCtx.closePath();
        }
        
        this.overlayCtx.stroke();
        
        // Draw points
        this.points.forEach(point => {
            this.drawPoint(point.x, point.y);
        });
        
        // If polygon is closed (at least 3 points), perform segmentation
        if (this.points.length >= 3) {
            const polygonPoints = this.points.map(p => [p.x, p.y]);
            polygonPoints.push([this.points[0].x, this.points[0].y]); // Close polygon
            this.performPolygonSegmentation(polygonPoints);
        }
    }
    
    async performPointSegmentation(x, y, label) {
        this.showLoading(true);
        
        try {
            const imageData = this.imageCanvas.toDataURL('image/png');
            
            const formData = new FormData();
            formData.append('image_data', imageData);
            formData.append('x', x);
            formData.append('y', y);
            formData.append('label', label);
            
            const response = await fetch('/api/segment/point', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.displaySegmentation(result);
            }
        } catch (error) {
            console.error('Segmentation error:', error);
            alert('Segmentation failed: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }
    
    async performBoxSegmentation(x1, y1, x2, y2) {
        this.showLoading(true);
        
        try {
            const imageData = this.imageCanvas.toDataURL('image/png');
            
            const formData = new FormData();
            formData.append('image_data', imageData);
            formData.append('x1', x1);
            formData.append('y1', y1);
            formData.append('x2', x2);
            formData.append('y2', y2);
            
            const response = await fetch('/api/segment/box', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.displaySegmentation(result);
            }
        } catch (error) {
            console.error('Segmentation error:', error);
            alert('Segmentation failed: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }
    
    async performPolygonSegmentation(points) {
        this.showLoading(true);
        
        try {
            const imageData = this.imageCanvas.toDataURL('image/png');
            
            const formData = new FormData();
            formData.append('image_data', imageData);
            formData.append('points', JSON.stringify(points));
            
            const response = await fetch('/api/segment/polygon', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.displaySegmentation(result);
            }
        } catch (error) {
            console.error('Segmentation error:', error);
            alert('Segmentation failed: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }
    
    displaySegmentation(result) {
        // Store current mask
        this.currentMask = result.mask;
        
        // Display visualization
        const img = new Image();
        img.onload = () => {
            this.imageCtx.clearRect(0, 0, this.imageCanvas.width, this.imageCanvas.height);
            this.imageCtx.drawImage(img, 0, 0, this.imageCanvas.width, this.imageCanvas.height);
        };
        img.src = result.visualization;
        
        // Update statistics
        const stats = document.getElementById('stats');
        stats.innerHTML = `
            <p><strong>Score:</strong> ${result.score.toFixed(3)}</p>
            <p><strong>Area:</strong> ${result.area} pixels</p>
            <p><strong>Coverage:</strong> ${((result.area / (this.imageCanvas.width * this.imageCanvas.height)) * 100).toFixed(1)}%</p>
        `;
    }
    
    clearOverlay() {
        this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
        this.points = [];
        this.box = null;
    }
    
    clearAll() {
        this.clearOverlay();
        this.imageCtx.clearRect(0, 0, this.imageCanvas.width, this.imageCanvas.height);
        if (this.currentImage) {
            this.displayImage();
        }
        document.getElementById('stats').innerHTML = '<p>No selection yet</p>';
        this.currentMask = null;
    }
    
    showLoading(show) {
        const loadingEl = document.getElementById('loading');
        if (loadingEl) {
            loadingEl.style.display = show ? 'flex' : 'none';
        }
    }
    
    downloadMask() {
        if (!this.currentMask) {
            alert('No mask to download');
            return;
        }
        
        const link = document.createElement('a');
        link.download = 'mask.png';
        link.href = this.currentMask;
        link.click();
    }
    
    downloadResult() {
        const link = document.createElement('a');
        link.download = 'segmentation_result.png';
        link.href = this.imageCanvas.toDataURL('image/png');
        link.click();
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SAMDemo();
});