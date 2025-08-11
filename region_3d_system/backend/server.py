"""
FastAPI backend server for SAM segmentation demo.
Provides REST API for image segmentation using SAM.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import numpy as np
import cv2
import base64
import io
from PIL import Image
import json
from typing import Optional, List, Dict, Any
import uvicorn
import torch
import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from src.segmentation.sam_processor import SAMProcessor
except ImportError:
    # If running from backend directory
    sys.path.insert(0, str(parent_dir.parent))
    from src.segmentation.sam_processor import SAMProcessor

# Global SAM processor instance
sam_processor = None

def initialize_sam():
    """Initialize SAM model on startup."""
    global sam_processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    
    print("📦 Initializing SAM model...")
    sam_processor = SAMProcessor(
        model_type="vit_b",
        device=device,
        use_half_precision=False
    )
    print("✅ SAM initialized successfully")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    initialize_sam()
    yield
    # Shutdown
    print("👋 Shutting down...")

app = FastAPI(
    title="SAM Segmentation Demo", 
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Serve the main HTML page."""
    html_path = parent_dir / "frontend" / "index.html"
    return FileResponse(str(html_path))

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload and process an image for segmentation.
    Returns image ID and dimensions.
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image in SAM processor
        sam_processor.set_image(image, cache_embeddings=True)
        
        # Store image in memory (in production, use proper storage)
        image_id = base64.b64encode(contents).decode('utf-8')[:20]
        
        # Return image info
        return JSONResponse({
            "success": True,
            "image_id": image_id,
            "width": image.shape[1],
            "height": image.shape[0],
            "message": "Image uploaded successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/segment/point")
async def segment_point(
    image_data: str = Form(...),
    x: float = Form(...),
    y: float = Form(...),
    label: int = Form(1)
):
    """
    Perform point-based segmentation.
    """
    try:
        # Check if SAM is initialized
        if sam_processor is None:
            raise HTTPException(status_code=503, detail="SAM model not initialized")
        
        # Decode image
        image = decode_base64_image(image_data)
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        
        # Ensure image is in correct format
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {image.shape}. Expected (H, W, 3)")
        
        sam_processor.set_image(image)
        
        # Perform segmentation
        point_coords = np.array([[x, y]])
        point_labels = np.array([label])
        
        result = sam_processor.segment_with_points(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        
        # Get best mask
        mask = result['best_mask']
        
        # Create visualization
        viz_image = create_mask_visualization(image, mask)
        
        # Encode results
        mask_encoded = encode_mask(mask)
        viz_encoded = encode_image(viz_image)
        
        return JSONResponse({
            "success": True,
            "mask": mask_encoded,
            "visualization": viz_encoded,
            "score": float(result['scores'][0]),
            "area": int(np.sum(mask))
        })
        
    except Exception as e:
        print(f"Error in segment_point: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/segment/box")
async def segment_box(
    image_data: str = Form(...),
    x1: float = Form(...),
    y1: float = Form(...),
    x2: float = Form(...),
    y2: float = Form(...)
):
    """
    Perform box-based segmentation.
    """
    try:
        # Decode image
        image = decode_base64_image(image_data)
        sam_processor.set_image(image)
        
        # Perform segmentation
        box = np.array([x1, y1, x2, y2])
        
        result = sam_processor.segment_with_box(
            box=box,
            multimask_output=False
        )
        
        # Get mask
        mask = result['best_mask']
        
        # Create visualization
        viz_image = create_mask_visualization(image, mask)
        
        # Encode results
        mask_encoded = encode_mask(mask)
        viz_encoded = encode_image(viz_image)
        
        return JSONResponse({
            "success": True,
            "mask": mask_encoded,
            "visualization": viz_encoded,
            "score": float(result['scores'][0]),
            "area": int(np.sum(mask))
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/segment/polygon")
async def segment_polygon(
    image_data: str = Form(...),
    points: str = Form(...)
):
    """
    Perform polygon-based segmentation.
    """
    try:
        # Decode image
        image = decode_base64_image(image_data)
        sam_processor.set_image(image)
        
        # Parse polygon points
        points_list = json.loads(points)
        polygon_points = np.array(points_list)
        
        result = sam_processor.segment_with_polygon(
            polygon_points=polygon_points,
            multimask_output=False
        )
        
        # Get mask
        mask = result['best_mask']
        
        # Create visualization
        viz_image = create_mask_visualization(image, mask)
        
        # Encode results
        mask_encoded = encode_mask(mask)
        viz_encoded = encode_image(viz_image)
        
        return JSONResponse({
            "success": True,
            "mask": mask_encoded,
            "visualization": viz_encoded,
            "score": float(result['scores'][0]),
            "area": int(np.sum(mask))
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/segment/everything")
async def segment_everything(image_data: str = Form(...)):
    """
    Segment all objects in the image (automatic mode).
    """
    try:
        # Decode image
        image = decode_base64_image(image_data)
        
        # For now, return a placeholder
        # TODO: Implement automatic segmentation
        
        return JSONResponse({
            "success": True,
            "message": "Automatic segmentation coming soon",
            "segments": []
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def decode_base64_image(image_data: str) -> np.ndarray:
    """Decode base64 image to numpy array."""
    # Remove data URL prefix if present
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    # Decode base64
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array (RGB)
    image_array = np.array(image)
    
    # Ensure it's uint8 and has 3 channels
    if len(image_array.shape) == 2:
        # Grayscale image, convert to RGB
        image_array = np.stack([image_array] * 3, axis=-1)
    
    return image_array.astype(np.uint8)

def encode_image(image: np.ndarray) -> str:
    """Encode numpy array to base64 image."""
    # Convert to PIL Image
    pil_image = Image.fromarray(image.astype(np.uint8))
    
    # Save to bytes
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    
    # Encode to base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

def encode_mask(mask: np.ndarray) -> str:
    """Encode binary mask to base64."""
    # Convert boolean mask to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Encode as PNG
    _, buffer = cv2.imencode('.png', mask_uint8)
    mask_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/png;base64,{mask_base64}"

def create_mask_visualization(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create visualization with adaptive texture overlay."""
    viz = image.copy()
    
    # Create diagonal stripe texture
    h, w = image.shape[:2]
    texture = np.zeros((h, w), dtype=np.uint8)
    
    stripe_width = 8
    stripe_spacing = 16
    
    for i in range(-h, w + h, stripe_spacing):
        for j in range(stripe_width):
            y_coords = np.arange(h)
            x_coords = i + j - y_coords
            valid = (x_coords >= 0) & (x_coords < w)
            y_coords = y_coords[valid]
            x_coords = x_coords[valid]
            if len(x_coords) > 0:
                texture[y_coords, x_coords] = 255
    
    # Determine contrasting color
    masked_pixels = image[mask]
    if len(masked_pixels) > 0:
        avg_color = np.mean(masked_pixels, axis=0)
        r, g, b = avg_color
        
        if r > 150 and r > g and r > b:
            texture_color = np.array([100, 255, 255])  # Cyan for red
            contour_color = (0, 255, 255)
        elif g > 150 and g > b and g > r:
            texture_color = np.array([255, 0, 255])  # Magenta for green
            contour_color = (255, 0, 255)
        elif b > 150 and b > g and b > r:
            texture_color = np.array([255, 255, 0])  # Yellow for blue
            contour_color = (255, 255, 0)
        else:
            texture_color = np.array([255, 255, 255])  # White default
            contour_color = (255, 255, 255)
    else:
        texture_color = np.array([255, 255, 255])
        contour_color = (255, 255, 255)
    
    # Apply texture
    texture_colored = np.zeros_like(image)
    texture_colored[:, :, 0] = (texture > 0) * texture_color[0]
    texture_colored[:, :, 1] = (texture > 0) * texture_color[1]
    texture_colored[:, :, 2] = (texture > 0) * texture_color[2]
    
    texture_mask = mask & (texture > 0)
    viz[texture_mask] = viz[texture_mask] * 0.5 + texture_colored[texture_mask] * 0.5
    
    # Add contour
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(viz, contours, -1, contour_color, 3)
    
    return viz

# Mount static files
static_dir = parent_dir / "frontend" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

if __name__ == "__main__":
    # Use app string for reload to work properly
    uvicorn.run(
        "server:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_dirs=["backend", "src"]
    )