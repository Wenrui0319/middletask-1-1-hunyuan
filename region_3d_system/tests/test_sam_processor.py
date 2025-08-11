"""
Test script for SAM (Segment Anything Model) processor.
Tests point, box, and polygon segmentation capabilities.
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import time
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def test_sam_processor():
    """Test SAM processor with different selection modes."""
    
    print("=" * 60)
    print("SAM Processor Test Suite")
    print("=" * 60)
    
    # Check if segment-anything is installed
    try:
        from segmentation.sam_processor import SAMProcessor
        print("✅ SAM processor module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import SAM processor: {e}")
        print("\nPlease install required packages:")
        print("pip install segment-anything opencv-python matplotlib")
        return
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    if device == "cpu":
        print("⚠️  Warning: CPU mode will be slower. GPU recommended for production.")
    
    # Initialize SAM processor
    print("\n📦 Initializing SAM model...")
    try:
        # Use smaller model for testing (vit_b is faster)
        sam = SAMProcessor(
            model_type="vit_b",  # smallest model for quick testing
            device=device,
            use_half_precision=False  # SAM doesn't work well with half precision
        )
        print("✅ SAM model initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize SAM: {e}")
        return
    
    # Load or create test image
    test_image = create_test_image()
    print(f"\n🖼️  Created test image: {test_image.shape}")
    
    # Set image for SAM
    print("\n🔄 Computing image embeddings...")
    start_time = time.time()
    sam.set_image(test_image, cache_embeddings=True)
    embedding_time = time.time() - start_time
    print(f"✅ Embeddings computed in {embedding_time:.2f}s")
    
    # Test 1: Point-based segmentation
    print("\n" + "=" * 40)
    print("Test 1: Point-based Segmentation")
    print("=" * 40)
    test_point_segmentation(sam, test_image)
    
    # Test 2: Box-based segmentation
    print("\n" + "=" * 40)
    print("Test 2: Box-based Segmentation")
    print("=" * 40)
    test_box_segmentation(sam, test_image)
    
    # Test 3: Polygon-based segmentation
    print("\n" + "=" * 40)
    print("Test 3: Polygon-based Segmentation")
    print("=" * 40)
    test_polygon_segmentation(sam, test_image)
    
    # Test 4: Mask refinement
    print("\n" + "=" * 40)
    print("Test 4: Mask Refinement")
    print("=" * 40)
    test_mask_refinement(sam, test_image)
    
    # Test 5: Performance benchmark
    print("\n" + "=" * 40)
    print("Test 5: Performance Benchmark")
    print("=" * 40)
    test_performance(sam, test_image)
    
    # Clear cache
    sam.clear_cache()
    print("\n✅ All tests completed!")
    print("=" * 60)


def create_test_image():
    """Create a synthetic test image with multiple objects."""
    # Create blank canvas
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    # Draw some shapes to segment (OpenCV uses BGR format)
    # Blue rectangle (BGR: 255,0,0)
    cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), -1)
    
    # Green circle (BGR: 0,255,0)
    cv2.circle(img, (350, 150), 80, (0, 255, 0), -1)
    
    # Red triangle (BGR: 0,0,255)
    triangle = np.array([[250, 300], [150, 450], [350, 450]], np.int32)
    cv2.fillPoly(img, [triangle], (0, 0, 255))
    
    # Complex shape (yellow)
    complex_shape = np.array([
        [400, 300], [450, 320], [480, 380], 
        [450, 420], [400, 400], [380, 350]
    ], np.int32)
    cv2.fillPoly(img, [complex_shape], (255, 255, 0))
    
    return img


def test_point_segmentation(sam, image):
    """Test point-based segmentation."""
    # Click on the blue rectangle
    point_coords = np.array([[125, 125]])  # Center of blue rectangle
    point_labels = np.array([1])  # 1 = foreground
    
    print("🎯 Testing point at (125, 125) - should select blue rectangle")
    
    start_time = time.time()
    result = sam.segment_with_points(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    seg_time = time.time() - start_time
    
    print(f"✅ Segmentation completed in {seg_time*1000:.1f}ms")
    print(f"   Generated {len(result['masks'])} masks")
    print(f"   Best mask score: {result['scores'].max():.3f}")
    
    # Visualize result
    visualize_result(image, result['best_mask'], "Point Segmentation")


def test_box_segmentation(sam, image):
    """Test box-based segmentation."""
    # Box around the green circle
    box = np.array([270, 70, 430, 230])  # [x1, y1, x2, y2]
    
    print("📦 Testing box selection - should select green circle")
    
    start_time = time.time()
    result = sam.segment_with_box(
        box=box,
        multimask_output=False
    )
    seg_time = time.time() - start_time
    
    print(f"✅ Segmentation completed in {seg_time*1000:.1f}ms")
    print(f"   Mask score: {result['scores'][0]:.3f}")
    
    # Visualize result
    visualize_result(image, result['best_mask'], "Box Segmentation", box=box)


def test_polygon_segmentation(sam, image):
    """Test polygon-based segmentation."""
    # Polygon around the red triangle
    polygon_points = np.array([
        [240, 290], [140, 460], [360, 460], [240, 290]
    ])
    
    print("📐 Testing polygon selection - should select red triangle")
    
    start_time = time.time()
    result = sam.segment_with_polygon(
        polygon_points=polygon_points,
        multimask_output=False
    )
    seg_time = time.time() - start_time
    
    print(f"✅ Segmentation completed in {seg_time*1000:.1f}ms")
    print(f"   Mask score: {result['scores'][0]:.3f}")
    
    # Visualize result
    visualize_result(image, result['best_mask'], "Polygon Segmentation", polygon=polygon_points)


def test_mask_refinement(sam, image):
    """Test mask refinement capabilities."""
    # Get a mask first
    point_coords = np.array([[125, 125]])
    point_labels = np.array([1])
    
    result = sam.segment_with_points(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False
    )
    
    original_mask = result['best_mask']
    
    print("🔧 Testing mask refinement")
    
    # Refine the mask
    refined_mask = sam.refine_mask(
        mask=original_mask,
        erosion_size=3,
        dilation_size=3,
        smooth_borders=True,
        min_area=100
    )
    
    # Calculate difference
    diff_pixels = np.sum(original_mask != refined_mask)
    total_pixels = original_mask.size
    diff_percent = (diff_pixels / total_pixels) * 100
    
    print(f"✅ Mask refined successfully")
    print(f"   Pixels changed: {diff_pixels} ({diff_percent:.2f}%)")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_mask, cmap='gray')
    axes[0].set_title('Original Mask')
    axes[0].axis('off')
    
    axes[1].imshow(refined_mask, cmap='gray')
    axes[1].set_title('Refined Mask')
    axes[1].axis('off')
    
    diff_viz = np.zeros_like(image)
    diff_viz[original_mask != refined_mask] = [255, 0, 0]
    axes[2].imshow(diff_viz)
    axes[2].set_title('Difference')
    axes[2].axis('off')
    
    plt.suptitle('Mask Refinement Comparison')
    save_or_show_plot("mask_refinement")


def test_performance(sam, image):
    """Benchmark performance with different selection types."""
    print("⚡ Running performance benchmark...")
    
    # Test cached vs uncached
    print("\n📊 Embedding Cache Test:")
    
    # First call (should use cache)
    start = time.time()
    sam.set_image(image, cache_embeddings=True)
    cached_time = time.time() - start
    print(f"   Cached embedding: {cached_time*1000:.1f}ms")
    
    # Clear cache and try again
    sam.clear_cache()
    start = time.time()
    sam.set_image(image, cache_embeddings=False)
    uncached_time = time.time() - start
    print(f"   Uncached embedding: {uncached_time*1000:.1f}ms")
    print(f"   Speedup: {uncached_time/cached_time:.1f}x")
    
    # Benchmark different selection types
    print("\n📊 Selection Type Benchmark:")
    
    # Point selection
    point_times = []
    for _ in range(5):
        point = np.array([[np.random.randint(0, 512), np.random.randint(0, 512)]])
        start = time.time()
        sam.segment_with_points(point, np.array([1]))
        point_times.append(time.time() - start)
    
    print(f"   Point selection: {np.mean(point_times)*1000:.1f}ms (±{np.std(point_times)*1000:.1f}ms)")
    
    # Box selection
    box_times = []
    for _ in range(5):
        x1, y1 = np.random.randint(0, 400, 2)
        box = np.array([x1, y1, x1+100, y1+100])
        start = time.time()
        sam.segment_with_box(box)
        box_times.append(time.time() - start)
    
    print(f"   Box selection: {np.mean(box_times)*1000:.1f}ms (±{np.std(box_times)*1000:.1f}ms)")
    
    # Check performance targets
    print("\n🎯 Performance Targets:")
    avg_time = np.mean(point_times + box_times)
    if avg_time < 2.0:
        print(f"   ✅ Met target: <2s for 512x512 image ({avg_time:.2f}s)")
    else:
        print(f"   ⚠️  Missed target: {avg_time:.2f}s > 2s")


def visualize_result(image, mask, title, box=None, polygon=None):
    """Visualize segmentation result with texture overlay."""
    viz = image.copy()
    
    # Create a texture pattern (diagonal stripes)
    h, w = image.shape[:2]
    texture = np.zeros((h, w), dtype=np.uint8)
    
    # Create diagonal stripe pattern
    stripe_width = 8  # Width of stripes
    stripe_spacing = 16  # Total spacing between stripe starts
    
    for i in range(-h, w + h, stripe_spacing):
        # Create diagonal lines from top-left to bottom-right
        for j in range(stripe_width):
            y_coords = np.arange(h)
            x_coords = i + j - y_coords
            
            # Keep coordinates within bounds
            valid = (x_coords >= 0) & (x_coords < w)
            y_coords = y_coords[valid]
            x_coords = x_coords[valid]
            
            if len(x_coords) > 0:
                texture[y_coords, x_coords] = 255
    
    # Alternative patterns (uncomment to use):
    
    # # Dots pattern
    # for y in range(0, h, 12):
    #     for x in range(0, w, 12):
    #         cv2.circle(texture, (x, y), 2, 255, -1)
    
    # # Crosshatch pattern
    # for i in range(0, max(h, w), 15):
    #     cv2.line(texture, (i, 0), (i, h), 255, 1)  # Vertical lines
    #     cv2.line(texture, (0, i), (w, i), 255, 1)  # Horizontal lines
    
    # Determine the dominant color of the masked region to choose contrasting texture
    masked_pixels = image[mask]
    if len(masked_pixels) > 0:
        # Calculate average color in the masked region (BGR format)
        avg_color = np.mean(masked_pixels, axis=0)
        b, g, r = avg_color
        
        # Choose contrasting texture color based on dominant channel
        if r > 150 and r > g and r > b:  # Red dominant
            # Use cyan/white stripes for red objects
            texture_color = np.array([255, 255, 100])  # Light cyan (BGR)
            contour_color = (255, 255, 0)  # Cyan
        elif g > 150 and g > b and g > r:  # Green dominant
            # Use magenta/purple stripes for green objects
            texture_color = np.array([255, 0, 255])  # Magenta (BGR)
            contour_color = (255, 0, 255)  # Magenta
        elif b > 150 and b > g and b > r:  # Blue dominant
            # Use yellow stripes for blue objects
            texture_color = np.array([0, 255, 255])  # Yellow (BGR)
            contour_color = (0, 255, 255)  # Yellow
        elif (r + g + b) / 3 > 200:  # Light/white background
            # Use dark blue stripes for light objects
            texture_color = np.array([128, 0, 0])  # Dark blue (BGR)
            contour_color = (128, 0, 0)  # Dark blue
        else:  # Dark or mixed colors
            # Use bright white stripes for dark objects
            texture_color = np.array([255, 255, 255])  # White (BGR)
            contour_color = (255, 255, 255)  # White
    else:
        # Default to white if no masked pixels
        texture_color = np.array([255, 255, 255])
        contour_color = (255, 255, 255)
    
    # Create colored texture
    texture_colored = np.zeros_like(image)
    texture_colored[:, :, 0] = (texture > 0) * texture_color[0]
    texture_colored[:, :, 1] = (texture > 0) * texture_color[1]
    texture_colored[:, :, 2] = (texture > 0) * texture_color[2]
    
    # Apply texture only where mask is True
    texture_mask = mask & (texture > 0)
    viz[texture_mask] = viz[texture_mask] * 0.5 + texture_colored[texture_mask] * 0.5
    
    # Add contour for clear boundary
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(viz, contours, -1, contour_color, 3)  # Matching contour color
    
    # Draw selection indicator
    if box is not None:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    if polygon is not None:
        cv2.polylines(viz, [polygon.astype(int)], True, (255, 0, 0), 2)
    
    # Display or save
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    
    save_or_show_plot(title.lower().replace(' ', '_'))


def save_or_show_plot(filename):
    """Save plot to file or show it."""
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / f"{filename}.png", dpi=100, bbox_inches='tight')
    # Uncomment to show plots interactively:
    # plt.show()
    plt.close()
    print(f"   📸 Saved visualization: test_outputs/{filename}.png")


if __name__ == "__main__":
    test_sam_processor()