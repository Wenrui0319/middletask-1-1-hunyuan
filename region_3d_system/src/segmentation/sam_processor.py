"""
SAM (Segment Anything Model) integration for precise region segmentation.
Handles point, box, and polygon prompts for interactive selection.
"""

import numpy as np
import torch
import cv2
from typing import Optional, List, Tuple, Dict, Any
from segment_anything import sam_model_registry, SamPredictor
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SAMProcessor:
    """
    Handles SAM model initialization and inference for region segmentation.
    
    Performance targets:
    - Segmentation: <2s for 2K images
    - Memory usage: <4GB GPU
    """
    
    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        use_half_precision: bool = True
    ):
        """
        Initialize SAM model with optimization settings.
        
        Args:
            model_type: SAM model variant (vit_h, vit_l, vit_b)
            checkpoint_path: Path to SAM checkpoint
            device: Compute device (cuda/cpu)
            use_half_precision: Use FP16 for faster inference
        """
        self.device = device
        self.use_half = use_half_precision and device == "cuda"
        
        # Download checkpoint if not provided
        if checkpoint_path is None:
            checkpoint_path = self._download_checkpoint(model_type)
        
        # Initialize model
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.model.to(device=device)
        
        if self.use_half:
            self.model.half()
        
        self.model.eval()
        self.predictor = SamPredictor(self.model)
        
        # Cache for performance
        self._image_embedding_cache = {}
        self._current_image_hash = None
        
        logger.info(f"SAM initialized: {model_type} on {device}")
    
    def _download_checkpoint(self, model_type: str) -> str:
        """Download SAM checkpoint if not available locally."""
        checkpoints = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        }
        
        cache_dir = Path.home() / ".cache" / "sam"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = cache_dir / f"sam_{model_type}.pth"
        
        if not checkpoint_file.exists():
            import urllib.request
            logger.info(f"Downloading SAM checkpoint: {model_type}")
            urllib.request.urlretrieve(checkpoints[model_type], checkpoint_file)
        
        return str(checkpoint_file)
    
    def set_image(self, image: np.ndarray, cache_embeddings: bool = True) -> None:
        """
        Set the image for segmentation and compute embeddings.
        
        Args:
            image: Input image (H, W, 3) in RGB format
            cache_embeddings: Cache embeddings for repeated segmentation
        """
        # Generate hash for caching
        image_hash = hash(image.tobytes()) if cache_embeddings else None
        
        # Check cache
        if image_hash and image_hash in self._image_embedding_cache:
            logger.info("Using cached image embeddings")
            self.predictor.features = self._image_embedding_cache[image_hash]["features"]
            self.predictor.original_size = self._image_embedding_cache[image_hash]["original_size"]
            self.predictor.input_size = self._image_embedding_cache[image_hash]["input_size"]
            self.predictor.is_image_set = True
        else:
            # Compute new embeddings
            self.predictor.set_image(image)
            
            # Cache if enabled
            if cache_embeddings and image_hash:
                self._image_embedding_cache[image_hash] = {
                    "features": self.predictor.features,
                    "original_size": self.predictor.original_size,
                    "input_size": self.predictor.input_size
                }
                # Limit cache size
                if len(self._image_embedding_cache) > 10:
                    self._image_embedding_cache.pop(next(iter(self._image_embedding_cache)))
        
        self._current_image_hash = image_hash
    
    def segment_with_points(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        multimask_output: bool = False,
        return_logits: bool = False
    ) -> Dict[str, Any]:
        """
        Segment region using point prompts.
        
        Args:
            point_coords: Point coordinates (N, 2)
            point_labels: Point labels (N,) - 1 for foreground, 0 for background
            multimask_output: Return multiple mask predictions
            return_logits: Return raw logits
        
        Returns:
            Dictionary containing masks, scores, and logits
        """
        # Convert inputs to float32 to avoid dtype issues with half precision
        point_coords = point_coords.astype(np.float32)
        point_labels = point_labels.astype(np.float32)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
            return_logits=return_logits
        )
        
        # When multimask_output is True, choose the mask with appropriate size
        # rather than just the highest score (which might be the background)
        if multimask_output:
            # Calculate area for each mask
            mask_areas = [np.sum(m) for m in masks]
            total_pixels = masks[0].size
            
            # Filter out masks that are too large (likely background)
            # or too small (likely noise)
            valid_masks = []
            valid_scores = []
            for i, (mask, score, area) in enumerate(zip(masks, scores, mask_areas)):
                coverage = area / total_pixels
                # Keep masks between 1% and 50% of image
                if 0.01 <= coverage <= 0.5:
                    valid_masks.append(i)
                    valid_scores.append(score)
            
            # Choose best valid mask
            if valid_masks:
                best_idx = valid_masks[np.argmax(valid_scores)]
                best_mask = masks[best_idx]
            else:
                # Fallback to smallest mask if no valid masks
                best_mask = masks[np.argmin(mask_areas)]
        else:
            best_mask = masks[0]
        
        return {
            "masks": masks,
            "scores": scores,
            "logits": logits if return_logits else None,
            "best_mask": best_mask
        }
    
    def segment_with_box(
        self,
        box: np.ndarray,
        multimask_output: bool = False,
        return_logits: bool = False
    ) -> Dict[str, Any]:
        """
        Segment region using bounding box prompt.
        
        Args:
            box: Bounding box coordinates (x1, y1, x2, y2)
            multimask_output: Return multiple mask predictions
            return_logits: Return raw logits
        
        Returns:
            Dictionary containing masks, scores, and logits
        """
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=multimask_output,
            return_logits=return_logits
        )
        
        # When multimask_output is True, choose the mask with appropriate size
        # rather than just the highest score (which might be the background)
        if multimask_output:
            # Calculate area for each mask
            mask_areas = [np.sum(m) for m in masks]
            total_pixels = masks[0].size
            
            # Filter out masks that are too large (likely background)
            # or too small (likely noise)
            valid_masks = []
            valid_scores = []
            for i, (mask, score, area) in enumerate(zip(masks, scores, mask_areas)):
                coverage = area / total_pixels
                # Keep masks between 1% and 50% of image
                if 0.01 <= coverage <= 0.5:
                    valid_masks.append(i)
                    valid_scores.append(score)
            
            # Choose best valid mask
            if valid_masks:
                best_idx = valid_masks[np.argmax(valid_scores)]
                best_mask = masks[best_idx]
            else:
                # Fallback to smallest mask if no valid masks
                best_mask = masks[np.argmin(mask_areas)]
        else:
            best_mask = masks[0]
        
        return {
            "masks": masks,
            "scores": scores,
            "logits": logits if return_logits else None,
            "best_mask": best_mask
        }
    
    def segment_with_polygon(
        self,
        polygon_points: np.ndarray,
        multimask_output: bool = False,
        return_logits: bool = False
    ) -> Dict[str, Any]:
        """
        Segment region using polygon prompt.
        
        Args:
            polygon_points: Polygon vertices (N, 2)
            multimask_output: Return multiple mask predictions
            return_logits: Return raw logits
        
        Returns:
            Dictionary containing masks, scores, and logits
        """
        # Strategy: Use bounding box of polygon + center point
        # This often works better than edge points
        
        # Get bounding box of polygon
        min_x, min_y = polygon_points.min(axis=0)
        max_x, max_y = polygon_points.max(axis=0)
        box = np.array([min_x, min_y, max_x, max_y])
        
        # Also add center point for better accuracy
        center = np.mean(polygon_points[:-1], axis=0, keepdims=True)
        
        # Use box with center point hint
        masks, scores, logits = self.predictor.predict(
            point_coords=center,
            point_labels=np.array([1]),
            box=box,
            multimask_output=multimask_output,
            return_logits=return_logits
        )
        
        # When multimask_output is True, choose the mask with appropriate size
        # rather than just the highest score (which might be the background)
        if multimask_output:
            # Calculate area for each mask
            mask_areas = [np.sum(m) for m in masks]
            total_pixels = masks[0].size
            
            # Filter out masks that are too large (likely background)
            # or too small (likely noise)
            valid_masks = []
            valid_scores = []
            for i, (mask, score, area) in enumerate(zip(masks, scores, mask_areas)):
                coverage = area / total_pixels
                # Keep masks between 1% and 50% of image
                if 0.01 <= coverage <= 0.5:
                    valid_masks.append(i)
                    valid_scores.append(score)
            
            # Choose best valid mask
            if valid_masks:
                best_idx = valid_masks[np.argmax(valid_scores)]
                best_mask = masks[best_idx]
            else:
                # Fallback to smallest mask if no valid masks
                best_mask = masks[np.argmin(mask_areas)]
        else:
            best_mask = masks[0]
        
        return {
            "masks": masks,
            "scores": scores,
            "logits": logits if return_logits else None,
            "best_mask": best_mask
        }
    
    def segment_combined(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = False,
        return_logits: bool = False
    ) -> Dict[str, Any]:
        """
        Segment using combined prompts (points + box + mask).
        
        Args:
            point_coords: Point coordinates (N, 2)
            point_labels: Point labels (N,)
            box: Bounding box (x1, y1, x2, y2)
            mask_input: Previous mask for refinement
            multimask_output: Return multiple masks
            return_logits: Return raw logits
        
        Returns:
            Dictionary containing masks, scores, and logits
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits
        )
        
        # When multimask_output is True, choose the mask with appropriate size
        # rather than just the highest score (which might be the background)
        if multimask_output:
            # Calculate area for each mask
            mask_areas = [np.sum(m) for m in masks]
            total_pixels = masks[0].size
            
            # Filter out masks that are too large (likely background)
            # or too small (likely noise)
            valid_masks = []
            valid_scores = []
            for i, (mask, score, area) in enumerate(zip(masks, scores, mask_areas)):
                coverage = area / total_pixels
                # Keep masks between 1% and 50% of image
                if 0.01 <= coverage <= 0.5:
                    valid_masks.append(i)
                    valid_scores.append(score)
            
            # Choose best valid mask
            if valid_masks:
                best_idx = valid_masks[np.argmax(valid_scores)]
                best_mask = masks[best_idx]
            else:
                # Fallback to smallest mask if no valid masks
                best_mask = masks[np.argmin(mask_areas)]
        else:
            best_mask = masks[0]
        
        return {
            "masks": masks,
            "scores": scores,
            "logits": logits if return_logits else None,
            "best_mask": best_mask
        }
    
    def refine_mask(
        self,
        mask: np.ndarray,
        erosion_size: int = 2,
        dilation_size: int = 2,
        smooth_borders: bool = True,
        min_area: int = 100
    ) -> np.ndarray:
        """
        Post-process and refine segmentation mask.
        
        Args:
            mask: Binary mask to refine
            erosion_size: Erosion kernel size
            dilation_size: Dilation kernel size
            smooth_borders: Apply border smoothing
            min_area: Remove components smaller than this
        
        Returns:
            Refined binary mask
        """
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Morphological operations
        if erosion_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
            mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=1)
        
        if dilation_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
            mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
        
        # Remove small components
        if min_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
            mask_uint8 = np.zeros_like(mask_uint8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    mask_uint8[labels == i] = 255
        
        # Smooth borders
        if smooth_borders:
            mask_uint8 = cv2.GaussianBlur(mask_uint8, (5, 5), 1)
            _, mask_uint8 = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
        
        return (mask_uint8 > 0).astype(bool)
    
    def _sample_polygon_points(self, polygon: np.ndarray, num_samples: int = 20) -> np.ndarray:
        """Sample points along polygon edges."""
        points = []
        n_vertices = len(polygon)
        
        for i in range(n_vertices):
            start = polygon[i]
            end = polygon[(i + 1) % n_vertices]
            
            # Sample points along edge
            samples_per_edge = max(2, num_samples // n_vertices)
            for t in np.linspace(0, 1, samples_per_edge, endpoint=False):
                point = start + t * (end - start)
                points.append(point)
        
        return np.array(points)
    
    def clear_cache(self):
        """Clear embedding cache to free memory."""
        self._image_embedding_cache.clear()
        self._current_image_hash = None
        torch.cuda.empty_cache() if self.device == "cuda" else None