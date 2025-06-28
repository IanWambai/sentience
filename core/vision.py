"""
Camera capture and frame processing for Sentience.

This module handles webcam access, frame grabbing at ~5Hz,
and conversion to tensor format required by Gemma's vision stem.
"""

import time
import torch
from PIL import Image
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    import warnings
    warnings.warn("OpenCV not found. Falling back to AVFoundation for camera capture.")


class CameraFeed:
    """
    Manages webcam access and frame processing.
    
    References:
    - OpenCV macOS install: https://docs.opencv.org/4.x/d0/db2/tutorial_macos_install.html
    """
    
    def __init__(self, width=640, height=480, device="mps"):
        """Initialize the camera feed with target resolution."""
        self.width = width
        self.height = height
        self.device = device
        self.camera = None
        self.last_capture_time = 0
        self._initialize_camera()
        
    def _initialize_camera(self):
        """Open the default macOS camera."""
        if CV2_AVAILABLE:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self.camera.isOpened():
                raise RuntimeError("Failed to open camera with OpenCV")
        else:
            # Fallback to AVFoundation would be implemented here
            # For now, we'll raise an error since it's beyond this implementation
            raise RuntimeError("OpenCV not available and AVFoundation fallback not implemented")
    
    def get_frame(self):
        """
        Grab a frame from the webcam and convert to tensor format.
        
        Returns:
            torch.Tensor: Normalized image tensor in [0,1] float16 format.
        """
        # Track time for consistent frame rate
        current_time = time.time()
        elapsed = current_time - self.last_capture_time
        if elapsed < 0.2:  # Target 5Hz (200ms between frames)
            time.sleep(max(0, 0.2 - elapsed))
        
        # Capture frame
        if CV2_AVAILABLE:
            ret, frame = self.camera.read()
            if not ret:
                raise RuntimeError("Failed to capture frame from camera")
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and resize to 640x640 (Gemma's expected input size)
            pil_image = Image.fromarray(frame_rgb).resize((640, 640))
        else:
            raise RuntimeError("No camera capture method available")
        
        # Convert to tensor, normalize to [0,1], and convert to float16
        tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).contiguous()
        tensor = tensor.float().div(255.0).to(torch.float16)
        
        # Move tensor to the right device
        tensor = tensor.to(self.device)
        
        # Update last capture time
        self.last_capture_time = time.time()
        
        return tensor
    
    def __del__(self):
        """Clean up camera resources."""
        if CV2_AVAILABLE and self.camera is not None:
            self.camera.release()
