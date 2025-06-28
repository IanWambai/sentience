"""
Camera capture and frame processing for Sentience.

This module handles webcam access, frame grabbing at ~5Hz,
and conversion to tensor format required by Gemma's vision stem.
"""

import time
import logging
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Try to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("⚠️ OpenCV not found. Attempting to use AVFoundation fallback.")
    
    # Try AVFoundation import (macOS)
    try:
        import AVFoundation
        AVFOUNDATION_AVAILABLE = True
    except ImportError:
        AVFOUNDATION_AVAILABLE = False
        logger.warning("⚠️ AVFoundation not available. Camera capture may not work.")


# Placeholder image for testing without camera
DEFAULT_IMAGE_PATH = Path(__file__).parent.parent / "assets" / "placeholder.jpg"


class CameraFeed:
    """
    Manages webcam access and frame processing with fallbacks.
    
    References:
    - OpenCV macOS install: https://docs.opencv.org/4.x/d0/db2/tutorial_macos_install.html
    """
    
    def __init__(self, width=640, height=480, device="mps", test_mode=False):
        """Initialize the camera feed with target resolution."""
        self.width = width
        self.height = height
        self.device = device
        self.camera = None
        self.last_capture_time = 0
        self.test_mode = test_mode
        self.frame_count = 0
        self.capture_method = None
        
        # Create test image if it doesn't exist
        if not DEFAULT_IMAGE_PATH.exists():
            self._create_placeholder_image()
        
        # Try to initialize camera (will use fallback if needed)
        try:
            self._initialize_camera()
        except Exception as e:
            logger.error(f"Camera initialization error: {str(e)}")
            if test_mode:
                logger.warning("ℹ️ Running in test mode with placeholder image")
            else:
                raise RuntimeError(f"Failed to initialize camera: {str(e)}")
        
    def _initialize_camera(self):
        """Open the default macOS camera with fallbacks."""
        # Try OpenCV first
        if CV2_AVAILABLE:
            try:
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                
                # Check if camera opened successfully
                if self.camera.isOpened():
                    ret, test_frame = self.camera.read()
                    if ret:
                        logger.info("✓ Camera initialized using OpenCV")
                        self.capture_method = "opencv"
                        return
                    else:
                        self.camera.release()
                        logger.warning("Camera opened but no frames could be read")
                else:
                    logger.warning("Failed to open camera with OpenCV")
            except Exception as e:
                logger.warning(f"OpenCV camera error: {str(e)}")
        
        # Try AVFoundation fallback (macOS specific)
        # This is a placeholder - real implementation would use AVFoundation
        if self.test_mode:
            logger.info("Using test image as camera feed")
            self.capture_method = "test"
            return
            
        # If we reach here, no camera method worked
        raise RuntimeError(
            "Failed to initialize camera. Please check camera permissions and connections.\n"
            "Try running 'sudo killall VDCAssistant' if the camera is in use by another app."
        )
    
    def _create_placeholder_image(self):
        """Create a simple test image for fallback mode."""
        # Create assets directory if it doesn't exist
        DEFAULT_IMAGE_PATH.parent.mkdir(exist_ok=True)
        
        # Create a simple gradient image
        img = Image.new('RGB', (640, 480))
        pixels = img.load()
        
        # Fill with gradient
        for i in range(img.width):
            for j in range(img.height):
                r = int(255 * i / img.width)
                g = int(255 * j / img.height)
                b = 100
                pixels[i, j] = (r, g, b)
                
        # Add text
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            text = "Camera not available\nUsing placeholder image"
            draw.text((img.width//4, img.height//3), text, fill="white")
        except Exception:
            pass  # Skip text if drawing fails
            
        # Save the image
        img.save(DEFAULT_IMAGE_PATH)
    
    def isOpened(self):
        """
        Wrapper for the cv2.VideoCapture.isOpened() method.
        Returns True if the camera is opened, False otherwise.
        """
        # The 'and' provides short-circuiting: if self.cap is None, it returns None (False-like)
        return self.camera and self.camera.isOpened()

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
            
        # Update frame counter (for debugging)
        self.frame_count += 1
        
        # Get frame based on available capture method
        pil_image = None
        
        try:
            if self.capture_method == "opencv":
                # OpenCV camera capture
                ret, frame = self.camera.read()
                if not ret:
                    if self.frame_count < 5:  # Critical failure at startup
                        raise RuntimeError("Failed to capture frame from camera")
                    else:
                        # Log error but try to continue with last good frame
                        logger.error("Camera frame capture failed - frame may be frozen")
                        # Create an error indicator frame
                        pil_image = self._create_error_frame()
                else:
                    # Convert BGR to RGB and create PIL image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
            elif self.capture_method == "test" or self.test_mode:
                # Test mode - use placeholder image
                pil_image = Image.open(DEFAULT_IMAGE_PATH)
                # Add some variation to test frames
                if hasattr(pil_image, "convert"):
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Brightness(pil_image)
                    factor = 0.8 + 0.4 * (self.frame_count % 5) / 4.0
                    pil_image = enhancer.enhance(factor)
            else:
                # No valid capture method
                raise RuntimeError("No camera capture method available")
                
        except Exception as e:
            logger.error(f"Frame capture error: {str(e)}")
            # Fall back to error frame
            pil_image = self._create_error_frame()
        
        # Resize to 640x640 (Gemma's expected input size)
        pil_image = pil_image.resize((640, 640))
        
        # Convert to tensor, normalize to [0,1], and convert to float16
        try:
            tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).contiguous()
            tensor = tensor.float().div(255.0).to(torch.float16)
            # Move tensor to the right device
            tensor = tensor.to(self.device)
        except Exception as e:
            logger.error(f"Tensor conversion error: {str(e)}")
            # Create an emergency tensor if conversion fails
            tensor = torch.zeros((3, 640, 640), dtype=torch.float16, device=self.device)
        
        # Update last capture time
        self.last_capture_time = time.time()
        
        return tensor
        
    def _create_error_frame(self):
        """Create a frame indicating camera error."""
        # Create a red-tinted frame to indicate error
        img = Image.new('RGB', (640, 480), color=(80, 0, 0))
        
        # Add error text if PIL draw is available
        try:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            text = "Camera Error\nPlease check connection"
            draw.text((img.width//4, img.height//3), text, fill="white")
        except Exception:
            pass  # Skip text if drawing fails
            
        return img
    
    def __del__(self):
        """Clean up camera resources."""
        try:
            if self.capture_method == "opencv" and self.camera is not None:
                self.camera.release()
                logger.debug("Camera resources released")
        except Exception as e:
            logger.warning(f"Error releasing camera resources: {str(e)}")
            
    def is_connected(self):
        """Check if camera is still connected and working."""
        if self.capture_method == "opencv":
            if self.camera and self.camera.isOpened():
                ret, _ = self.camera.read()
                return ret
            return False
        elif self.capture_method == "test":
            return True  # Test mode always "connected"
        return False
