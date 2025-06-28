"""
Model and asset downloader for Sentience.

This module handles the automatic downloading of the Gemma 3n model
and any other required assets when the application is first run.
"""

import os
import sys
import time
import logging
import platform
import psutil
from pathlib import Path
from tqdm import tqdm

from huggingface_hub import snapshot_download, HfFolder
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
from transformers import AutoConfig


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AssetManager:
    """
    Manages the downloading and verification of models and assets.
    """
    
    # Model constants
    # Using the E2B model with 256-dim attention heads for MPS compatibility
    MODEL_ID = "google/gemma-3n-E2B"  # This is the correct Hugging Face repository name
    WEIGHTS_DIR = "weights/gemma_e2b"
    MODEL_SIZE_GB = 2.0   # E2B model is ~2GB, much smaller than E4B
    REQUIRED_MEMORY_GB = 8.0  # E2B model requires less RAM
    
    def __init__(self):
        """Initialize the asset manager."""
        self.base_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = self.base_path / self.WEIGHTS_DIR
        self.download_timeout = 3600  # 1 hour timeout for large downloads
    
    def ensure_model_available(self):
        """
        Check if the model exists locally; if not, download it.
        Also verifies system has adequate memory before proceeding.
        
        Returns:
            str: Path to the model directory
        """
        # First check if system has enough memory to run the model
        if not self._check_system_requirements():
            self._handle_download_error("Insufficient system memory", exit_code=2)
        
        # Check for existing model files
        if self._is_model_complete():
            logger.info(f"✓ Gemma 3n model already downloaded at {self.model_path}")
            return str(self.model_path)
        
        # Display prominent download message
        self._show_download_banner()
        
        # Create the target directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Attempt to download the model
        start_time = time.time()
        try:
            logger.info(f"Starting download of Gemma 3n ({self.MODEL_SIZE_GB:.1f} GB)...")
            logger.info("This will take some time. Please don't interrupt the process.")
            
            # Download the model files from Hugging Face with progress tracking
            snapshot_download(
                repo_id=self.MODEL_ID,
                local_dir=str(self.model_path),
                ignore_patterns=["*.bin*", "*.gguf"],  # Exclude non-safetensor model files
                local_dir_use_symlinks=False,  # Full download, not symlinks
                tqdm_class=tqdm,  # Show progress bar
                max_workers=4,  # Optimize download speed
                etag_timeout=self.download_timeout  # Extend timeout for large files
            )
            
            download_time = time.time() - start_time
            download_speed = self.MODEL_SIZE_GB / (download_time / 60)  # GB per minute
            
            logger.info(f"✓ Model download complete in {download_time:.1f} seconds")
            logger.info(f"  Average download speed: {download_speed:.2f} GB/min")
            
            # Verify the downloaded model
            if not self._is_model_complete():
                logger.error("⚠️ Model verification failed after download.")
                self._handle_download_error("Incomplete or corrupted download")
            
            return str(self.model_path)
            
        except RepositoryNotFoundError:
            self._handle_download_error(
                f"Repository '{self.MODEL_ID}' not found on Hugging Face.\n"
                "Please check the model ID and your internet connection."
            )
        except RevisionNotFoundError:
            self._handle_download_error(
                f"Specific version of '{self.MODEL_ID}' not found.\n"
                "The model may have been updated or removed."
            )
        except Exception as e:
            self._handle_download_error(str(e))
    
    def _is_model_complete(self):
        """
        Check if the model files are already downloaded and valid.
        
        Returns:
            bool: True if model files exist and are valid, False otherwise
        """
        if not self.model_path.exists():
            return False
        
        # Check for essential model files
        # For sharded models, the index file is the key verification target
        required_files = ["config.json", "model.safetensors.index.json", "preprocessor_config.json"]
        for file in required_files:
            if not (self.model_path / file).exists():
                logger.debug(f"Missing required model file: {file}")
                return False
        
        # Verify model integrity by loading config
        try:
            _ = AutoConfig.from_pretrained(
                str(self.model_path), 
                local_files_only=True, 
                trust_remote_code=True
            )
            return True
        except Exception as e:
            logger.debug(f"Model validation failed: {str(e)}")
            return False
    
    def _show_download_banner(self):
        """
        Display a clear banner about the upcoming download.
        """
        banner = f"""
╔{'═' * 78}╗
║{' ' * 78}║
║{' DOWNLOADING GEMMA 3n MODEL '.center(78)}║
║{' ' * 78}║
║{f' Size: ~{self.MODEL_SIZE_GB:.1f} GB | Estimated time: 10-30 minutes depending on connection '.center(78)}║
║{' ' * 78}║
║{' This will only happen once. The model will be saved for future use. '.center(78)}║
║{' ' * 78}║
╚{'═' * 78}╝
        """
        print(banner)
    
    def _handle_download_error(self, error_message, exit_code=1):
        """
        Handle errors during model download with clear user messaging.
        
        Args:
            error_message: The error message to display
            exit_code: The exit code to use (default: 1)
        """
        logger.error("❌ Model download failed: " + error_message)
        
        print("\n" + "="*60)
        print("ERROR: Unable to download or run Gemma 3n model")
        print("="*60)
        print(f"\nDetails: {error_message}")
        print("\nPlease try one of the following solutions:")
        
        if exit_code == 2:  # Memory-related error
            print(f"  1. Free up memory (at least {self.REQUIRED_MEMORY_GB} GB required)")
            print("  2. Close other applications consuming significant memory")
            if platform.system() == "Darwin":  # macOS
                print("  3. Verify your Apple Silicon Mac has at least 16GB RAM")
        else:  # Download-related error
            print("  1. Check your internet connection and try again")
            print("  2. Manually download the model from Hugging Face:")
            print(f"     https://huggingface.co/{self.MODEL_ID}")
            print(f"     and place files in: {self.model_path}")
            
        print("\nSentience will now exit.")
        sys.exit(exit_code)

    def _check_system_requirements(self):
        """
        Check if the system has enough memory to run the model.
        
        On macOS (Apple Silicon), we check against the total physical memory,
        as the OS handles memory management very efficiently. On other systems,
        we check available memory.

        Returns:
            bool: True if system has enough memory, False otherwise
        """
        mem = psutil.virtual_memory()
        if platform.system() == "Darwin":  # macOS
            total_memory_gb = mem.total / (1024.0 ** 3)
            if total_memory_gb < self.REQUIRED_MEMORY_GB:
                logger.error(f"Insufficient system memory: {total_memory_gb:.2f} GB total, but {self.REQUIRED_MEMORY_GB} GB is recommended to run this model.")
                return False
            logger.info(f"System memory check passed: {total_memory_gb:.2f} GB total memory detected.")
            return True
        else:  # Other OS
            available_memory = mem.available / (1024.0 ** 3)  # Convert to GB
            if available_memory < self.REQUIRED_MEMORY_GB:
                logger.error(f"Insufficient system memory: {available_memory:.2f} GB available, but {self.REQUIRED_MEMORY_GB} GB required")
                return False
            
            logger.info(f"System memory check passed: {available_memory:.2f} GB available.")
            return True
