"""
Model and asset downloader for Sentience.

This module handles the automatic downloading of the Gemma 3n model
and any other required assets when the application is first run.
"""

import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig


class AssetManager:
    """
    Manages the downloading and verification of models and assets.
    """
    
    # Model constants
    MODEL_ID = "google/gemma-3n-E2B"
    WEIGHTS_DIR = "weights/gemma_e2b_int4"
    
    def __init__(self):
        """Initialize the asset manager."""
        self.base_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = self.base_path / self.WEIGHTS_DIR
    
    def ensure_model_available(self):
        """
        Check if the model exists locally; if not, download it.
        
        Returns:
            str: Path to the model directory
        """
        if self._is_model_complete():
            print(f"Gemma 3n model already downloaded at {self.model_path}")
            return str(self.model_path)
            
        print("\n" + "="*80)
        print("Downloading Gemma 3n E2B model (first run only)")
        print("This may take several minutes depending on your connection.")
        print("="*80 + "\n")
        
        # Create the target directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        start_time = time.time()
        try:
            # Download the model files from Hugging Face
            snapshot_download(
                repo_id=self.MODEL_ID,
                local_dir=str(self.model_path),
                ignore_patterns=["*.safetensors.index.json"],  # Exclude unnecessary files
                local_dir_use_symlinks=False  # Full download, not symlinks
            )
            
            download_time = time.time() - start_time
            print(f"\nModel download complete in {download_time:.1f} seconds")
            
            return str(self.model_path)
            
        except Exception as e:
            print(f"\nError downloading model: {str(e)}")
            print("\nPlease check your internet connection and try again.")
            print("If the problem persists, you may need to download the model manually:")
            print(f"1. Visit: https://huggingface.co/{self.MODEL_ID}")
            print(f"2. Download the files to: {self.model_path}")
            sys.exit(1)
    
    def _is_model_complete(self):
        """
        Check if the model files are already downloaded.
        
        Returns:
            bool: True if model files exist, False otherwise
        """
        if not self.model_path.exists():
            return False
            
        # Check for critical model files
        try:
            # Try to load the config which should exist if model is present
            config_path = self.model_path / "config.json"
            if not config_path.exists():
                return False
                
            # Attempt to validate the config
            _ = AutoConfig.from_pretrained(str(self.model_path), local_files_only=True)
            return True
            
        except Exception:
            return False
