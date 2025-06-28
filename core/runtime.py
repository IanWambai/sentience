"""
Runtime system for Sentience.

Handles device detection, model initialization, and runs the continuous
perception-inference loop.
"""

import os
import time
import torch
import psutil
from pathlib import Path

from .model_interface import GemmaEngine
from .vision import CameraFeed
from .streamer import ThoughtSink
from .downloader import AssetManager


class Initialiser:
    """
    Handles system initialization and compatibility checks.
    
    References:
    - Apple PyTorch MPS: https://developer.apple.com/metal/pytorch/
    - Accelerate MPS guide: https://huggingface.co/docs/accelerate/en/usage_guides/mps
    """
    
    def __init__(self, device_preference="mps"):
        """
        Initialize the system with device preference.
        
        Args:
            device_preference (str): Preferred compute device ("mps" for Apple GPU)
        """
        self.device_preference = device_preference
        self.device = self._detect_device()
        self.mission = self._load_mission()
    
    def _detect_device(self):
        """Check for MPS availability and set device accordingly."""
        if self.device_preference == "mps" and torch.backends.mps.is_available():
            print(f"MPS (Metal Performance Shaders) device detected")
            return "mps"
        else:
            raise RuntimeError(
                "Sentience requires Apple Silicon with MPS support. "
                "This build only supports macOS 13+ on M1/M2/M3 processors."
            )
    
    def _load_mission(self):
        """Load the mission prompt from disk."""
        mission_path = Path(__file__).parent / "assets" / "mission.txt"
        if not mission_path.exists():
            raise FileNotFoundError(f"Mission file not found at {mission_path}")
        
        with open(mission_path, "r") as f:
            mission = f.read().strip()
        
        print(f"Mission loaded: {len(mission)} characters")
        return mission


def run():
    """
    Primary entry point for Sentience.
    
    Initializes the system and runs the continuous perception-inference loop.
    """
    print("\nSentience Cognition Engine initializing...")
    start_time = time.time()
    
    # Initialize system
    initializer = Initialiser(device_preference="mps")
    device = initializer.device
    
    # Ensure model is downloaded
    asset_manager = AssetManager()
    model_path = asset_manager.ensure_model_available()
    
    # Set up model engine
    engine = GemmaEngine(model_path=model_path, device=device)
    
    # Set up camera
    camera = CameraFeed(device=device)
    
    # Set up output stream
    thought_sink = ThoughtSink()
    
    # Report cold boot time
    cold_boot_time = time.time() - start_time
    print(f"\nSentience ready! Cold boot time: {cold_boot_time:.2f}s")
    
    # Performance monitoring
    iteration_count = 0
    throughput_start_time = time.time()
    
    # Main loop
    try:
        print("\nStarting continuous perception-inference loop...\n")
        
        while True:
            loop_start_time = time.time()
            
            # 1. Capture frame
            frame = camera.get_frame()
            
            # 2. Describe scene
            scene_text = engine.describe_scene(frame)
            
            # 3. Plan action based on scene
            plan_text = engine.plan_action(scene_text)
            
            # 4. Emit thought
            thought_sink.emit(scene_text, plan_text)
            
            # Performance monitoring
            iteration_count += 1
            
            if iteration_count == 1:
                first_thought_time = time.time() - start_time
                print(f"[Performance] First thought time: {first_thought_time:.2f}s")
            
            if iteration_count == 10:
                # Check memory usage after 10 iterations
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / (1024 * 1024)
                print(f"[Performance] Memory usage after 10 iterations: {memory_mb:.2f} MB")
                
            if iteration_count == 240:  # After 60 seconds at 4 Hz
                duration = time.time() - throughput_start_time
                throughput = iteration_count / duration
                print(f"[Performance] Sustained throughput: {throughput:.2f} thoughts/second")
            
            # 5. Sleep if needed to maintain ~5Hz
            elapsed = time.time() - loop_start_time
            sleep_time = max(0, 0.2 - elapsed)  # Target 5Hz (200ms per loop)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nSentience shutting down gracefully...")
    finally:
        # Final performance report
        total_runtime = time.time() - start_time
        print(f"\nTotal runtime: {total_runtime:.2f}s")
        print(f"Total thoughts generated: {iteration_count}")
        print(f"Average throughput: {iteration_count / total_runtime:.2f} thoughts/second")


if __name__ == "__main__":
    run()
