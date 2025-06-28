"""
Runtime system for Sentience.

Handles device detection, model initialization, and runs the continuous
perception-inference loop.
"""

import os
import sys
import time
import torch
import psutil
import logging
from pathlib import Path

from .model_interface import GemmaEngine
from .vision import CameraFeed
from .audio import AudioStream
from .streamer import ThoughtSink
from .downloader import AssetManager

# Configure logging
logger = logging.getLogger(__name__)


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


def run(test_mode=False, enable_audio=True):
    """
    Primary entry point for Sentience.
    
    Initializes the system and runs the continuous perception-inference loop.
    """
    print("\nSentience Cognition Engine initializing...")
    start_time = time.time()
    
    # Initialize system
    try:
        initializer = Initialiser(device_preference="mps")
        device = initializer.device
    except RuntimeError as e:
        logger.critical(f"❌ System initialization failed: {e}")
        sys.exit(1)
    
    # Ensure model is downloaded
    try:
        asset_manager = AssetManager()
        model_path = asset_manager.ensure_model_available()
    except SystemExit:
        # Asset manager handles its own exit on critical download failure
        return
    
    # Set up model engine
    try:
        engine = GemmaEngine(model_path=model_path, device=device)
    except Exception as e:
        logger.critical(f"❌ Failed to load GemmaEngine: {e}")
        sys.exit(1)
    
    # Set up camera with retries for permission handling
    camera = None
    for attempt in range(3):
        try:
            camera = CameraFeed(device=device, test_mode=test_mode)
            if camera.is_opened():
                logger.info("✓ Camera initialized successfully.")
                break
        except RuntimeError as e:
            logger.warning(f"Camera initialization attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # Wait for permissions to propagate
    
    if not camera or not camera.is_opened():
        logger.critical("❌ Failed to initialize camera after multiple attempts.")
        sys.exit(1)
        
    # Set up audio if enabled, with retries for permission handling
    audio_stream = None
    if enable_audio:
        for attempt in range(3):
            try:
                logger.info(f"Initializing audio stream (attempt {attempt + 1})...")
                audio_stream = AudioStream(test_mode=test_mode)
                logger.info("✓ Audio capture initialized")
                break  # Success
            except Exception as e:
                logger.warning(f"⚠️ Audio initialization attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2)  # Wait before retrying
                else:
                    logger.error("❌ Failed to initialize audio after multiple attempts. Continuing without audio.")
                    audio_stream = None
                    enable_audio = False
    
    # Set up output stream with colors
    thought_sink = ThoughtSink(use_colors=True)
    
    # Report cold boot time
    cold_boot_time = time.time() - start_time
    print(f"\nSentience ready! Cold boot time: {cold_boot_time:.2f}s")
    
    # Performance monitoring
    iteration_count = 0
    throughput_start_time = time.time()
    
    # Main loop
    try:
        logger.info("🚀 Starting continuous perception-inference loop...\n")
        
        while True:
            loop_start_time = time.time()
            
            try:
                # 1. Capture inputs (frame and audio)
                frame = camera.get_frame()
                
                # Get audio if available
                audio_tensor = None
                if enable_audio and audio_stream:
                    audio_tensor = audio_stream.get_audio(device=device)
                    
                # 2. Describe scene with multimodal input
                scene_text = engine.describe_scene(
                    image=frame,
                    audio=audio_tensor,
                    audio_sampling_rate=16000 if audio_tensor is not None else None
                )
                
                # 3. Plan action based on scene
                plan_text = engine.plan_action(scene_text)
                
                # 4. Emit thought
                thought_sink.emit(scene_text, plan_text)

            except Exception as e:
                logger.error(f"An error occurred in the main loop: {e}", exc_info=True)
                time.sleep(2) # Cooldown to prevent rapid-fire errors
                continue

            # Performance monitoring
            iteration_count += 1
            
            if iteration_count == 1:
                first_thought_time = time.time() - start_time
                logger.info(f"📊 [Performance] First thought generated in: {first_thought_time:.2f}s")
            
            if iteration_count == 10:
                # Check memory usage after 10 iterations
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / (1024 * 1024)
                logger.info(f"📊 [Performance] Memory usage after 10 iterations: {memory_mb:.2f} MB")
                
            if iteration_count > 0 and iteration_count % 240 == 0:  # Every ~60 seconds
                duration = time.time() - throughput_start_time
                throughput = iteration_count / duration
                logger.info(f"📊 [Performance] Sustained throughput: {throughput:.2f} thoughts/sec over {duration:.0f}s")
            
            # 5. Dynamic Sleep to maintain ~5Hz
            elapsed = time.time() - loop_start_time
            sleep_time = max(0, 0.2 - elapsed)  # Target 200ms per loop
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        logger.info("\n🛑 User requested shutdown. Exiting gracefully...")
    finally:
        # Final performance report
        total_runtime = time.time() - start_time
        logger.info("="*50)
        logger.info("SESSION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total runtime: {total_runtime:.2f}s")
        logger.info(f"Total thoughts generated: {iteration_count}")
        if total_runtime > 1:
            avg_throughput = iteration_count / total_runtime
            logger.info(f"Average throughput: {avg_throughput:.2f} thoughts/sec")
            
        # Clean up resources
        if enable_audio and audio_stream:
            audio_stream.close()
            
        logger.info("Sentience has shut down.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sentience: Multimodal AI Cognition Engine')
    parser.add_argument('--test', action='store_true', help='Run in test mode with synthetic inputs')
    parser.add_argument('--no-audio', action='store_true', help='Disable audio input')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run with parsed arguments
    run(test_mode=args.test, enable_audio=not args.no_audio)
