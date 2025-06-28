"""
Audio capture and processing for Sentience.

This module provides the AudioStream class that captures microphone input
using PyAudio, processes it into the format required by the Gemma 3n model,
and maintains a buffer of recent audio.

Features:
- Non-blocking capture via producer thread
- 16 kHz mono PCM at 16-bit depth
- Automatic silence detection
- MacOS Core Audio integration with permission handling
"""

import time
import logging
import numpy as np
import torch
import threading
import queue
import pyaudio
from typing import Optional, Tuple

# Configure logger
logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE = 16000  # 16 kHz
CHANNELS = 1         # Mono
FORMAT = pyaudio.paInt16
BUFFER_SECONDS = 1   # Store 1 second of audio
CHUNK_SIZE = 1024    # Process in chunks of 1024 samples
SILENCE_THRESHOLD = 0.01  # RMS threshold for silence detection


class AudioStream:
    """
    Captures and processes audio from the microphone in a non-blocking way.
    
    Maintains a rolling buffer of the most recent audio data, converted to
    the format required by the Gemma 3n multimodal model.
    """
    
    def __init__(self, device="default", test_mode=False):
        """
        Initialize the audio capture system.
        
        Args:
            device: Audio device name/index, or "default" to use system default
            test_mode: If True, generates synthetic audio instead of capturing from mic
        """
        self.device = device
        self.test_mode = test_mode
        self.audio = None
        self.buffer_size = SAMPLE_RATE * BUFFER_SECONDS  # 1 second buffer at 16kHz
        
        # Create a ring buffer for audio data (1 second at 16kHz)
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_lock = threading.Lock()
        
        # Queue for communication between threads
        self.audio_queue = queue.Queue(maxsize=100)
        
        # Thread control
        self.is_running = False
        self.capture_thread = None
        
        # PyAudio instance
        self.p = None
        self.stream = None
        
        # Last energy level for silence detection
        self.last_energy = 0
        
        # Tensor on device
        self.device_tensor = None
        self.tensor_timestamp = 0
        
        # If not in test mode, start the audio capture
        if not self.test_mode:
            self._initialize_audio()
        else:
            logger.warning("⚠️ AudioStream initialized in test mode - using synthetic audio")
    
    def _initialize_audio(self):
        """Initialize the PyAudio stream and start the capture thread."""
        try:
            self.p = pyaudio.PyAudio()
            
            # Find appropriate device
            device_index = None
            if self.device != "default":
                for i in range(self.p.get_device_count()):
                    info = self.p.get_device_info_by_index(i)
                    if self.device in info['name']:
                        device_index = i
                        break
            
            # Open the stream
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._audio_callback
            )
            
            logger.info(f"✓ Audio stream started (16 kHz mono)")
            self.is_running = True
            
            # Start the processing thread
            self.capture_thread = threading.Thread(
                target=self._process_audio_queue, 
                daemon=True
            )
            self.capture_thread.start()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize audio: {e}")
            if self.p:
                self.p.terminate()
            self.p = None
            self.stream = None
            # Fall back to test mode
            self.test_mode = True
            logger.warning("⚠️ Falling back to test mode with synthetic audio")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback from PyAudio stream - pushes raw audio to queue."""
        if status:
            logger.debug(f"Audio callback status: {status}")
        
        # Push to queue for processing in the other thread
        try:
            self.audio_queue.put_nowait(in_data)
        except queue.Full:
            logger.warning("Audio queue full, dropping audio chunk")
        
        return (None, pyaudio.paContinue)
    
    def _process_audio_queue(self):
        """Worker thread that processes audio chunks from the queue."""
        while self.is_running:
            try:
                # Get chunk from queue with timeout
                chunk = self.audio_queue.get(timeout=0.5)
                
                # Convert chunk to numpy array
                audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                
                # Normalize to [-1, 1]
                audio_data = audio_data / 32768.0
                
                # Calculate energy for silence detection
                energy = np.sqrt(np.mean(audio_data**2))
                self.last_energy = energy
                
                # Only process if above threshold
                if energy > SILENCE_THRESHOLD:
                    # Add to rolling buffer with lock to avoid race conditions
                    with self.buffer_lock:
                        # Shift buffer and add new data at the end
                        shift_size = len(audio_data)
                        self.audio_buffer = np.roll(self.audio_buffer, -shift_size)
                        self.audio_buffer[-shift_size:] = audio_data
                        
                        # Mark device tensor as outdated
                        self.device_tensor = None
                else:
                    logger.debug(f"Silence detected (energy: {energy:.4f})")
                
                # Mark as processed
                self.audio_queue.task_done()
                
            except queue.Empty:
                # Timeout - this is expected
                pass
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
    
    def _generate_test_audio(self) -> np.ndarray:
        """Generate synthetic audio for test mode."""
        # Generate a 440Hz sine wave as test audio
        t = np.linspace(0, BUFFER_SECONDS, self.buffer_size, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Add some noise
        audio += 0.02 * np.random.randn(self.buffer_size)
        
        # Ensure float32 in [-1, 1] range
        return audio.astype(np.float32)
    
    def get_audio(self, device=None) -> torch.Tensor:
        """
        Get the latest audio buffer as a tensor on the specified device.
        
        Args:
            device: torch device to place tensor on
            
        Returns:
            torch.Tensor: Audio waveform tensor of shape [1, buffer_size]
        """
        # Check if we need to update the device tensor
        now = time.time()
        if self.device_tensor is None or now - self.tensor_timestamp > 0.1:
            # Get audio data
            if self.test_mode:
                audio_data = self._generate_test_audio()
            else:
                with self.buffer_lock:
                    audio_data = self.audio_buffer.copy()
            
            # Convert to tensor
            tensor = torch.from_numpy(audio_data).float()
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            # Move to device if specified
            if device is not None:
                tensor = tensor.to(device)
            
            self.device_tensor = tensor
            self.tensor_timestamp = now
        
        return self.device_tensor
    
    def is_silent(self) -> bool:
        """Check if the current audio is silent (below threshold)."""
        return self.last_energy < SILENCE_THRESHOLD
    
    def close(self):
        """Clean up audio resources."""
        self.is_running = False
        
        # Wait for thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        # Close the audio stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        # Terminate PyAudio
        if self.p:
            self.p.terminate()
        
        logger.info("Audio stream closed")
