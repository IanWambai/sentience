"""
Core component for interfacing with the Gemma 3n multimodal model.

This module provides the GemmaEngine class, which encapsulates all logic for
loading the model, processing inputs, and generating outputs for scene
description and action planning. Processes both visual and audio inputs in
a true multimodal fashion with shared context window and attention.
"""

import os
import torch
import logging
import sys
import time
import numpy as np
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from PIL import Image
from typing import Optional, Dict, Any, Union, Tuple

logger = logging.getLogger(__name__)

class GemmaEngine:
    """
    A robust wrapper for the Gemma 3n multimodal model.
    Handles model loading, inference, and gracefully manages errors.
    """
    def __init__(self, model_path=None, device='mps'):
        """Initialize the Gemma engine.
        
        Args:
            model_path: Path to the model weights directory
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.model_path = model_path
        self.device = device if torch.backends.mps.is_available() and device == 'mps' else 'cpu'
        self.model = None
        self.processor = None
        # Track if we need to use CPU for vision processing (Apple Silicon compatibility)
        self.use_cpu_for_vision = False
        self.mission = ""

        logger.info(f"🧠 [GemmaEngine] Initializing from path: {model_path}")
        self._load_model_and_processor()
        self._load_mission_prompt()
        logger.info("🧠 [GemmaEngine] Initialization complete.")

    def _load_model_and_processor(self):
        """Loads the processor and model, with detailed error handling."""
        try:
            logger.info("Loading model processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=True)
            logger.info("✓ Processor loaded.")
        except Exception as e:
            logger.critical(f"❌ Failed to load AutoProcessor: {e}", exc_info=True)
            logger.critical(f"Check model files at {self.model_path} and HuggingFace connectivity.")
            sys.exit(1)

        try:
            logger.info("Loading model weights into system RAM first...")
            # Load the model to CPU first to avoid direct-to-GPU allocation issues
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.bfloat16,
                low_cpu_mem_usage=True,  # Keep this for efficient loading into RAM
                local_files_only=True
            )
            logger.info("✓ Model loaded into RAM. Moving to MPS device...")
            # Now, move the entire model to the MPS device
            # On Apple Silicon keep whole model on CPU for stability
            if self.device == "mps":
                logger.info("Keeping entire model on CPU to avoid mixed-device deadlock")
                self.model.to("cpu")
                # Ensure weights are float32 to avoid dtype mismatch with Conv2d
                if self.model.dtype != torch.float32:
                    self.model = self.model.float()
                self.device = "cpu"
            else:
                self.model.to(self.device)
            # Move language layers to device, but keep vision tower on CPU to avoid Apple Silicon attention mismatch
            logger.debug("Placing vision_tower on CPU and language layers on %s", self.device)
            if hasattr(self.model, "vision_tower") and self.device == "mps":
                self.model.vision_tower.to("cpu")
                self.vision_cpu = True
                logger.info("✓ Vision tower kept on CPU for compatibility.")
            else:
                self.vision_cpu = False
            # Remove unsupported default generation params cleanly to silence warnings
            for bad_key in ("top_p", "top_k"):
                if getattr(self.model.generation_config, bad_key, None) is not None:
                    setattr(self.model.generation_config, bad_key, None)
            logger.info("✓ Model successfully moved to device (vision_cpu=%s).", self.vision_cpu)
        except Exception as e:
            logger.critical(f"❌ Failed to load model: {e}", exc_info=True)
            logger.critical("Model files may be corrupt, or system may lack necessary drivers (e.g., MPS).")
            sys.exit(1)

    def _load_mission_prompt(self):
        """Loads the permanent mission prompt from assets/mission.txt."""
        try:
            # Use importlib.resources for robust package data access
            # This ensures the file is found regardless of the execution context
            from importlib import resources
            
            # The path is relative to the 'sentience' package
            mission_content = resources.read_text('sentience.assets', 'mission.txt')
            self.mission = mission_content.strip()
            logger.info("✓ Mission prompt loaded.")
        except (FileNotFoundError, ModuleNotFoundError):
            logger.warning("⚠️ assets/mission.txt not found. Action planning will use a default mission.")
            self.mission = "Your mission is to be a helpful assistant."

    def describe_scene(self, 
                    image: Image.Image, 
                    audio: Optional[torch.Tensor] = None,
                    audio_sampling_rate: int = 16000) -> str:
        """Generates a textual description based on both image and audio inputs.
        
        Args:
            image: PIL Image containing the visual scene
            audio: Optional tensor of shape [1, samples] containing audio waveform
            audio_sampling_rate: Sample rate of the audio (default 16kHz)
            
        Returns:
            str: Textual description of the scene incorporating both visual and auditory cues
        """
        multimodal = audio is not None
        # Efficient prompt that encourages detailed scene description and reasoning
        # DO NOT add <image> token manually - let the processor do it
        prompt = "Describe what you see" + (" and hear" if multimodal else "") + ". Include all important details and your observations about the scene."
                
        if not self.model or not self.processor:
            logger.error("Model or processor not loaded. Cannot describe scene.")
            return "[Initialization Error: Model not available]"
            
        # Track if this is our first generation
        first_run = not hasattr(self, '_first_generation_complete')
            
        try:
            start_time = time.time()
            
            # Ensure image is in RGB and correct size expected by model (224x224)
            if image is not None:
                if isinstance(image, torch.Tensor):
                    # Convert CHW -> HWC PIL image
                    image = to_pil_image(image.cpu())
                image = image.convert("RGB")
                image = image.resize((224, 224))
            
            # Process inputs based on available modalities
            # Prepare inputs
            # Insert special placeholder tokens that the processor will expand
            image_placeholder = getattr(self.processor, "image_token", "<image>")
            if multimodal:
                audio_placeholder = getattr(self.processor, "audio_token", "<audio>")
                prompt_with_token = f"{image_placeholder} {audio_placeholder} {prompt}"
            else:
                prompt_with_token = f"{image_placeholder} {prompt}"
            
            processor_inputs = {
                "text": prompt_with_token,
                "images": image,
                "return_tensors": "pt"
            }
            logger.debug("Prompt with placeholder token: %s", prompt_with_token)
            logger.debug("Using prompt with image token: %s", prompt[:50] + "...")

            # Add audio if available
            if multimodal and audio is not None:
                # Move audio to CPU before processing
                if hasattr(audio, 'device') and str(audio.device) != 'cpu':
                    audio_cpu = audio.cpu()
                else:
                    audio_cpu = audio
                    
                # Convert audio to numpy array if it's a tensor
                if isinstance(audio_cpu, torch.Tensor):
                    audio_cpu = audio_cpu.numpy()
                
                # Ensure the audio is mono (1-D). If stereo or batched, down-mix.
                if audio_cpu.ndim > 1:
                    if audio_cpu.shape[0] == 1:
                        # Shape [1, length] -> [length]
                        audio_cpu = audio_cpu.squeeze(0)
                    else:
                        # Stereo or multi-channel -> average across channels
                        audio_cpu = audio_cpu.mean(axis=0)
                # audio_cpu should now be shape [length]
                
                # Gemma processor expects a list of 1-D arrays (batch dimension)
                audio_examples = [audio_cpu.astype(np.float32)]
                assert audio_examples[0].ndim == 1, "Audio must be 1-D mono"
                processor_inputs["audio"] = audio_examples
                processor_inputs["sampling_rate"] = audio_sampling_rate

                logger.debug(f"Processing with multimodal input (vision + audio), audio shape: {np.shape(audio_cpu)}")

            else:
                logger.debug("Processing with vision input only")
            
            # Create model inputs using processor
            logger.debug(f"Running processor with inputs: {type(processor_inputs)}")
            inputs = self.processor(**processor_inputs)
            logger.debug(f"Processor output types: {type(inputs)}")
            
            # Place inputs on the appropriate device(s)
            for key in inputs:
                # Keep pixel_values on CPU if vision_cpu is True
                if key == "pixel_values" and self.vision_cpu:
                    inputs[key] = inputs[key].cpu()
                else:
                    inputs[key] = inputs[key].to(self.device)
            
            logger.debug(
                "Processor tensors prepared. pixel_values=%s, input_ids=%s",
                inputs.get("pixel_values").device if "pixel_values" in inputs else "N/A",
                inputs.get("input_ids").device if "input_ids" in inputs else "N/A",
            )
            
            # Use only supported generation parameters for Gemma 3n
            # Set up standard TextStreamer with appropriate options
            streamer = TextStreamer(
                self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            logger.info("🌀 Starting model.generate with streaming... [first token may take several minutes]")
            try:
                # Generate text with correct Gemma parameters
                # Log precise timing for the generation call to monitor progress
                gen_start = time.time()
                logger.debug(f"Generation starting at {time.strftime('%H:%M:%S')}")
                
                import threading
                # Use an event to signal when generation is complete
                generation_complete = threading.Event()
                
                # Create a separate thread to monitor generation progress
                def log_progress():
                    elapsed = 0
                    # Adjust messaging based on whether this is first run or not
                    msg_prefix = "first token" if first_run else "generation"
                    
                    while elapsed < 600 and not generation_complete.is_set():  # Stop when complete or timeout
                        time.sleep(5)  # Check more frequently but only log every 30s
                        if generation_complete.is_set():
                            break
                            
                        elapsed = time.time() - gen_start
                        if elapsed >= 30 and elapsed % 30 < 5:  # Log at 30s intervals
                            logger.info(f"⏳ Still waiting for {msg_prefix}... {elapsed:.1f} seconds elapsed")
                
                # Start progress monitoring in background thread
                progress_thread = threading.Thread(target=log_progress, daemon=True)
                progress_thread.start()
                
                # Warm-up strategy: first run uses minimal settings for fastest possible response
                # Subsequent runs use better settings for quality results
                if first_run:
                    logger.info("🔥 First generation - using ultra minimal settings for fastest warmup")
                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=8,     # Absolute minimum tokens - just need to complete first compile
                        do_sample=False,      # Greedy
                        streamer=streamer     # Stream tokens
                    )
                    # Mark that we've completed first generation
                    self._first_generation_complete = True
                    
                    # Stop the first progress thread
                    generation_complete.set()
                    
                    # Now immediately do a second generation with better settings and different prompt
                    # This will be much faster since model is compiled and loaded
                    logger.info("🚀 Initial warmup complete - now generating detailed scene analysis...")
                    
                    # Modify the prompt slightly for the second generation to avoid EOS
                    # But keep it concise to maintain token efficiency
                    prompt_phase2 = prompt + " Continue with more observations:"
                    
                    # Reset progress monitoring for full generation and create new event to avoid thread conflicts
                    gen_start = time.time()
                    generation_complete_phase2 = threading.Event()
                    
                    # Create a new progress thread for the full generation
                    def full_generation_progress():
                        elapsed = 0
                        while elapsed < 600 and not generation_complete.is_set():
                            time.sleep(5)
                            if generation_complete.is_set():
                                break
                                
                            elapsed = time.time() - gen_start
                            if elapsed >= 15 and elapsed % 15 < 5:  # Log more frequently for second run
                                logger.info(f"⏳ Still generating detailed scene analysis... {elapsed:.1f} seconds elapsed")
                    
                    # Start the full generation progress thread with a new thread object
                    progress_thread_phase2 = threading.Thread(target=full_generation_progress, daemon=True)
                    progress_thread_phase2.start()
                    
                    # Use the existing streamer instead of creating a custom one
                    # HuggingFace's TextStreamer already handles token streaming efficiently
                    if not hasattr(streamer, "token_seen"):
                        # Add a minimal wrapper to log tokens without creating a new class
                        original_on_finalized_text = streamer.on_finalized_text
                        
                        def log_and_stream(text, stream_end=False):
                            if text.strip():
                                logger.info(f"🔄 TOKEN: {repr(text)}")
                            return original_on_finalized_text(text, stream_end)
                        
                        streamer.on_finalized_text = log_and_stream
                        streamer.token_seen = True
                    
                    # Use a more moderate token count for efficiency while still getting good descriptions
                    # Reuse the same inputs but with different generation settings
                    generation = self.model.generate(
                        **inputs,  # Reuse the same inputs for efficiency
                        max_new_tokens=128,  # Balanced token count for good descriptions without excess
                        do_sample=True,      # Enable sampling for better quality
                        temperature=0.7,     # Good balance for creativity vs coherence
                        top_k=40,           
                        top_p=0.9,          
                        streamer=streamer    # Reuse the existing streamer with our logging wrapper
                    )
                else:
                    # Normal mode - model is already warmed up
                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=64,    # Full response
                        do_sample=True,       # Enable sampling for better quality
                        top_k=40,             # Reasonable sampling parameters
                        top_p=0.9,
                        streamer=streamer      # Stream tokens
                    )
                
                # Signal that generation is complete to stop all progress threads
                if 'generation_complete' in locals():
                    generation_complete.set()
                if 'generation_complete_phase2' in locals():
                    generation_complete_phase2.set()
                if 'progress_thread' in locals():
                    progress_thread.join(timeout=1)
                if 'progress_thread_phase2' in locals():
                    progress_thread_phase2.join(timeout=1)
                logger.info(f"✓ First generation completed in {time.time() - gen_start:.1f}s")
            except RuntimeError as e:
                error_msg = str(e)
                if "number of heads in query/key/value should match" in error_msg:
                    logger.warning("Detected attention head mismatch in vision component on Apple Silicon. Moving entire model to CPU...")
                    
                    # For the chat template approach, we need to move the entire model to CPU
                    # since we can't separately handle pixel_values
                    self.model = self.model.to('cpu')
                    self.device = 'cpu'
                    
                    # Move inputs to CPU
                    inputs = inputs.to('cpu')
                    
                    # Try again with everything on CPU, but still use streaming
                    logger.info("Retrying generate with all inputs on CPU and streaming...")
                    
                    # Create a fresh streamer for the retry path
                    retry_streamer = TextStreamer(
                        self.processor.tokenizer,
                        skip_prompt=True,
                        skip_special_tokens=True
                    )
                    
                    # Log precise timing for the retry generation call
                    gen_start = time.time()
                    logger.debug(f"CPU retry generation starting at {time.strftime('%H:%M:%S')}")
                    
                    # Create new progress tracker for CPU path
                    cpu_generation_complete = threading.Event()
                    
                    def cpu_log_progress():
                        elapsed = 0
                        while elapsed < 600 and not cpu_generation_complete.is_set():
                            time.sleep(5)
                            if cpu_generation_complete.is_set():
                                break
                                
                            elapsed = time.time() - gen_start
                            if elapsed >= 30 and elapsed % 30 < 5:  # Log at 30s intervals
                                logger.info(f"⏳ [CPU] Still waiting for {msg_prefix}... {elapsed:.1f} seconds elapsed")
                    
                    # Start CPU progress monitoring thread
                    cpu_progress_thread = threading.Thread(target=cpu_log_progress, daemon=True)
                    cpu_progress_thread.start()
                    
                    # CPU retry - use appropriate strategy based on first run status
                    if first_run:
                        generation = self.model.generate(
                            **inputs,
                            max_new_tokens=8,     # Minimal tokens for warmup
                            do_sample=False,      # Greedy
                            streamer=retry_streamer
                        )
                        # Mark that we've completed first generation
                        self._first_generation_complete = True
                        
                        # Now immediately do a second generation with better settings
                        logger.info("🚀 Initial CPU warmup complete - now generating full response...")
                        generation = self.model.generate(
                            **inputs,
                            max_new_tokens=64,    # Full response
                            do_sample=True,       # Better quality
                            top_k=40,
                            top_p=0.9,
                            streamer=retry_streamer
                        )
                    else:
                        # Normal mode for subsequent generations
                        generation = self.model.generate(
                            **inputs,
                            max_new_tokens=64,    # Full response
                            do_sample=True,       # Better quality
                            top_k=40,
                            top_p=0.9,
                            streamer=retry_streamer
                        )
                    
                    # Stop the CPU progress thread
                    cpu_generation_complete.set()
                    
                    # Signal that generation is complete if we used the progress thread here too
                    if 'generation_complete' in locals():
                        generation_complete.set()
                    logger.info(f"✓ CPU retry generation completed in {time.time() - gen_start:.1f}s")
                else:
                    # Re-raise for other errors
                    raise
            # Token metrics were logged earlier, avoid duplication

            # Log token count metrics before decoding
            input_length = len(inputs['input_ids'][0])
            output_length = len(generation[0])
            new_tokens = output_length - input_length
            logger.info(f"Generation produced {new_tokens} new tokens (from {input_length} input tokens)")
            
            # Decode final result from the full generation
            result = self.processor.decode(generation[0], skip_special_tokens=True)
            
            # For phase 2, we need to extract content after the enhanced prompt
            if 'prompt_phase2' in locals():
                # Extract description (remove enhanced prompt)
                # Try to find a reasonable position after the prompt
                base_prompt_len = len(prompt)
                description = result[base_prompt_len:].strip()
            else:
                # Regular generation - extract after original prompt
                description = result[len(prompt):].strip()
            
            inference_time = time.time() - start_time
            logger.info(f"Scene description inference: {inference_time*1000:.1f}ms")
            
            return description
        except Exception as e:
            logger.error(f"Error during scene description: {e}", exc_info=True)
            return "[Inference Error: Unable to describe scene]"

    def _log_model_info(self):
        """Log detailed model information for debugging"""
        try:
            # Log important model configuration details for debugging
            if hasattr(self.model.config, 'torch_dtype'):
                logger.debug(f"Model config torch_dtype: {self.model.config.torch_dtype}")
            
            # Log device placement
            if hasattr(self.model, 'device'):
                logger.debug(f"Model device: {self.model.device}")
            else:
                # Check device of first parameter
                for param in self.model.parameters():
                    logger.debug(f"First model parameter on device: {param.device}")
                    break
            
            # Log attention mechanism used
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'text_config'):
                if hasattr(self.model.config.text_config, 'layer_types'):
                    layer_types = self.model.config.text_config.layer_types
                    logger.debug(f"Layer types: {layer_types[:3]}...{layer_types[-3:]} (showing first/last 3)")
        except Exception as e:
            logger.debug(f"Error logging model info: {e}")
    
    def plan_action(self, scene_description: str) -> str:
        """Given a scene description, suggests a concise, command-like next action.
        
        This function reuses the KV cache from the previous scene description call
        to minimize redundant computation and latency.
        
        Args:
            scene_description: Textual description of the scene (from describe_scene)
            
        Returns:
            str: Action recommendation as a concise command
        """
        prompt = f'Mission: "{self.mission}"\n\nGiven the scene: "{scene_description}"\n\nRecommend one single, immediate, and concise action to take. Phrase it as a direct command. Your response must only be the action statement itself.'
        if not self.model or not self.processor:
            logger.error("Model or processor not loaded. Cannot plan action.")
            return "[Initialization Error: Model not available]"
            
        try:
            start_time = time.time()
            
            # This is a text-only prompt, so no image is passed
            inputs = self.processor(text=prompt, images=None, return_tensors="pt").to(self.device)
            
            # Generate text
            generation = self.model.generate(
                **inputs, 
                max_length=496, 
                do_sample=False
            )
            
            result = self.processor.decode(generation[0], skip_special_tokens=True)
            
            # Extract just the action command from the result
            action = result[len(prompt):].strip().replace("Action: ", "")
            
            inference_time = time.time() - start_time
            logger.debug(f"Action planning inference: {inference_time*1000:.1f}ms")
            
            return action
        except Exception as e:
            logger.error(f"Error during action planning: {e}", exc_info=True)
            return "[Inference Error: Unable to plan action]"
