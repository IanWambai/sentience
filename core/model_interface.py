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
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
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
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,  # Keep this for efficient loading into RAM
                local_files_only=True
            )
            logger.info("✓ Model loaded into RAM. Moving to MPS device...")
            # Now, move the entire model to the MPS device
            # On Apple Silicon keep whole model on CPU for stability
            if self.device == "mps":
                logger.info("Keeping entire model on CPU to avoid mixed-device deadlock")
                self.model.to("cpu")
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
        # Keep prompt very short - mission is handled elsewhere
        # DO NOT add <image> token manually - let the processor do it
        prompt = "Describe what you see" + (" and hear" if multimodal else "") + " in a detailed sentence."
                
        if not self.model or not self.processor:
            logger.error("Model or processor not loaded. Cannot describe scene.")
            return "[Initialization Error: Model not available]"
            
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
            # Insert single placeholder image token that the processor will expand
            image_placeholder = self.processor.image_token if hasattr(self.processor, "image_token") else "<image>"
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
                
                processor_inputs["audio"] = audio_cpu
                processor_inputs["sampling_rate"] = audio_sampling_rate
                # Update text field to include placeholder
                processor_inputs["text"] = prompt_with_token
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
            logger.debug("Calling model.generate ...")
            try:
                # Generate text with correct Gemma parameters
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=64,  # Start with fewer tokens for faster first response
                    do_sample=False,  # Use greedy decoding
                )
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
                    
                    # Try again with everything on CPU
                    logger.debug("Retrying generate with all inputs on CPU...")
                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=64,  # Start with fewer tokens for faster first response
                        do_sample=False     # Use greedy decoding
                    )
                else:
                    # Re-raise for other errors
                    raise
            result = self.processor.decode(generation[0], skip_special_tokens=True)
            
            # Extract description (remove prompt)
            description = result[len(prompt):].strip()
            
            inference_time = time.time() - start_time
            logger.debug(f"Scene description inference: {inference_time*1000:.1f}ms")
            
            return description
        except Exception as e:
            logger.error(f"Error during scene description: {e}", exc_info=True)
            return "[Inference Error: Unable to describe scene]"

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
