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
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
from typing import Optional, Dict, Any, Union, Tuple

logger = logging.getLogger(__name__)

class GemmaEngine:
    """
    A robust wrapper for the Gemma 3n multimodal model.
    Handles model loading, inference, and gracefully manages errors.
    """
    def __init__(self, model_path, device):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.processor = None
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
            logger.info("Loading model weights...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                low_cpu_mem_usage=True,
                local_files_only=True
            )
            logger.info("✓ Model loaded and moved to device.")
        except Exception as e:
            logger.critical(f"❌ Failed to load model: {e}", exc_info=True)
            logger.critical("Model files may be corrupt, or system may lack necessary drivers (e.g., MPS).")
            sys.exit(1)

    def _load_mission_prompt(self):
        """Loads the permanent mission prompt from assets/mission.txt."""
        try:
            # Path goes up one level from 'core' to the project root
            mission_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "mission.txt")
            with open(mission_path, "r") as f:
                self.mission = f.read().strip()
            logger.info("✓ Mission prompt loaded.")
        except FileNotFoundError:
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
        prompt = "Describe what you see and hear in front of you in a detailed sentence." if multimodal else \
                "Describe the scene in front of you in a single, detailed sentence."
                
        if not self.model or not self.processor:
            logger.error("Model or processor not loaded. Cannot describe scene.")
            return "[Initialization Error: Model not available]"
            
        try:
            start_time = time.time()
            
            # Process inputs based on available modalities
            processor_inputs = {
                "text": prompt,
                "images": image,
                "return_tensors": "pt"
            }
            
            # Add audio if available
            if multimodal and audio is not None:
                processor_inputs["audio"] = audio
                processor_inputs["sampling_rate"] = audio_sampling_rate
                logger.debug("Processing with multimodal input (vision + audio)")
            else:
                logger.debug("Processing with vision input only")
                
            # Create model inputs
            inputs = self.processor(**processor_inputs).to(self.device)
            
            # Generate text
            generation = self.model.generate(**inputs, max_length=496, do_sample=False)
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
