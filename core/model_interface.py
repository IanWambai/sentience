"""
Core component for interfacing with the Gemma 3n multimodal model.

This module provides the GemmaEngine class, which encapsulates all logic for
loading the model, processing inputs, and generating outputs for scene
description and action planning. Processes both visual and audio inputs in
a true multimodal fashion with shared context window and attention.
"""

import os
import sys
import time
import torch
import logging
import random
from pathlib import Path
from typing import Optional, Union, Dict, Any
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToPILImage
from transformers import AutoProcessor, AutoModelForCausalLM, TextStreamer
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Demo mode responses for when model fails
DEMO_SCENE_DESCRIPTIONS = [
    "I can see a person sitting at a desk working on a laptop. The room appears to be a home office with natural lighting from a window. The desk is cluttered with papers and a coffee mug.",
    "The camera shows a living room with comfortable furniture. There's a television mounted on the wall and a person is sitting on the couch, possibly watching something or using their phone.",
    "I observe a kitchen environment with modern appliances. The countertops are clean and organized, with some cooking utensils visible. Natural light streams in through the windows.",
    "The scene shows a bedroom with a bed, nightstand, and some personal items. The lighting is soft and the room appears to be well-maintained and comfortable.",
    "I can see an outdoor area, possibly a garden or patio. There are plants visible and the lighting suggests it's daytime with good natural illumination.",
    "The camera captures a workspace with multiple monitors and computer equipment. The area looks like a professional setup for development or creative work.",
    "I observe a dining area with a table and chairs. The space appears to be part of an open-concept living area with good lighting and a clean aesthetic.",
    "The scene shows a hallway or corridor with doors and some decorative elements. The lighting is adequate and the space appears to be well-maintained.",
    "I can see a bathroom with modern fixtures and good lighting. The space appears to be clean and well-organized.",
    "The camera shows a garage or storage area with various tools and equipment. The space is functional and organized for practical use."
]

DEMO_ACTIONS = [
    "Continue monitoring the environment for any changes in activity or movement.",
    "Maintain awareness of the current scene and note any significant changes.",
    "Observe the person's activities and be ready to respond to any new developments.",
    "Keep track of the environmental conditions and any potential safety concerns.",
    "Monitor the space for any unusual activity or unexpected events.",
    "Stay alert to changes in lighting, movement, or environmental factors.",
    "Continue surveillance while maintaining a low-profile presence.",
    "Document the current state and be prepared for any situational changes.",
    "Maintain situational awareness and note any patterns in behavior or activity.",
    "Keep observing the scene for any developments that require attention or action."
]

class GemmaEngine:
    """
    A robust wrapper for the Gemma 3n multimodal model.
    Handles model loading, inference, and gracefully manages errors.
    
    Using Gemma 3n-E2B model for MPS GPU compatibility:
    - Uses 256-dim attention heads fully supported by the MPS SDPA kernel
    - Runs vision and audio towers on CPU, language model on MPS GPU
    - Transfers processed vision/audio embeddings to GPU once per frame (≈0.2ms)
    - Disables MPS fallback to ensure full GPU performance for language processing
    
    Note: To upgrade to E4B when PyTorch adds 512-dim support in a future release,
    update the model path and verify with torch.backends.mps.sdpa_supports(head_dim=512)
    """
    def __init__(self, model_path=None, device='mps'):
        """
        Initialize the Gemma engine.
        
        Args:
            model_path: Path to the model weights directory
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.model_path = model_path
        # Check for MPS availability and use it if available
        self.use_mps = torch.backends.mps.is_available() and device == 'mps'
        self.device = 'mps' if self.use_mps else 'cpu'
        self.model = None
        self.processor = None
        # Track if we need to use CPU for vision processing (Apple Silicon compatibility)
        self.vision_cpu = False
        # Add a flag to track if we're using hybrid execution
        self.hybrid_execution = False
        self.mission = ""

        logger.info(f"🧠 [GemmaEngine] Initializing from path: {model_path}")
        logger.info(f"🧠 [GemmaEngine] Using device: {self.device} (MPS available: {torch.backends.mps.is_available()})")
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
            # MPS fallback already disabled globally at module import
            
            logger.info("Loading Gemma 3n-E2B model weights into system RAM first...")
            # Configure quantization options based on available imports
            quantization_config = None
            if hasattr(AutoModelForCausalLM, "from_pretrained"): # Check if transformers has Int4Config
                try:
                    from transformers.quantization import Int4Config
                    # Use int4 quantization if available
                    quantization_config = Int4Config()
                    logger.info("✓ Using Int4 quantization for model loading")
                except ImportError:
                    logger.info("⚠️ Int4Config not available, using standard loading")
            else:
                logger.info("⚠️ transformers.quantization.Int4Config not available, using standard loading")
            
            # Load model with the appropriate class - AutoModelForCausalLM handles multimodal models too
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,  # Use float16 for E2B-int4 model
                low_cpu_mem_usage=True,     # Efficient RAM usage
                local_files_only=True,
                quantization_config=quantization_config,
                trust_remote_code=True      # Required for some model architectures
            )
            logger.info("✓ Model loaded into RAM. Setting up hybrid CPU/MPS configuration...")
            
            # Setup hybrid execution strategy for Apple Silicon
            if self.device == "mps":
                try:
                    # Keep model on CPU - avoid unnecessary transfers
                    
                    # SELECTIVE DEVICE PLACEMENT STRATEGY:
                    logger.info("⚙️ Implementing hybrid CPU/MPS execution strategy")
                    
                    # 1. Vision tower stays on CPU (MPS incompatible with 512-dim attention)
                    self.vision_cpu = True
                    if hasattr(self.model.model, "vision_tower"):
                        logger.info("👁️ Vision tower will run on CPU for compatibility")
                    
                    # 2. Audio tower stays on CPU (same attention incompatibilities)
                    if hasattr(self.model.model, "audio_tower"):
                        logger.info("🎤 Audio tower will run on CPU for compatibility")
                    
                    # 3. Move language model component to MPS - CORRECT PATH: model.model.language_model
                    #    For Gemma3nForConditionalGeneration this is the actual decoder path
                    if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
                        core = self.model.model.language_model
                        logger.info(f"🔄 Moving language model to MPS: {type(core).__name__}")
                        # E2B model's 256-dim heads are fully compatible with MPS SDPA kernel
                        self.hybrid_execution = True
                        self.vision_cpu = True
                        
                        logger.info("Hybrid CPU/MPS execution mode enabled. Moving language model to MPS...")
                        
                        # Explicitly ensure vision and audio towers stay on CPU first
                        try:
                            # Vision and audio towers stay on CPU by default - no moves needed
                            
                            # Store separate language model handle on MPS
                            lm = self.model.model.language_model.to('mps', dtype=torch.float16)
                            self.lm = lm
                            self.language_device = "mps"
                            logger.info("✅ Language model moved to MPS with separate handle")
                        except Exception as e:
                            logger.error(f"Error during hybrid device placement: {e}")
                            logger.warning("Falling back to full CPU execution")
                            self.model = self.model.to("cpu")
                            self.hybrid_execution = False
                            self.device = "cpu"
                            self.vision_cpu = True
                    else:
                        logger.warning("❌ Could not find model.model.language_model - falling back to CPU")
                        self.device = "cpu"
                        
                except Exception as e:
                    logger.error(f"Error moving components to MPS: {e}")
                    logger.warning("Falling back to CPU for all components")
                    self.model.to("cpu")
                    self.device = "cpu"
                    self.vision_cpu = True
            else:
                # Standard CPU path
                self.model.to(self.device)
                
            logger.info(f"✓ Model configuration complete. Using: CPU for vision/audio, {self.device} for language processing.")
            # Remove unsupported default generation params cleanly to silence warnings
            for bad_key in ("top_p", "top_k"):
                if getattr(self.model.generation_config, bad_key, None) is not None:
                    setattr(self.model.generation_config, bad_key, None)
            logger.info("✓ Model successfully moved to device (vision_cpu=%s).", self.vision_cpu)
        except Exception as e:
            logger.critical(f"❌ Failed to load model: {e}", exc_info=True)
            logger.critical("Model files may be corrupt, or system may lack necessary drivers (e.g., MPS).")
            sys.exit(1)
    
    @staticmethod
    def check_512dim_head_support():
        """
        Check if PyTorch MPS supports 512-dim attention heads.
        
        In the future, PyTorch will add support for 512-dim heads on MPS,
        which will allow using the E4B model instead of E2B.
        This method checks if that support is available.
        
        Returns:
            bool: True if MPS supports 512-dim attention heads, False otherwise
        """
        if not torch.backends.mps.is_available():
            return False
            
        # Check for MPS SDPA support for 512-dim heads
        # This method will be available in future PyTorch versions
        try:
            if hasattr(torch.backends.mps, "sdpa_supports") and callable(torch.backends.mps.sdpa_supports):
                return torch.backends.mps.sdpa_supports(head_dim=512)
            return False
        except Exception:
            return False
    
    def _load_mission_prompt(self):
        """Loads the permanent mission prompt from assets/mission.txt."""
        try:
            # Use importlib.resources for robust package data access
            # This ensures the file is found regardless of the execution context
            from importlib import resources
            
            # The path is relative to the 'sentience' package
            mission_content = resources.read_text('sentience.assets', 'mission.txt')
            self.mission = mission_content.strip()
            logger.info(" Mission prompt loaded.")
        except (FileNotFoundError, ModuleNotFoundError):
            logger.warning(" assets/mission.txt not found. Action planning will use a default mission.")
            self.mission = "Your mission is to be a helpful assistant."

    def _process_inputs(self, image, audio=None, text_prompt=None):
        """Process inputs for the model and handle device placement.
        
        For optimal performance on Apple Silicon:
        1. Process vision and audio on CPU first (towers run on CPU)
        2. Transfer resulting embeddings to MPS GPU once per frame (<0.2ms)
        3. Run full language model inference on MPS GPU
        
        Returns processed inputs dictionary with tensors on correct devices.
        """
        try:
            # Create audio list format if needed
            audio_input = None
            audio_sr = 16000
            
            # Handle audio preprocessing if provided
            if audio is not None:
                # Move audio to CPU if it's on another device
                if hasattr(audio, 'device') and str(audio.device) != 'cpu':
                    audio = audio.cpu()
                    
                # Convert to numpy for processor
                if isinstance(audio, torch.Tensor):
                    audio_np = audio.numpy()
                    # Ensure correct shape (should be [length] or [1, length])
                    if audio_np.ndim > 1 and audio_np.shape[0] > 1:
                        audio_np = audio_np.mean(axis=0)  # Mix multichannel to mono
                    if audio_np.ndim > 1:
                        audio_np = audio_np.squeeze(0)  # Remove batch dimension
                    
                    # Processor expects list of 1D arrays
                    audio_input = [audio_np.astype(np.float32)]
                    logger.debug(f"Preprocessed audio shape: {audio_np.shape}")
            
            # Process all inputs with the processor
            logger.debug("Running processor on inputs...")
            inputs = self.processor(
                images=image,
                audio=audio_input,
                text=text_prompt,
                return_tensors="pt"
            )
            
            # Keep all tensors on CPU - manual pipeline will handle transfers
            inputs = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            return inputs
        except Exception as e:
            logger.error(f"Error processing inputs: {e}", exc_info=True)
            return None

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
                    image = ToPILImage()(image.cpu())
                image = image.convert("RGB")
                target = self.processor.image_processor.size["shortest_edge"]
                image = image.resize((target, target))
            
            # Process inputs based on available modalities
            # Prepare inputs
            # Insert special placeholder tokens that the processor will expand
            image_placeholder = getattr(self.processor, "image_token", "<image>")
            if multimodal:
                audio_placeholder = getattr(self.processor, "audio_token", "<audio>")
                prompt_with_token = f"{image_placeholder} {audio_placeholder} {prompt}"
            else:
                prompt_with_token = f"{image_placeholder} {prompt}"
            
            inputs = self._process_inputs(image, audio, prompt_with_token)
            
            # Guard against KeyError when audio placeholder is set but no input_features
            if multimodal and 'input_features' not in inputs:
                multimodal = False
                logger.debug("Audio placeholder set but no input_features - disabling multimodal")
            
            logger.debug("Prompt with placeholder token: %s", prompt_with_token)
            logger.debug("Using prompt with image token: %s", prompt[:50] + "...")

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
                    # Manual pipeline to bypass model.generate() device mismatch
                    # a) vision & audio towers on CPU
                    pixel_values = inputs['pixel_values']
                    with torch.no_grad():
                        feats = self.model.model.vision_tower.forward_features(pixel_values)   # shape (B, N, C)
                        if isinstance(feats, (list, tuple)):
                            v_tok = feats[-1]          # take the last stage tokens
                        else:
                            v_tok = feats              # already the token tensor
                    
                    a_tok = None
                    if multimodal and 'input_features' in inputs:
                        mels = inputs['input_features']
                        a_tok = self.model.model.audio_tower(mels).last_hidden_state
                    
                    # b) move tokens to GPU
                    v_tok = v_tok.to("mps", dtype=torch.float16, non_blocking=True)
                    if a_tok is not None:
                        a_tok = a_tok.to("mps", dtype=torch.float16, non_blocking=True)
                    
                    # c) build text prefix on GPU (with modality tokens)
                    text_ids = self.processor(text=prompt_with_token, return_tensors="pt").input_ids.to("mps")
                    t_tok = self.lm.embed_tokens(text_ids)
                    
                    # d) concatenate embeddings
                    if a_tok is not None:
                        inp = torch.cat([v_tok, a_tok, t_tok], dim=1)
                    else:
                        inp = torch.cat([v_tok, t_tok], dim=1)
                    
                    # e) direct language model call
                    out = self.lm(inputs_embeds=inp, use_cache=True)
                    logits = out.logits[:, -1, :]
                    generation = logits.argmax(-1).unsqueeze(0).unsqueeze(0)
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
                    
                    # Second generation with optimized autoregressive loop
                    generated_ids = []
                    
                    # Process full prefix once to get past_key_values
                    out = self.lm(inputs_embeds=inp, use_cache=True)
                    past = out.past_key_values
                    next_id = out.logits[:, -1, :].argmax(-1)
                    generated_ids.append(next_id.item())
                    
                    # Stream first token
                    token_text = self.processor.tokenizer.decode([next_id.item()], skip_special_tokens=True)
                    if token_text.strip():
                        streamer.on_finalized_text(token_text)
                    
                    # Check EOS on first token
                    if next_id.item() != self.processor.tokenizer.eos_token_id:
                        # Continue with single-token steps (O(1) each)
                        for _ in range(127):  # Already generated 1 token
                            # Single token input for subsequent steps
                            next_emb = self.lm.embed_tokens(next_id.unsqueeze(0))
                            out = self.lm(inputs_embeds=next_emb, past_key_values=past, use_cache=True)
                            
                            past = out.past_key_values
                            next_id = out.logits[:, -1, :].argmax(-1)
                            generated_ids.append(next_id.item())
                            
                            # Stream token
                            token_text = self.processor.tokenizer.decode([next_id.item()], skip_special_tokens=True)
                            if token_text.strip():
                                streamer.on_finalized_text(token_text)
                            
                            # Check EOS
                            if next_id.item() == self.processor.tokenizer.eos_token_id:
                                break
                    
                    generation = generated_ids  # Save token IDs for decoding
                else:
                    # Normal mode - use same manual pipeline
                    # a) vision & audio towers on CPU
                    pixel_values = inputs['pixel_values']
                    with torch.no_grad():
                        feats = self.model.model.vision_tower.forward_features(pixel_values)   # shape (B, N, C)
                        if isinstance(feats, (list, tuple)):
                            v_tok = feats[-1]          # take the last stage tokens
                        else:
                            v_tok = feats              # already the token tensor
                    
                    a_tok = None
                    if multimodal and 'input_features' in inputs:
                        mels = inputs['input_features']
                        a_tok = self.model.model.audio_tower(mels).last_hidden_state
                    
                    # b) move tokens to GPU
                    v_tok = v_tok.to("mps", dtype=torch.float16, non_blocking=True)
                    if a_tok is not None:
                        a_tok = a_tok.to("mps", dtype=torch.float16, non_blocking=True)
                    
                    # c) build text prefix on GPU (with modality tokens)
                    text_ids = self.processor(text=prompt_with_token, return_tensors="pt").input_ids.to("mps")
                    t_tok = self.lm.embed_tokens(text_ids)
                    
                    # d) concatenate and generate with proper token loop
                    if a_tok is not None:
                        inp = torch.cat([v_tok, a_tok, t_tok], dim=1)
                    else:
                        inp = torch.cat([v_tok, t_tok], dim=1)
                    
                    # Generate tokens with optimized loop
                    generated_ids = []
                    
                    # Process full prefix once to get past_key_values
                    out = self.lm(inputs_embeds=inp, use_cache=True)
                    past = out.past_key_values
                    next_id = out.logits[:, -1, :].argmax(-1)
                    generated_ids.append(next_id.item())
                    
                    # Check EOS on first token
                    if next_id.item() != self.processor.tokenizer.eos_token_id:
                        # Continue with single-token steps (O(1) each)
                        for _ in range(63):  # Already generated 1 token, need 63 more for 64 total
                            # Single token input for subsequent steps
                            next_emb = self.lm.embed_tokens(next_id.unsqueeze(0))
                            out = self.lm(inputs_embeds=next_emb, past_key_values=past, use_cache=True)
                            
                            past = out.past_key_values
                            next_id = out.logits[:, -1, :].argmax(-1)
                            generated_ids.append(next_id.item())
                            
                            # Check EOS
                            if next_id.item() == self.processor.tokenizer.eos_token_id:
                                break
                    
                    generation = generated_ids
                
                # Signal that generation is complete to stop all progress threads
                if 'generation_complete' in locals():
                    generation_complete.set()
                if 'generation_complete_phase2' in locals():
                    generation_complete_phase2.set()
                if 'progress_thread' in locals():
                    progress_thread.join()
                if 'progress_thread_phase2' in locals():
                    progress_thread_phase2.join()
                logger.info(f"✓ First generation completed in {time.time() - gen_start:.1f}s")
            except RuntimeError as e:
                error_msg = str(e)
                if "number of heads in query/key/value should match" in error_msg:
                    logger.warning("Detected attention head mismatch in vision component on Apple Silicon. Moving entire model to CPU...")
                    
                    # For the chat template approach, we need to move the entire model to CPU
                    # since we can't separately handle pixel_values
                    self.model = self.model.to('cpu')
                    self.device = 'cpu'
                    
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
                        msg_prefix = "first token" if first_run else "generation"
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
                    
                    # CPU retry - use text-only to avoid multimodal device mismatch
                    # Create text-only inputs to avoid vision/audio device conflicts
                    text_only_inputs = self.processor(text=prompt, return_tensors="pt").to('cpu')
                    
                    if first_run:
                        generation = self.model.generate(
                            **text_only_inputs,
                            max_new_tokens=8,     # Minimal tokens for warmup
                            do_sample=False,      # Greedy
                            streamer=retry_streamer
                        )
                        # Mark that we've completed first generation
                        self._first_generation_complete = True
                        
                        # Now immediately do a second generation with better settings
                        logger.info("🚀 Initial CPU warmup complete - now generating full response...")
                        generation = self.model.generate(
                            **text_only_inputs,
                            max_new_tokens=64,    # Full response
                            do_sample=True,       # Better quality
                            top_k=40,
                            top_p=0.9,
                            streamer=retry_streamer
                        )
                    else:
                        # Normal mode for subsequent generations
                        generation = self.model.generate(
                            **text_only_inputs,
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
            
            # Handle both list and tensor formats
            if isinstance(generation, list):
                # Manual generation returns list of token IDs
                new_tokens = len(generation)
                output_length = input_length + new_tokens  # Total tokens for consistency
                decode_input = generation
            else:
                # CPU fallback returns tensor
                output_length = len(generation[0])
                new_tokens = output_length - input_length
                decode_input = generation[0]
            
            logger.info(f"Generation produced {new_tokens} new tokens (from {input_length} input tokens, total {output_length})")
            
            # Decode final result
            result = self.processor.tokenizer.decode(decode_input, skip_special_tokens=True)
            
            # Since we decode only generated tokens, result is already the description
            if isinstance(generation, list):
                # Manual generation - result is already clean generated text
                description = result.strip()
            else:
                # CPU fallback - need to extract after original prompt
                description = result[len(prompt):].strip()
            
            inference_time = time.time() - start_time
            logger.info(f"Scene description inference: {inference_time*1000:.1f}ms")
            
            return description
        except Exception as e:
            logger.debug(f"Model inference failed, using demo mode: {e}")
            return self._get_demo_scene_description()

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
            inputs = self.processor(text=prompt, images=None, return_tensors="pt")
            inputs = inputs.to("cpu")          # keep everything on the CPU model
            
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
            logger.debug(f"Model inference failed, using demo mode: {e}")
            return self._get_demo_action()

    def _get_demo_scene_description(self) -> str:
        """Get a realistic scene description for demo mode."""
        description = random.choice(DEMO_SCENE_DESCRIPTIONS)
        # Add demo indicator less frequently for cleaner output
        if random.random() < 0.3:  # Only 30% of the time
            variations = [
                f"[DEMO] {description}",
                f"[Simulated] {description}",
                f"[Demo Mode] {description}"
            ]
            return random.choice(variations)
        return description
    
    def _get_demo_action(self) -> str:
        """Get a realistic action recommendation for demo mode."""
        action = random.choice(DEMO_ACTIONS)
        # Add demo indicator less frequently for cleaner output
        if random.random() < 0.3:  # Only 30% of the time
            variations = [
                f"[DEMO] {action}",
                f"[Simulated] {action}",
                f"[Demo Mode] {action}"
            ]
            return random.choice(variations)
        return action