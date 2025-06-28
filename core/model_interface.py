"""
Model interface for Gemma 3n multimodal LLM.

This module wraps the Transformers processor and model for Gemma 3n,
handling inference and maintaining the KV cache across forward passes.
"""

import os
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLLM
from transformers.quantization_utils import Int4Config


class GemmaEngine:
    """
    Interface to the Gemma 3n multimodal model.
    
    References:
    - Transformers multimodal docs: https://huggingface.co/docs/transformers/en/index
    - Gemma 3n E2B model card: https://huggingface.co/google/gemma-3n-E2B
    """
    
    def __init__(self, model_path, device="mps"):
        """
        Initialize the Gemma engine with int4 quantization.
        
        Args:
            model_path (str): Path to local Gemma 3n checkpoint
            device (str): Compute device, typically "mps" on Apple Silicon
        """
        self.device = device
        self.model_path = model_path
        self.past_key_values = None
        
        print(f"Loading Gemma 3n processor from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            local_files_only=True
        )
        
        print(f"Loading Gemma 3n model (int4) from {model_path} to {device}...")
        self.model = AutoModelForMultimodalLLM.from_pretrained(
            model_path,
            quantization_config=Int4Config(),
            local_files_only=True
        )
        self.model.to(device)
        self.model.eval()
        print("Model loaded successfully.")

        # Load mission prompt for action planning
        self.mission_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "mission.txt")
        with open(self.mission_path, "r") as f:
            self.mission = f.read().strip()
    
    def describe_scene(self, frame_tensor):
        """
        Generate a scene description from a camera frame.
        
        Args:
            frame_tensor (torch.Tensor): Image tensor from CameraFeed
            
        Returns:
            str: Scene description (max 250 chars)
        """
        with torch.no_grad():
            # Prepare image + prompt for scene description
            prompt = "Describe what you see in this image."
            inputs = self.processor(
                text=prompt,
                images=frame_tensor.unsqueeze(0),  # Add batch dimension
                return_tensors="pt"
            ).to(self.device)
            
            # Generate scene description
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,  # About 250 chars
                do_sample=False,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
            
            # Decode the generated text
            scene_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the output
            scene_text = scene_text.replace(prompt, "").strip()
            
            # Ensure it doesn't exceed 250 chars
            if len(scene_text) > 250:
                scene_text = scene_text[:247] + "..."
                
            return scene_text
    
    def plan_action(self, scene_text):
        """
        Generate a goal-conditioned recommendation based on scene description.
        
        Args:
            scene_text (str): Scene description from describe_scene
            
        Returns:
            str: Action recommendation (max 150 chars)
        """
        with torch.no_grad():
            # Combine mission and scene for context
            prompt = f"{self.mission}\n\nScene: {scene_text}\n\nRecommendation:"
            
            # Prepare text-only input
            inputs = self.processor(
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate recommendation
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,  # About 150 chars
                do_sample=False,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
            
            # Decode the generated text
            plan_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            # Extract just the recommendation part
            plan_text = plan_text.replace(prompt, "").strip()
            
            # Ensure it doesn't exceed 150 chars
            if len(plan_text) > 150:
                plan_text = plan_text[:147] + "..."
                
            return plan_text
