"""
Thought streaming system for Sentience.

Formats inference outputs and streams them to stdout with timestamp prefixes.
"""

import sys
import time
from datetime import datetime


class ThoughtSink:
    """
    Handles formatting and streaming of inference outputs.
    
    Prepends timestamps and ensures proper flushing to stdout.
    """
    
    def __init__(self):
        """Initialize the thought sink."""
        pass
    
    def format_timestamp(self):
        """Generate a timestamp string in HH:MM:SS.mss format."""
        now = datetime.now()
        return now.strftime("%H:%M:%S.%f")[:-3]  # Truncate microseconds to milliseconds
    
    def emit(self, scene_text, plan_text):
        """
        Format and emit a thought to stdout.
        
        Args:
            scene_text (str): Scene description from model
            plan_text (str): Action recommendation from model
        """
        timestamp = self.format_timestamp()
        
        # Format the output as a single line
        thought_line = f"[{timestamp}] SCENE: {scene_text.strip()} | ACTION: {plan_text.strip()}"
        
        # Write to stdout and flush
        sys.stdout.write(thought_line + "\n")
        sys.stdout.flush()
