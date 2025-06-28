"""
Thought streaming system for Sentience.

Formats inference outputs and streams them to stdout with timestamp prefixes.
Provides robust error handling and configurable output formatting.
"""

import sys
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any

# Configure module logger
logger = logging.getLogger(__name__)


class ThoughtSink:
    """
    Handles formatting and streaming of inference outputs.
    
    Prepends timestamps and ensures proper flushing to stdout.
    Supports configurable formatting and error handling.
    """
    
    def __init__(self, 
                 use_colors: bool = True, 
                 verbose: bool = False):
        """
        Initialize the thought sink.
        
        Args:
            use_colors (bool): Whether to use ANSI colors in terminal output
            verbose (bool): Whether to include additional details in output
        """
        self.use_colors = use_colors and sys.stdout.isatty()  # Only use colors if terminal supports it
        self.verbose = verbose
        self.thought_count = 0
        self.last_emit_time = time.time()
        
        # ANSI color codes
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'timestamp': '\033[90m',  # Dark gray
            'scene': '\033[36m',      # Cyan
            'action': '\033[33m',     # Yellow
            'error': '\033[31m'       # Red
        }
        
        logger.info("💭 ThoughtSink initialized")
    
    def format_timestamp(self) -> str:
        """Generate a timestamp string in HH:MM:SS.mss format."""
        now = datetime.now()
        return now.strftime("%H:%M:%S.%f")[:-3]  # Truncate microseconds to milliseconds
    
    def _colorize(self, text: str, color_key: str) -> str:
        """Apply ANSI color to text if colors are enabled."""
        if not self.use_colors:
            return text
        return f"{self.colors[color_key]}{text}{self.colors['reset']}"
    
    def _format_thought(self, 
                        timestamp: str, 
                        scene_text: str, 
                        plan_text: str, 
                        error: Optional[str] = None) -> str:
        """Format the complete thought line with optional colors."""
        if error:
            return self._colorize(f"[{timestamp}] ERROR: {error}", 'error')
            
        ts = self._colorize(f"[{timestamp}]", 'timestamp')
        scene_label = self._colorize("SCENE:", 'bold')
        scene = self._colorize(scene_text.strip(), 'scene')
        action_label = self._colorize("ACTION:", 'bold')
        action = self._colorize(plan_text.strip(), 'action')
        
        # Add thought counter if verbose
        counter = f"#{self.thought_count} " if self.verbose else ""
        
        return f"{ts} {counter}{scene_label} {scene} | {action_label} {action}"
    
    def emit(self, scene_text: str, plan_text: str) -> bool:
        """
        Format and emit a thought to stdout.
        
        Args:
            scene_text (str): Scene description from model
            plan_text (str): Action recommendation from model
            
        Returns:
            bool: True if emission was successful, False otherwise
        """
        try:
            # Calculate throughput stats
            now = time.time()
            interval = now - self.last_emit_time
            self.last_emit_time = now
            
            # Safety checks for inputs
            if not isinstance(scene_text, str):
                scene_text = str(scene_text) if scene_text is not None else "[Missing scene description]"
                
            if not isinstance(plan_text, str):
                plan_text = str(plan_text) if plan_text is not None else "[Missing action plan]"
            
            timestamp = self.format_timestamp()
            self.thought_count += 1
            
            # Format the output
            thought_line = self._format_thought(timestamp, scene_text, plan_text)
            
            # Write to stdout and flush
            sys.stdout.write(thought_line + "\n")
            sys.stdout.flush()
            
            # Log throughput stats if verbose
            if self.verbose and self.thought_count > 1:
                logger.debug(f"Thought latency: {interval:.3f}s ({1/interval:.2f} Hz)")
            
            return True
            
        except Exception as e:
            # Log the error but don't crash
            error_msg = f"Failed to emit thought: {e}"
            logger.error(error_msg)
            
            try:
                # Try to emit an error message instead
                error_line = self._format_thought(self.format_timestamp(), "", "", error=error_msg)
                sys.stdout.write(error_line + "\n")
                sys.stdout.flush()
            except:
                # Last resort fallback
                pass
                
            return False
