"""
Configuration management for AI Meeting Monitor.
All settings can be overridden via environment variables.
"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Feature flags
    ENABLE_VIDEO: bool = True  # Set to True if webcam available
    ENABLE_AUDIO: bool = True
    
    # Model settings
    WHISPER_MODEL: str = "base"  # tiny, base, small, medium, large
    
    # Detection thresholds
    HATE_THRESHOLD: float = 0.5
    DANGER_THRESHOLD: float = 0.7
    
    # Audio processing
    MIN_AUDIO_SIZE: int = 1000  # bytes
    AUDIO_BUFFER_CHUNKS: int = 1  # Process each chunk immediately
    
    # Session tracking
    FUSION_WINDOW_SECONDS: int = 30
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            HOST=os.getenv("HOST", cls.HOST),
            PORT=int(os.getenv("PORT", cls.PORT)),
            ENABLE_VIDEO=os.getenv("ENABLE_VIDEO", "true").lower() == "true",
            ENABLE_AUDIO=os.getenv("ENABLE_AUDIO", "true").lower() == "true",
            WHISPER_MODEL=os.getenv("WHISPER_MODEL", cls.WHISPER_MODEL),
            HATE_THRESHOLD=float(os.getenv("HATE_THRESHOLD", cls.HATE_THRESHOLD)),
            DANGER_THRESHOLD=float(os.getenv("DANGER_THRESHOLD", cls.DANGER_THRESHOLD)),
            MIN_AUDIO_SIZE=int(os.getenv("MIN_AUDIO_SIZE", cls.MIN_AUDIO_SIZE)),
            AUDIO_BUFFER_CHUNKS=int(os.getenv("AUDIO_BUFFER_CHUNKS", cls.AUDIO_BUFFER_CHUNKS)),
            FUSION_WINDOW_SECONDS=int(os.getenv("FUSION_WINDOW_SECONDS", cls.FUSION_WINDOW_SECONDS)),
        )


# Global config instance
config = Config.from_env()
