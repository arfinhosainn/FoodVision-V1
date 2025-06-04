from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""
    APP_NAME: str = "The Nutritionist"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "API for food detection and nutritional analysis"
    
    # API Keys
    GEMINI_API_KEY: Optional[str] = None
    USDA_API_KEY: Optional[str] = None
    
    # Default values
    DEFAULT_PLATE_DIAMETER_MM: float = 250.0
    DEFAULT_NUTRIENT_BASE_AMOUNT: int = 100
    
    # Image processing settings
    MAX_IMAGE_SIZE: int = 800  # Reduced from 1024 for faster processing
    MIN_CONTOUR_AREA: int = 300  # Reduced from 500 to detect smaller items
    ENABLE_ADVANCED_PROCESSING: bool = False  # New flag to control processing level
    PROCESSING_QUALITY: str = "fast"  # New setting: "fast", "balanced", "quality"
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra fields from .env file

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings() 