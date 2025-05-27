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
    MAX_IMAGE_SIZE: int = 1024  # Maximum dimension for image processing
    MIN_CONTOUR_AREA: int = 500  # Minimum area for food detection
    
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