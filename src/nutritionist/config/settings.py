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
    NUTRITIONIX_APP_ID: Optional[str] = None
    NUTRITIONIX_APP_KEY: Optional[str] = None
    
    # Default values
    DEFAULT_PLATE_DIAMETER_MM: float = 250.0
    DEFAULT_NUTRIENT_BASE_AMOUNT: int = 100
    
    # Image processing settings
    MAX_IMAGE_SIZE: int = 512  # Further reduced for faster inference
    MIN_CONTOUR_AREA: int = 300  # Reduced from 500 to detect smaller items
    ENABLE_ADVANCED_PROCESSING: bool = False  # New flag to control processing level
    PROCESSING_QUALITY: str = "fast"  # New setting: "fast", "balanced", "quality"

    # YOLO integration
    USE_YOLO_DETECTION: bool = False  # Set true to enable YOLOv8 object detection
    YOLO_MODEL: str = "yolov8n.pt"    # Ultraytics model name or path

    # Optional ONNX/TensorRT acceleration
    USE_ONNX_ACCELERATION: bool = False  # If true, load/export models in ONNX/TensorRT format for faster inference
    YOLO_ONNX_MODEL: str = "yolov8n.onnx"  # Path to optimized YOLO model (exported with Ultralytics)

    # Mask R-CNN integration
    USE_MASK_RCNN: bool = False
    MASK_RCNN_MIN_SCORE: float = 0.5
    MASK_RCNN_ONNX_MODEL: str = "mask_rcnn_resnet50_fpn.onnx"  # Path to optimized Mask R-CNN ONNX model

    # MiDaS depth-estimation integration
    USE_MIDAS_DEPTH: bool = False  # Enable monocular depth estimation to convert area to volume
    MIDAS_MODEL_NAME: str = "MiDaS_small"  # Options: "MiDaS_small", "DPT_Large", "DPT_Hybrid" etc.

    # Redis cache
    REDIS_URL: str = "redis://localhost:6379/0"
    
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