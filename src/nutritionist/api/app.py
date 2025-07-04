from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from typing import Optional, Any, Dict, cast
import asyncio
from functools import lru_cache
import hashlib
from numbers import Number
import json

from ..config.settings import Settings, get_settings
from ..services.image_processor import ImageProcessor
from ..services.enhanced_ai_service import EnhancedAIService as AIService
# Nutrition sources
from ..services.multi_source_nutrition import MultiSourceNutritionService
from ..models.food import AnalysisResponse

# Create FastAPI app
app = FastAPI(
    title="The Nutritionist API",
    description="API for food detection and nutritional analysis",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional Redis cache (graceful fallback if module or server missing)
try:
    import redis.asyncio as redis_async  # type: ignore

    class _RedisCacheWrapper:
        def __init__(self, url: str):
            try:
                self.client = redis_async.from_url(url, encoding=None, decode_responses=True)
            except Exception as conn_err:
                print(f"[WARN] Redis connection failed: {conn_err}. Using no-op cache.")
                self.client = None

        async def get(self, key: str):
            if self.client:
                try:
                    return await self.client.get(key)
                except Exception as e:
                    print(f"[WARN] Redis GET failed: {e}")
            return None

        async def set(self, key: str, value: str, ex: int):
            if self.client:
                try:
                    await self.client.set(key, value, ex=ex)
                except Exception as e:
                    print(f"[WARN] Redis SET failed: {e}")

    _HAS_REDIS = True
except ModuleNotFoundError:
    # Redis package not installed – fall back to dummy async cache
    print("[INFO] redis-py not installed – caching disabled.")

    class _RedisCacheWrapper:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        async def get(self, *_args, **_kwargs):
            return None

        async def set(self, *_args, **_kwargs):
            return None

    _HAS_REDIS = False

# Redis cache setup (may be no-op wrapper)
_settings = get_settings()
redis_client = _RedisCacheWrapper(_settings.REDIS_URL)
CACHE_TTL_SECONDS = 3600  # 1 hour TTL

# ------------------------------------------------------------------
# Utility: round all float values in nested dict/list structures to 2 decimals

def _round_floats(obj):
    """Recursively round all float values to 2 decimal places."""
    if isinstance(obj, float):
        return round(obj, 2)
    if isinstance(obj, list):
        return [_round_floats(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    return obj

def get_cache_key(file_contents: bytes, plate_diameter: float, reference_object: Optional[str]) -> str:
    """Generate cache key for analysis results"""
    key_data = file_contents + str(plate_diameter).encode() + (reference_object or "").encode()
    return hashlib.md5(key_data).hexdigest()

# Dependency injection
def get_image_processor(settings: Settings = Depends(get_settings)) -> ImageProcessor:
    return ImageProcessor(settings)

def get_ai_service(settings: Settings = Depends(get_settings)) -> AIService:
    return AIService(settings)

def get_nutrition_service(settings: Settings = Depends(get_settings)) -> MultiSourceNutritionService:
    return MultiSourceNutritionService(settings)

@app.post("/analyze-food", response_model=AnalysisResponse)
async def analyze_food(
    file: UploadFile = File(...),
    gemini_api_key: str = Form(...),
    usda_api_key: Optional[str] = Form(None),  # deprecated; kept for backward compatibility
    plate_diameter_mm: float = Form(250.0),
    reference_object: Optional[str] = Form(None),
    nutrient_base_amount: int = Form(100),
    image_processor: ImageProcessor = Depends(get_image_processor),
    ai_service: AIService = Depends(get_ai_service),
    nutrition_service: MultiSourceNutritionService = Depends(get_nutrition_service),
    settings: Settings = Depends(get_settings)
):
    """Analyze food image and provide nutritional information"""
    try:
        # Configure AI service
        ai_service.configure_api_key(gemini_api_key)
        
        # Read and validate image
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
            
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Check Redis cache
        cache_key = get_cache_key(contents, plate_diameter_mm, reference_object)
        cached_payload = None
        try:
            cached_payload = await redis_client.get(cache_key)
        except Exception as redis_err:
            # Redis not reachable – gracefully degrade to no-cache
            print(f"[WARN] Redis unavailable: {redis_err}. Proceeding without cache.")

        if cached_payload:
            try:
                return AnalysisResponse.model_validate_json(cached_payload)
            except Exception:
                # If parsing fails, ignore cache and proceed
                pass
            
        # Process image with CV
        cv_results = image_processor.detect_food_items(image, reference_object)
        if not cv_results:
            raise HTTPException(status_code=500, detail="Failed to process image")
            
        # Get AI analysis and USDA data concurrently
        ai_task = asyncio.create_task(
            ai_service.get_food_analysis(
                [{"mime_type": file.content_type, "data": contents}],
                cv_results=cv_results,
                nutrient_base_amount=nutrient_base_amount
            )
        )
        
        merged_data = {}
        if cv_results.get('food_instances'):
            tasks = [asyncio.create_task(nutrition_service.get_nutrition_info(item['type']))
                     for item in cv_results['food_instances']]
            results = await asyncio.gather(*tasks)
            merged_data = {item['type']: res for item, res in zip(cv_results['food_instances'], results) if res}
        
        # Get AI analysis result
        analysis = await ai_task
        
        # Enhance with USDA data if available
        if merged_data and analysis.food_items:
            for item in analysis.food_items:
                if item.name in merged_data:
                    merged_info = merged_data[item.name]
                    nutrients = merged_info.get('nutrients', {})
                    conversion_factor = nutrient_base_amount / 100.0
                    
                    # Update with USDA values
                    item.calories_per_base = nutrients.get('Energy', {}).get('amount', item.calories_per_base) * conversion_factor
                    item.nutrients_per_base.protein = nutrients.get('Protein', {}).get('amount', item.nutrients_per_base.protein) * conversion_factor
                    item.nutrients_per_base.carbs = nutrients.get('Carbohydrate, by difference', {}).get('amount', item.nutrients_per_base.carbs) * conversion_factor
                    item.nutrients_per_base.fat = nutrients.get('Total lipid (fat)', {}).get('amount', item.nutrients_per_base.fat) * conversion_factor
                    item.nutrients_per_base.fiber = nutrients.get('Fiber, total dietary', {}).get('amount', item.nutrients_per_base.fiber) * conversion_factor
        
        rounded_data = cast(Dict[str, Any], _round_floats(analysis.dict()))

        response = AnalysisResponse(
            success=True,
            data=rounded_data,
            error=None
        )
        
        # Cache response in Redis
        try:
            await redis_client.set(cache_key, response.json(), ex=CACHE_TTL_SECONDS)
        except Exception as e:
            print(f"[WARN] Failed to cache analysis result to Redis: {e}")
        
        return response
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root(settings: Settings = Depends(get_settings)):
    """API root endpoint"""
    return {
        "message": f"Welcome to {settings.APP_NAME} API",
        "version": settings.APP_VERSION,
        "endpoints": {
            "/analyze-food": {
                "method": "POST",
                "description": "Analyze food image and get nutritional information",
                "parameters": {
                    "file": "Image file (multipart/form-data)",
                    "gemini_api_key": "Your Gemini API key",
                    "usda_api_key": "Optional USDA Food Database API key",
                    "plate_diameter_mm": f"Optional plate diameter in mm (default: {settings.DEFAULT_PLATE_DIAMETER_MM})",
                    "reference_object": "Optional reference object for size calibration",
                    "nutrient_base_amount": f"Optional nutrient base amount in grams (default: {settings.DEFAULT_NUTRIENT_BASE_AMOUNT})"
                }
            }
        }
    } 