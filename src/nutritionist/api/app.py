from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from typing import Optional
import asyncio
from functools import lru_cache
import hashlib

from ..config.settings import Settings, get_settings
from ..services.image_processor import ImageProcessor
from ..services.ai_service import AIService
from ..services.usda_service import USDAService
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

# Cache for analysis results
analysis_cache = {}

def get_cache_key(file_contents: bytes, plate_diameter: float, reference_object: Optional[str]) -> str:
    """Generate cache key for analysis results"""
    key_data = file_contents + str(plate_diameter).encode() + (reference_object or "").encode()
    return hashlib.md5(key_data).hexdigest()

# Dependency injection
def get_image_processor(settings: Settings = Depends(get_settings)) -> ImageProcessor:
    return ImageProcessor(settings)

def get_ai_service(settings: Settings = Depends(get_settings)) -> AIService:
    return AIService(settings)

def get_usda_service(settings: Settings = Depends(get_settings)) -> USDAService:
    return USDAService(settings)

@app.post("/analyze-food", response_model=AnalysisResponse)
async def analyze_food(
    file: UploadFile = File(...),
    gemini_api_key: str = Form(...),
    usda_api_key: Optional[str] = Form(None),
    plate_diameter_mm: Optional[float] = Form(250.0),
    reference_object: Optional[str] = Form(None),
    nutrient_base_amount: Optional[int] = Form(100),
    image_processor: ImageProcessor = Depends(get_image_processor),
    ai_service: AIService = Depends(get_ai_service),
    usda_service: USDAService = Depends(get_usda_service),
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
        
        # Check cache
        cache_key = get_cache_key(contents, plate_diameter_mm, reference_object)
        if cache_key in analysis_cache:
            return analysis_cache[cache_key]
            
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
        
        usda_data = {}
        if usda_api_key and cv_results.get('food_instances'):
            usda_tasks = []
            for item in cv_results['food_instances']:
                usda_tasks.append(
                    asyncio.create_task(
                        usda_service.get_nutrition_info(item['type'], usda_api_key)
                    )
                )
            usda_results = await asyncio.gather(*usda_tasks)
            usda_data = {item['type']: result for item, result in zip(cv_results['food_instances'], usda_results) if result}
        
        # Get AI analysis result
        analysis = await ai_task
        
        # Enhance with USDA data if available
        if usda_data and analysis.food_items:
            for item in analysis.food_items:
                if item.name in usda_data:
                    usda_info = usda_data[item.name]
                    nutrients = usda_info.get('nutrients', {})
                    conversion_factor = nutrient_base_amount / 100.0
                    
                    # Update with USDA values
                    item.calories_per_base = nutrients.get('Energy', {}).get('amount', item.calories_per_base) * conversion_factor
                    item.nutrients_per_base.protein = nutrients.get('Protein', {}).get('amount', item.nutrients_per_base.protein) * conversion_factor
                    item.nutrients_per_base.carbs = nutrients.get('Carbohydrate, by difference', {}).get('amount', item.nutrients_per_base.carbs) * conversion_factor
                    item.nutrients_per_base.fat = nutrients.get('Total lipid (fat)', {}).get('amount', item.nutrients_per_base.fat) * conversion_factor
                    item.nutrients_per_base.fiber = nutrients.get('Fiber, total dietary', {}).get('amount', item.nutrients_per_base.fiber) * conversion_factor
        
        response = AnalysisResponse(
            success=True,
            data=analysis.dict(),
            error=None
        )
        
        # Cache the result
        analysis_cache[cache_key] = response
        
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