from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class NutrientInfo(BaseModel):
    """Nutrient information per base amount"""
    protein: float = Field(..., description="Protein content in grams")
    carbs: float = Field(..., description="Carbohydrate content in grams")
    fat: float = Field(..., description="Fat content in grams")
    fiber: float = Field(..., description="Fiber content in grams")

class FoodItem(BaseModel):
    """Individual food item analysis"""
    name: str
    category: str
    portion_size: str
    calories_per_base: float
    nutrients_per_base: NutrientInfo
    preparation_method: str
    estimated_calories: float
    actual_protein: float
    actual_carbs: float
    actual_fat: float

class MealAnalysis(BaseModel):
    """Overall meal analysis"""
    total_calories: float
    total_protein: float
    total_carbs: float
    total_fat: float
    meal_type: str
    portion_size_analysis: str
    health_score: float

class DietaryConsiderations(BaseModel):
    """Dietary considerations and restrictions"""
    allergens: List[str]
    dietary_restrictions: List[str]
    health_implications: List[str]

class FoodAnalysis(BaseModel):
    """Complete food analysis response"""
    timestamp: datetime = Field(default_factory=datetime.now)
    food_items: List[FoodItem]
    meal_analysis: MealAnalysis
    dietary_considerations: DietaryConsiderations
    confidence_level: float = Field(default=0.0, ge=0.0, le=1.0)

class AnalysisResponse(BaseModel):
    """API response model"""
    success: bool
    data: Dict
    error: Optional[str] = None 