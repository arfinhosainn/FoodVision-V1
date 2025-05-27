import google.generativeai as genai
from typing import Dict, List, Optional
import json
from ..config.settings import Settings
from ..models.food import FoodAnalysis

class AIService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = genai.GenerativeModel('gemini-pro-vision')
        
    def configure_api_key(self, api_key: str):
        """Configure Gemini API key"""
        genai.configure(api_key=api_key)
        
    def get_food_analysis(
        self, 
        image_parts: List[Dict],
        cv_results: Optional[Dict] = None,
        nutrient_base_amount: int = 100
    ) -> FoodAnalysis:
        """Get food analysis from Gemini Vision API"""
        try:
            # Enhanced prompt incorporating CV results
            cv_analysis_text = self._format_cv_results(cv_results) if cv_results else "No CV pre-analysis available."
            
            enhanced_prompt = f"""
            You are an expert nutritionist and food analyst. Analyze this food image with exceptional detail and precision.
            
            Computer Vision Pre-Analysis:
            {cv_analysis_text}
            
            Please provide a detailed analysis in JSON format with the following structure:
            {{
                "food_items": [
                    {{
                        "name": "item name",
                        "category": "meat/vegetable/grain/etc",
                        "portion_size": "estimated portion in grams",
                        "calories_per_{nutrient_base_amount}g": number,
                        "nutrients_per_{nutrient_base_amount}g": {{
                            "protein": number,
                            "carbs": number,
                            "fat": number,
                            "fiber": number
                        }},
                        "preparation_method": "description",
                        "estimated_calories": number,
                        "actual_protein": number,
                        "actual_carbs": number,
                        "actual_fat": number
                    }}
                ],
                "meal_analysis": {{
                    "total_calories": number,
                    "total_protein": number,
                    "total_carbs": number,
                    "total_fat": number,
                    "meal_type": "breakfast/lunch/dinner/snack",
                    "portion_size_analysis": "description",
                    "health_score": number (0-10)
                }},
                "dietary_considerations": {{
                    "allergens": ["list of potential allergens"],
                    "dietary_restrictions": ["list of relevant restrictions"],
                    "health_implications": ["list of health implications"]
                }}
            }}
            
            Focus on accuracy and provide realistic nutritional values based on visible portions.
            """
            
            # Generate response
            response = self.model.generate_content([enhanced_prompt, *image_parts])
            
            # Extract and parse JSON
            try:
                json_str = response.text.split('```json\n')[1].split('\n```')[0]
                analysis_dict = json.loads(json_str)
            except:
                analysis_dict = json.loads(response.text)
            
            # Convert to FoodAnalysis model
            return FoodAnalysis(**analysis_dict)
            
        except Exception as e:
            raise ValueError(f"Error in Gemini analysis: {str(e)}")
            
    def _format_cv_results(self, cv_results: Dict) -> str:
        """Format CV results for prompt enhancement"""
        if not cv_results or not cv_results.get('food_instances'):
            return "No food instances detected."
            
        formatted = ["Detected food items:"]
        for instance in cv_results['food_instances']:
            item_desc = f"- {instance['type']}"
            if instance.get('volume_cm3'):
                item_desc += f" (estimated volume: {instance['volume_cm3']:.1f}cm³)"
            if instance.get('confidence'):
                item_desc += f" (confidence: {instance['confidence']:.2f})"
            formatted.append(item_desc)
            
        if cv_results.get('calibration'):
            formatted.append(f"\nSize calibration: Using reference object with "
                          f"{cv_results['calibration']['reference_size_mm']}mm size.")
            
        return "\n".join(formatted) 