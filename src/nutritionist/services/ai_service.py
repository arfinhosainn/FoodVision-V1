import google.generativeai as genai
from typing import Dict, List, Optional
import json
from ..config.settings import Settings
from ..models.food import FoodAnalysis, FoodItem, MealAnalysis, DietaryConsiderations, NutrientInfo


class AIService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')

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
                        "calories_per_base": number,
                        "nutrients_per_base": {{
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

            IMPORTANT: Use exactly "calories_per_base" and "nutrients_per_base" as field names.
            The base amount is {nutrient_base_amount}g.
            Focus on accuracy and provide realistic nutritional values based on visible portions.
            """

            # Generate response
            response = self.model.generate_content([enhanced_prompt, *image_parts])

            # Extract and parse JSON
            try:
                if '```json' in response.text:
                    json_str = response.text.split('```json\n')[1].split('\n```')[0]
                else:
                    json_str = response.text
                analysis_dict = json.loads(json_str)
            except (IndexError, json.JSONDecodeError) as e:
                # Fallback parsing
                try:
                    analysis_dict = json.loads(response.text)
                except json.JSONDecodeError:
                    # Create fallback response
                    analysis_dict = self._create_fallback_response()

            # Validate and fix the structure
            analysis_dict = self._fix_response_structure(analysis_dict, nutrient_base_amount)

            # Convert to FoodAnalysis model
            return FoodAnalysis(**analysis_dict)

        except Exception as e:
            raise ValueError(f"Error in Gemini analysis: {str(e)}")

    def _fix_response_structure(self, analysis_dict: Dict, nutrient_base_amount: int) -> Dict:
        """Fix common structure issues in API response"""
        # Fix food items
        if 'food_items' in analysis_dict:
            for item in analysis_dict['food_items']:
                # Fix field name mismatches
                if f'calories_per_{nutrient_base_amount}g' in item:
                    item['calories_per_base'] = item.pop(f'calories_per_{nutrient_base_amount}g')
                elif 'calories_per_100g' in item:
                    item['calories_per_base'] = item.pop('calories_per_100g')
                elif 'calories_per_base' not in item:
                    item['calories_per_base'] = 0.0

                # Fix nutrients structure
                nutrients_key = f'nutrients_per_{nutrient_base_amount}g'
                if nutrients_key in item:
                    item['nutrients_per_base'] = item.pop(nutrients_key)
                elif 'nutrients_per_100g' in item:
                    item['nutrients_per_base'] = item.pop('nutrients_per_100g')
                elif 'nutrients_per_base' not in item:
                    item['nutrients_per_base'] = {
                        'protein': 0.0,
                        'carbs': 0.0,
                        'fat': 0.0,
                        'fiber': 0.0
                    }

                # Ensure all required fields exist
                defaults = {
                    'name': 'Unknown Food',
                    'category': 'unknown',
                    'portion_size': '100g',
                    'preparation_method': 'unknown',
                    'estimated_calories': 0.0,
                    'actual_protein': 0.0,
                    'actual_carbs': 0.0,
                    'actual_fat': 0.0
                }

                for key, default_value in defaults.items():
                    if key not in item:
                        item[key] = default_value

        # Ensure all top-level sections exist
        if 'meal_analysis' not in analysis_dict:
            analysis_dict['meal_analysis'] = {
                'total_calories': 0.0,
                'total_protein': 0.0,
                'total_carbs': 0.0,
                'total_fat': 0.0,
                'meal_type': 'unknown',
                'portion_size_analysis': 'Unable to analyze',
                'health_score': 5.0
            }

        if 'dietary_considerations' not in analysis_dict:
            analysis_dict['dietary_considerations'] = {
                'allergens': [],
                'dietary_restrictions': [],
                'health_implications': []
            }

        return analysis_dict

    def _create_fallback_response(self) -> Dict:
        """Create a fallback response when parsing fails"""
        return {
            'food_items': [{
                'name': 'Unknown Food Item',
                'category': 'unknown',
                'portion_size': '100g',
                'calories_per_base': 0.0,
                'nutrients_per_base': {
                    'protein': 0.0,
                    'carbs': 0.0,
                    'fat': 0.0,
                    'fiber': 0.0
                },
                'preparation_method': 'unknown',
                'estimated_calories': 0.0,
                'actual_protein': 0.0,
                'actual_carbs': 0.0,
                'actual_fat': 0.0
            }],
            'meal_analysis': {
                'total_calories': 0.0,
                'total_protein': 0.0,
                'total_carbs': 0.0,
                'total_fat': 0.0,
                'meal_type': 'unknown',
                'portion_size_analysis': 'Analysis failed',
                'health_score': 0.0
            },
            'dietary_considerations': {
                'allergens': [],
                'dietary_restrictions': [],
                'health_implications': ['Unable to analyze due to processing error']
            }
        }

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