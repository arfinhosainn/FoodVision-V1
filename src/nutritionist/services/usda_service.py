import requests
from typing import Dict, Optional
from ..config.settings import Settings

class USDAService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        
    def get_nutrition_info(self, food_name: str, api_key: str) -> Optional[Dict]:
        """Get nutritional information from USDA Food Database"""
        try:
            # Search for food
            search_url = f"{self.base_url}/foods/search"
            params = {
                "api_key": api_key,
                "query": food_name,
                "dataType": ["SR Legacy", "Foundation", "Survey (FNDDS)"],
                "pageSize": 1
            }
            
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if not data.get('foods'):
                return None
                
            food = data['foods'][0]
            nutrients = {}
            
            # Map nutrient IDs to names
            nutrient_map = {
                1008: "Energy",  # kcal
                1003: "Protein",  # g
                1005: "Carbohydrate, by difference",  # g
                1004: "Total lipid (fat)",  # g
                1079: "Fiber, total dietary"  # g
            }
            
            # Extract nutrients
            for nutrient in food.get('foodNutrients', []):
                nutrient_id = nutrient.get('nutrientId')
                if nutrient_id in nutrient_map:
                    nutrients[nutrient_map[nutrient_id]] = {
                        "amount": nutrient.get('value', 0),
                        "unit": nutrient.get('unitName', 'g')
                    }
            
            return {
                "food_name": food.get('description', food_name),
                "fdc_id": food.get('fdcId'),
                "nutrients": nutrients
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching USDA data: {str(e)}")
            return None
        except Exception as e:
            print(f"Error processing USDA data: {str(e)}")
            return None 