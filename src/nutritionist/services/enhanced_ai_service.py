import google.generativeai as genai
from typing import Dict, List, Optional
import json
import hashlib
import statistics
from datetime import datetime, timedelta

from ..config.settings import Settings
from ..models.food import FoodAnalysis, FoodItem, MealAnalysis, DietaryConsiderations, NutrientInfo


class EnhancedAIService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = genai.GenerativeModel('gemini-2.0-flash')

        # Enhanced nutrition database for validation
        self.nutrition_database = {
            # Proteins (per 100g)
            'chicken breast': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6, 'fiber': 0},
            'beef': {'calories': 250, 'protein': 26, 'carbs': 0, 'fat': 15, 'fiber': 0},
            'salmon': {'calories': 208, 'protein': 20, 'carbs': 0, 'fat': 13, 'fiber': 0},
            'egg': {'calories': 155, 'protein': 13, 'carbs': 1.1, 'fat': 11, 'fiber': 0},
            'tofu': {'calories': 76, 'protein': 8, 'carbs': 1.9, 'fat': 4.8, 'fiber': 0.3},
            # Carbohydrates
            'white rice': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3, 'fiber': 0.4},
            'brown rice': {'calories': 123, 'protein': 2.6, 'carbs': 23, 'fat': 0.9, 'fiber': 1.8},
            'pasta': {'calories': 131, 'protein': 5, 'carbs': 25, 'fat': 1.1, 'fiber': 1.8},
            'bread': {'calories': 265, 'protein': 9, 'carbs': 49, 'fat': 3.2, 'fiber': 2.7},
            'potato': {'calories': 77, 'protein': 2, 'carbs': 17, 'fat': 0.1, 'fiber': 2.2},
            'sweet potato': {'calories': 86, 'protein': 1.6, 'carbs': 20, 'fat': 0.1, 'fiber': 3},
            # Vegetables
            'broccoli': {'calories': 34, 'protein': 2.8, 'carbs': 7, 'fat': 0.4, 'fiber': 2.6},
            'spinach': {'calories': 23, 'protein': 2.9, 'carbs': 3.6, 'fat': 0.4, 'fiber': 2.2},
            'carrots': {'calories': 41, 'protein': 0.9, 'carbs': 10, 'fat': 0.2, 'fiber': 2.8},
            'bell pepper': {'calories': 31, 'protein': 1, 'carbs': 7, 'fat': 0.3, 'fiber': 2.5},
            'tomato': {'calories': 18, 'protein': 0.9, 'carbs': 3.9, 'fat': 0.2, 'fiber': 1.2},
            'cucumber': {'calories': 16, 'protein': 0.7, 'carbs': 4, 'fat': 0.1, 'fiber': 0.5},
            'lettuce': {'calories': 15, 'protein': 1.4, 'carbs': 2.9, 'fat': 0.2, 'fiber': 1.3},
            # Fruits
            'apple': {'calories': 52, 'protein': 0.3, 'carbs': 14, 'fat': 0.2, 'fiber': 2.4},
            'banana': {'calories': 89, 'protein': 1.1, 'carbs': 23, 'fat': 0.3, 'fiber': 2.6},
            'orange': {'calories': 47, 'protein': 0.9, 'carbs': 12, 'fat': 0.1, 'fiber': 2.4},
            'strawberry': {'calories': 32, 'protein': 0.7, 'carbs': 8, 'fat': 0.3, 'fiber': 2},
            'avocado': {'calories': 160, 'protein': 2, 'carbs': 9, 'fat': 15, 'fiber': 7},
            # Dairy
            'milk': {'calories': 42, 'protein': 3.4, 'carbs': 5, 'fat': 1, 'fiber': 0},
            'cheese': {'calories': 113, 'protein': 7, 'carbs': 1, 'fat': 9, 'fiber': 0},
            'yogurt': {'calories': 59, 'protein': 10, 'carbs': 3.6, 'fat': 0.4, 'fiber': 0},
            # Nuts and Seeds
            'almonds': {'calories': 579, 'protein': 21, 'carbs': 22, 'fat': 50, 'fiber': 12},
            'walnuts': {'calories': 654, 'protein': 15, 'carbs': 14, 'fat': 65, 'fiber': 7},
            'peanuts': {'calories': 567, 'protein': 26, 'carbs': 16, 'fat': 49, 'fiber': 9},
            # Oils and Fats
            'olive oil': {'calories': 884, 'protein': 0, 'carbs': 0, 'fat': 100, 'fiber': 0},
            'butter': {'calories': 717, 'protein': 0.9, 'carbs': 0.1, 'fat': 81, 'fiber': 0},
        }

        # Cache
        self.analysis_cache: Dict[str, Dict] = {}
        self.cache_duration = timedelta(hours=24)

    def configure_api_key(self, api_key: str):
        genai.configure(api_key=api_key)

    async def get_food_analysis(
        self,
        image_parts: List[Dict],
        cv_results: Optional[Dict] = None,
        nutrient_base_amount: int = 100,
        user_specified_amount: Optional[float] = None,
        consistency_checks: bool = True
    ) -> FoodAnalysis:
        try:
            cache_key = self._generate_cache_key(image_parts, cv_results, nutrient_base_amount, user_specified_amount)
            if cache_key in self.analysis_cache:
                entry = self.analysis_cache[cache_key]
                if datetime.now() - entry['timestamp'] < self.cache_duration:
                    return entry['analysis']

            if consistency_checks:
                analyses = []
                for i in range(3):
                    analyses.append(
                        await self._single_analysis_round(
                            image_parts, cv_results, nutrient_base_amount, user_specified_amount, i
                        )
                    )
                final_analysis = self._combine_analyses(analyses, user_specified_amount)
            else:
                final_analysis = await self._single_analysis_round(
                    image_parts, cv_results, nutrient_base_amount, user_specified_amount, 0
                )

            self.analysis_cache[cache_key] = {'analysis': final_analysis, 'timestamp': datetime.now()}
            return final_analysis
        except Exception as e:
            raise ValueError(f"Error in enhanced food analysis: {str(e)}")

    async def _single_analysis_round(
        self,
        image_parts: List[Dict],
        cv_results: Optional[Dict],
        nutrient_base_amount: int,
        user_specified_amount: Optional[float],
        round_number: int
    ) -> FoodAnalysis:
        cv_text = self._format_cv_results(cv_results) if cv_results else "No CV pre-analysis available."
        # Build detailed system/user prompt instructing Gemini to return structured JSON
        prompt = f"""
You are a world-class, board-certified nutritionist and computer-vision expert.  Given:
1. A set of food images (as binary parts sent separately).
2. A computer-vision pre-analysis of those images (below).
3. The required nutrient base amount of {nutrient_base_amount} g.

Your task:
• Detect every distinct food item present.
• For each item fill in the following fields exactly as specified (no extra keys):
  name, category (protein | carbohydrate | vegetable | fruit | fat | dairy | other),
  portion_size (human readable, e.g. "150g", "2 slices"),
  calories_per_base, nutrients_per_base {{protein, carbs, fat, fiber}},
  preparation_method (e.g. grilled, boiled, raw, fried),
  estimated_calories (scaled to portion_size), actual_protein, actual_carbs, actual_fat.
• Build meal_analysis totals as sums of the items (total_calories, total_protein, total_carbs, total_fat) and include meal_type (breakfast|lunch|dinner|snack|unknown), portion_size_analysis ("auto" or "Fallback"), health_score (0-10 where 10 is healthiest).
• Build dietary_considerations as follows (each must be an *array of strings*):
  - allergens: list any of the major 14 allergens present (e.g., egg, milk, peanuts, tree nuts, fish, shellfish, soy, wheat, sesame, mustard, lupin, sulphites, celery, molluscs).
  - dietary_restrictions: list which common diets the meal is NOT suitable for (e.g., vegan, vegetarian, gluten-free, keto, halal, kosher, low-sodium, low-fat, paleo).
  - health_implications: provide AT LEAST three concise evidence-based statements (positive or negative) about health impact (e.g., "High in saturated fat", "Excellent source of omega-3", "Exceeds daily sodium limit").
• Set confidence_level between 0 and 1 indicating overall certainty.

Computer-vision analysis:
{cv_text}

Output format:
Respond with **ONLY** a JSON object that can be parsed by the following Pydantic model (keys in exact order are NOT required, but names must match):
FoodAnalysis = {{
  "timestamp": "ISO-8601 timestamp",
  "food_items": [FoodItem, ...],
  "meal_analysis": MealAnalysis,
  "dietary_considerations": DietaryConsiderations,
  "confidence_level": float
}}
(Where FoodItem, MealAnalysis, NutrientInfo, DietaryConsiderations follow the same field names given above.)

DO NOT wrap the JSON in markdown back-ticks or add any commentary – return the JSON only.
"""
        response = await self.model.generate_content_async([prompt, *image_parts])
        return self._parse_and_validate_response(response.text, nutrient_base_amount, user_specified_amount)

    def _parse_and_validate_response(self, response_text: str, nutrient_base_amount: int, user_specified_amount: Optional[float]) -> FoodAnalysis:
        try:
            if '```json' in response_text:
                json_str = response_text.split('```json\n')[1].split('\n```')[0]
            else:
                json_str = response_text
            analysis_dict = json.loads(json_str)
        except Exception:
            analysis_dict = self._create_enhanced_fallback_response(nutrient_base_amount, user_specified_amount).dict()

        analysis_dict = self._validate_and_correct_analysis(analysis_dict, nutrient_base_amount, user_specified_amount)
        return FoodAnalysis(**analysis_dict)

    # ------------------------------------------------------------------
    # Helper methods added below

    def _generate_cache_key(self, image_parts: List[Dict], cv_results: Optional[Dict], nutrient_base_amount: int, user_specified_amount: Optional[float]) -> str:
        key_data = str(image_parts) + str(cv_results) + str(nutrient_base_amount) + str(user_specified_amount)
        return hashlib.md5(key_data.encode()).hexdigest()

    def _format_cv_results(self, cv_results: Dict) -> str:
        if not cv_results or not cv_results.get('food_instances'):
            return "No food instances detected."
        parts = ["Detected food items with computer vision:"]
        for inst in cv_results['food_instances']:
            desc = f"- {inst['type']}"
            if inst.get('volume_cm3'):
                desc += f" (estimated volume: {inst['volume_cm3']:.1f}cm³)"
            if inst.get('confidence'):
                desc += f" (detection confidence: {inst['confidence']:.2f})"
            parts.append(desc)
        if cv_results.get('calibration'):
            parts.append(
                f"\nSize calibration available: Reference object detected with {cv_results['calibration']['reference_size_mm']}mm size for accurate portion estimation."
            )
        return "\n".join(parts)

    # ------------------ VALIDATION & CORRECTION ----------------------
    def _validate_and_correct_analysis(self, analysis_dict: Dict, nutrient_base_amount: int, user_specified_amount: Optional[float]) -> Dict:
        if 'food_items' in analysis_dict:
            for item in analysis_dict['food_items']:
                self._fix_field_names(item, nutrient_base_amount)
                self._validate_against_database(item)
                self._enforce_reasonable_ranges(item)
                if user_specified_amount:
                    self._adjust_for_user_amount(item, user_specified_amount)
        self._validate_meal_analysis(analysis_dict)
        self._add_missing_fields(analysis_dict)
        return analysis_dict

    def _fix_field_names(self, item: Dict, nutrient_base_amount: int):
        cal_variants = [f'calories_per_{nutrient_base_amount}g', 'calories_per_100g', 'calories_per_serving']
        for v in cal_variants:
            if v in item:
                item['calories_per_base'] = item.pop(v)
                break
        nutr_variants = [f'nutrients_per_{nutrient_base_amount}g', 'nutrients_per_100g', 'nutrients_per_serving']
        for v in nutr_variants:
            if v in item:
                item['nutrients_per_base'] = item.pop(v)
                break

    def _validate_against_database(self, item: Dict):
        name = item.get('name', '').lower()
        best_match = None; best_score = 0
        for db_food, db_vals in self.nutrition_database.items():
            if db_food in name:
                score = len(set(db_food.split()) & set(name.split()))
                if score > best_score:
                    best_score = score
                    best_match = db_vals
        if best_match:
            if abs(item.get('calories_per_base', 0) - best_match['calories']) > best_match['calories'] * 0.5:
                item['calories_per_base'] = best_match['calories']
            nutrients = item.get('nutrients_per_base', {})
            for n in ['protein', 'carbs', 'fat', 'fiber']:
                if n in nutrients and abs(nutrients[n] - best_match[n]) > best_match[n] * 0.5:
                    nutrients[n] = best_match[n]

    def _enforce_reasonable_ranges(self, item: Dict):
        ranges = {
            'protein': (100, 400),
            'carbohydrate': (50, 400),
            'vegetable': (10, 100),
            'fruit': (30, 200),
            'dairy': (30, 400),
            'fat': (200, 900),
            'other': (50, 500)
        }
        cat = item.get('category', 'other')
        min_cal, max_cal = ranges.get(cat, (50, 500))
        item['calories_per_base'] = max(min_cal, min(max_cal, item.get('calories_per_base', 0)))
        nutrients = item.get('nutrients_per_base', {})
        protein = nutrients.get('protein', 0)
        carbs = nutrients.get('carbs', 0)
        fat = nutrients.get('fat', 0)
        macro_cals = protein*4 + carbs*4 + fat*9
        if macro_cals > 0:
            ratio = item['calories_per_base'] / macro_cals
            if ratio < 0.8 or ratio > 1.2:
                nutrients['protein'] = protein * ratio
                nutrients['carbs'] = carbs * ratio
                nutrients['fat'] = fat * ratio

    def _adjust_for_user_amount(self, item: Dict, user_amount: float):
        weight = self._extract_weight_from_string(item.get('portion_size', '100g'))
        if weight:
            scale = user_amount / weight
            for field in ['estimated_calories', 'actual_protein', 'actual_carbs', 'actual_fat']:
                if field in item:
                    item[field] = item[field] * scale
            item['portion_size'] = f"{user_amount}g"

    def _extract_weight_from_string(self, s: str) -> float:
        import re
        m = re.search(r'(\d+\.?\d*)\s*g', s.lower())
        if m:
            return float(m.group(1))
        m = re.search(r'(\d+\.?\d*)\s*oz', s.lower())
        if m:
            return float(m.group(1))*28.35
        m = re.search(r'(\d+\.?\d*)\s*lb', s.lower())
        if m:
            return float(m.group(1))*453.592
        return 0.0

    def _validate_meal_analysis(self, analysis_dict: Dict):
        if 'meal_analysis' not in analysis_dict:
            analysis_dict['meal_analysis'] = {}
        meal = analysis_dict['meal_analysis']
        items = analysis_dict.get('food_items', [])
        meal['total_calories'] = sum(item.get('estimated_calories', 0) for item in items)
        meal['total_protein'] = sum(item.get('actual_protein', 0) for item in items)
        meal['total_carbs'] = sum(item.get('actual_carbs', 0) for item in items)
        meal['total_fat'] = sum(item.get('actual_fat', 0) for item in items)
        meal.setdefault('meal_type', 'unknown')
        meal.setdefault('portion_size_analysis', 'auto')
        meal.setdefault('health_score', 5.0)

    def _add_missing_fields(self, analysis_dict: Dict):
        analysis_dict.setdefault('dietary_considerations', {
            'allergens': [], 'dietary_restrictions': [], 'health_implications': []
        })

    # ---------------------- COMBINE ANALYSES -------------------------
    def _combine_analyses(self, analyses: List[FoodAnalysis], user_specified_amount: Optional[float]) -> FoodAnalysis:
        if not analyses:
            return self._create_enhanced_fallback_response(100, user_specified_amount)
        base = analyses[0]
        if len(analyses) > 1:
            for idx, item in enumerate(base.food_items):
                cal_values = [a.food_items[idx].calories_per_base for a in analyses if idx < len(a.food_items)]
                prot_vals = [a.food_items[idx].nutrients_per_base.protein for a in analyses if idx < len(a.food_items)]
                carb_vals = [a.food_items[idx].nutrients_per_base.carbs for a in analyses if idx < len(a.food_items)]
                fat_vals = [a.food_items[idx].nutrients_per_base.fat for a in analyses if idx < len(a.food_items)]
                if cal_values:
                    item.calories_per_base = statistics.median(cal_values)
                    item.nutrients_per_base.protein = statistics.median(prot_vals)
                    item.nutrients_per_base.carbs = statistics.median(carb_vals)
                    item.nutrients_per_base.fat = statistics.median(fat_vals)
                    w = self._extract_weight_from_string(item.portion_size)
                    scale = w/100 if w else 1
                    item.estimated_calories = item.calories_per_base*scale
                    item.actual_protein = item.nutrients_per_base.protein*scale
                    item.actual_carbs = item.nutrients_per_base.carbs*scale
                    item.actual_fat = item.nutrients_per_base.fat*scale
        base.meal_analysis.total_calories = sum(fi.estimated_calories for fi in base.food_items)
        base.meal_analysis.total_protein = sum(fi.actual_protein for fi in base.food_items)
        base.meal_analysis.total_carbs = sum(fi.actual_carbs for fi in base.food_items)
        base.meal_analysis.total_fat = sum(fi.actual_fat for fi in base.food_items)
        return base

    # ---------------------- FALLBACK --------------------------------
    def _create_enhanced_fallback_response(self, nutrient_base_amount: int, user_specified_amount: Optional[float]) -> FoodAnalysis:
        portion = f"{user_specified_amount}g" if user_specified_amount else "100g"
        scale = (user_specified_amount/100) if user_specified_amount else 1
        return FoodAnalysis(
            food_items=[FoodItem(
                name='Unknown Food Item',
                category='other',
                portion_size=portion,
                calories_per_base=150.0,
                nutrients_per_base=NutrientInfo(protein=8.0, carbs=20.0, fat=5.0, fiber=2.0),
                preparation_method='unknown',
                estimated_calories=150.0*scale,
                actual_protein=8.0*scale,
                actual_carbs=20.0*scale,
                actual_fat=5.0*scale
            )],
            meal_analysis=MealAnalysis(
                total_calories=150.0*scale,
                total_protein=8.0*scale,
                total_carbs=20.0*scale,
                total_fat=5.0*scale,
                meal_type='unknown',
                portion_size_analysis='Fallback',
                health_score=5.0
            ),
            dietary_considerations=DietaryConsiderations(allergens=[], dietary_restrictions=[], health_implications=[]),
            confidence_level=0.3
        ) 