import aiohttp
from typing import Dict, Optional

from ..config.settings import Settings
from .usda_service import USDAService


class MultiSourceNutritionService:
    """Query multiple public nutrition databases concurrently and merge results."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.usda = USDAService(settings)

    async def get_nutrition_info(self, food_name: str) -> Optional[Dict]:
        """Return merged nutrient dict for a given food name."""
        tasks = []
        # USDA (if API key)
        if self.settings.USDA_API_KEY:
            tasks.append(self.usda.get_nutrition_info(food_name, self.settings.USDA_API_KEY))

        # Nutritionix
        if self.settings.NUTRITIONIX_APP_ID and self.settings.NUTRITIONIX_APP_KEY:
            tasks.append(self._nutritionix_request(food_name))

        if not tasks:
            return None

        results = await aiohttp.asyncio.gather(*tasks, return_exceptions=True)  # type: ignore[attr-defined]
        # Filter valid dicts
        valid = [r for r in results if isinstance(r, dict) and r.get("nutrients")]
        if not valid:
            return None

        # Simple merge: choose the dict with most nutrients; fallback to first.
        best = max(valid, key=lambda d: len(d["nutrients"]))
        return best

    # ---------------------- private ------------------------
    async def _nutritionix_request(self, food_name: str) -> Optional[Dict]:
        url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
        headers = {
            "x-app-id": self.settings.NUTRITIONIX_APP_ID,
            "x-app-key": self.settings.NUTRITIONIX_APP_KEY,
            "Content-Type": "application/json",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json={"query": food_name}) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            foods = data.get("foods")
            if not foods:
                return None
            item = foods[0]
            nutrients = {
                "Energy": {"amount": item.get("nf_calories", 0), "unit": "kcal"},
                "Protein": {"amount": item.get("nf_protein", 0), "unit": "g"},
                "Carbohydrate, by difference": {"amount": item.get("nf_total_carbohydrate", 0), "unit": "g"},
                "Total lipid (fat)": {"amount": item.get("nf_total_fat", 0), "unit": "g"},
                "Fiber, total dietary": {"amount": item.get("nf_dietary_fiber", 0), "unit": "g"},
            }
            return {"food_name": item.get("food_name", food_name), "nutrients": nutrients}
        except Exception as e:
            print(f"Nutritionix error: {e}")
            return None 