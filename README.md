# FoodVision-v1

An AI-powered food analysis application that combines computer vision and natural language processing to provide detailed nutritional information from food images.

## Features

- Food detection and segmentation using computer vision
- Nutritional analysis using Gemini Vision API
- USDA Food Database integration for accurate nutritional data
- Size calibration using reference objects
- Configurable nutrient base amounts (per 100g, 200g, etc.)
- Detailed meal analysis with macronutrient breakdowns
- Dietary considerations and health implications

## Project Structure

```
src/nutritionist/
├── api/
│   └── app.py           # FastAPI application
├── core/
│   └── __init__.py
├── services/
│   ├── ai_service.py    # Gemini Vision API integration
│   ├── image_processor.py # Computer vision processing
│   └── usda_service.py  # USDA Food Database integration
├── models/
│   └── food.py          # Data models
├── utils/
│   └── __init__.py
├── config/
│   └── settings.py      # Application settings
└── __main__.py          # Entry point

```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   USDA_API_KEY=your_usda_api_key
   ```

4. Run the application:
   ```bash
   python -m src.nutritionist
   ```

The API will be available at `http://localhost:8000`

## API Usage

### Analyze Food Image

```bash
curl -X POST "http://localhost:8000/analyze-food" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@food_image.jpg" \
  -F "gemini_api_key=your_gemini_api_key" \
  -F "usda_api_key=your_usda_api_key" \
  -F "nutrient_base_amount=100"
```

See the API documentation at `http://localhost:8000/docs` for more details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
