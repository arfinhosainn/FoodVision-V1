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

## Deploying on Fly.io

Fly.io gives you an always-on micro-VM that avoids cold-starts. Once you've installed the Fly CLI (`brew install flyctl` on macOS):

```bash
# 1) Launch – this creates the app and writes a "fly.toml" (we committed a starter one)
flyctl launch --dockerfile Dockerfile --name foodvision-api --region ord --now

# 2) Add secrets (replace with your real keys)
flyctl secrets set GEMINI_API_KEY=xxxxx USDA_API_KEY=yyyy NUTRITIONIX_APP_ID=zzz NUTRITIONIX_APP_KEY=kkkk

# 3) (Optional) Use free Upstash Redis rather than localhost
flyctl secrets set REDIS_URL="redis://:<password>@<subdomain>.upstash.io:6379"

# 4) Subsequent updates
flyctl deploy
```

Things worth knowing:

* The Dockerfile already exposes `PORT 8000`; Fly automatically maps it to ports 80/443 outside.
* The free allocation (3 × shared-cpu-1x VMs, 256 MB each) is enough for CPU-only YOLOv8n if you disable CUDA and quantize weights.
* If you need more RAM (`torch + ultralytics` can reach ~300–350 MB resident) switch the VM size:
  ```bash
  flyctl scale memory 512      # 512 MB, $2.41/mo when running 24 × 7
  flyctl scale vm shared-cpu-1x # cheapest CPU class
  ```
* Logs in real time: `flyctl logs -a foodvision-api`.
* Add Postgres / Redis with `flyctl postgres create` or Upstash.
