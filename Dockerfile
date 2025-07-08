FROM python:3.9.18-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# increase pip timeout and upgrade pip first to reduce wheel download failures
ENV PIP_DEFAULT_TIMEOUT=100
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Make port 8000 available
EXPOSE 8000

# Set environment variable for port
ENV PORT=8000

CMD ["python", "-m", "src.nutritionist"] 