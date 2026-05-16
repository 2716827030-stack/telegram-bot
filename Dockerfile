FROM python:3.12-slim

# Install system dependencies required by OpenCV and av (video processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY bot/ bot/
COPY run.py .

# Create temp directory for steganography output
RUN mkdir -p bot/temp

CMD ["python", "run.py"]
