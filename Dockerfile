# Start with a minimal Python base
FROM python:3.10-slim

# Set environment variables to reduce cache usage
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install with no cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Set entrypoint
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
