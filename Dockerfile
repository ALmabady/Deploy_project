# Stage 1: Build Stage
FROM python:3.9-slim AS build

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools for PIL and PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy only necessary files
COPY requirements.txt app.py model_state.pth class_prototypes.pth /app/

# Install CPU-only PyTorch and dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir torch==1.12.1+cpu torchvision==0.13.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
 && pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime Stage
FROM python:3.9-slim

# Set up virtual environment
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app
COPY --from=build /app /app

# Expose port (optional but helps on some hosts)
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
