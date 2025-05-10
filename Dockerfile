# Stage 1: Build Stage
FROM python:3.9-slim AS build

# Create a virtual environment
RUN python -m venv /opt/venv
# Activate the virtual environment by updating PATH
ENV PATH="/opt/venv/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy only the necessary files
COPY requirements.txt app.py model_state.pth class_prototypes.pth /app/

# Install dependencies into the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime Stage
FROM python:3.9-slim

# Copy the virtual environment from the build stage
COPY --from=build /opt/venv /opt/venv
# Set PATH to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the application files from the build stage
COPY --from=build /app /app

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
