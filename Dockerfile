# Stage 1: Build Stage
FROM python:3.9-slim AS build

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Install dependencies (without cache)
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime Stage
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy only the necessary files from the build stage
COPY --from=build /app /app

# Install only the necessary runtime dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
