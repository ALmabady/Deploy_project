# Stage 1: Build
FROM python:3.9-slim AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt app.py model_state.pth class_prototypes.pth /app/

RUN pip install --upgrade pip \
 && pip install --no-cache-dir torch==1.12.1+cpu torchvision==0.13.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
 && pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY --from=build /app /app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
