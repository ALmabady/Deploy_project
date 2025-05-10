# Stage 1: Build
FROM python:3.9-slim AS build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app

COPY requirements.txt app.py model_state.pth class_prototypes.pth /app/
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY --from=build /app /app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
