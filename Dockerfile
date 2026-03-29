FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# HuggingFace Spaces expects port 7860
EXPOSE 7860

# Health check so HF Space knows when the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Start FastAPI server on port 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
